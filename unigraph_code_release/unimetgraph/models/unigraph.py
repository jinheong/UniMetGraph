"""
UniGraph: Hybrid model combining UnitedMet's rank-based Bayesian covariation
with GAZE's metabolic network topology for transcriptome-to-metabolome prediction.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
import pyro.poutine as poutine
from typing import Dict, Optional, Tuple
from torch_geometric.nn import GATv2Conv

sys.path.insert(0, '/workspace/UnitedMet')
from Performance_Benchmarking.scripts.utils import smart_perm_2D

from unigraph.data.preprocess import (
    rank_predictions_per_batch, log_transform_rna, normalize_rna
)


class PlackettLuce_2D(TorchDistribution):
    """Plackett-Luce distribution for 2D permutation matrix."""
    arg_constraints = {"logits": constraints.real, "n_obs": constraints.nonnegative_integer}

    def __init__(self, logits, n_obs):
        self.logits = logits
        self.n_obs = n_obs
        self.size = self.logits.size()
        super().__init__()

    def sample(self, num_samples=1):
        with torch.no_grad():
            u = torch.distributions.utils.clamp_probs(torch.rand_like(self.logits))
            z = self.logits - torch.log(-torch.log(u))
            return torch.sort(z, descending=True, stable=True, dim=0)[1]

    def log_prob(self, orders):
        assert self.size == orders.size()
        logits = smart_perm_2D(self.logits, orders)
        logp_matrix = logits - torch.flip(
            torch.logcumsumexp(torch.flip(logits, dims=(0,)), dim=0), dims=(0,))
        mask = torch.arange(orders.shape[0], device=orders.device).unsqueeze(1).expand(
            orders.shape[0], orders.shape[1]) < self.n_obs.view(1, orders.shape[1])
        return logp_matrix[mask].sum()


class MetaboliteGNN(nn.Module):
    """GATv2 encoder for metabolite embeddings from metabolic graph + fingerprints."""
    def __init__(self, n_features: int, hidden_dim: int = 256, latent_dim: int = 30,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * n_heads
            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim, heads=n_heads, dropout=dropout, add_self_loops=True))
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * n_heads, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))

    def forward(self, x, edge_index):
        x = self.feat_proj(x)
        for gat in self.gat_layers:
            x = F.elu(gat(x, edge_index))
        return self.out_proj(x)  # (N_nodes, latent_dim)


class UniGraphModel:
    """
    UniGraph hybrid model.

    H_met_mapped: GNN-encoded embeddings for metabolites mapped to Human-GEM
    H_met_unmapped: Free Bayesian embeddings for unmapped metabolites
    H_gene: Free Bayesian embeddings for genes
    W: Bayesian sample embeddings
    Observation: Plackett-Luce on rank-transformed [met|gene] data
    """

    def __init__(self, latent_dim=30, hidden_dim=256, n_heads=4, n_layers=3,
                 dropout=0.1, lr=0.001, n_steps=4000, device='cpu', seed=42,
                 use_graph=True, use_rank=True, use_bayesian=True, use_chemical=True):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.n_steps = n_steps
        self.device = device
        self.seed = seed
        # Ablation flags
        self.use_graph = use_graph
        self.use_rank = use_rank
        self.use_bayesian = use_bayesian
        self.use_chemical = use_chemical

        self.gnn = None
        self.fingerprints = None
        self.edge_index = None
        self.met_to_gnn_idx = None  # {camp_met_idx: gnn_node_idx}
        self.unmapped_met_indices = None  # array of unmapped camp met indices
        self.n_met = 0
        self.n_genes = 0
        self.n_batch = 0
        self.start_row = None
        self.stop_row = None

        # Stored posterior params
        self.W_loc = None
        self.W_scale = None
        self.H_met_unmapped_loc = None
        self.H_met_unmapped_scale = None
        self.H_gene_loc = None
        self.H_gene_scale = None
        self.H_met_mapped = None  # GNN output (deterministic)

    def _prepare_mappings(self, J_met, graph_data):
        """Prepare index mappings between CAMP metabolites and GNN nodes."""
        hgem_name_to_idx = {name: i for i, name in enumerate(graph_data['hgem_met_ids'])}
        camp_to_hgem = graph_data['camp_to_hgem']  # {camp_met_idx: hgem_met_name}

        self.met_to_gnn_idx = {}
        unmapped = []
        for i in range(J_met):
            if i in camp_to_hgem:
                hgem_name = camp_to_hgem[i]
                if hgem_name in hgem_name_to_idx:
                    self.met_to_gnn_idx[i] = hgem_name_to_idx[hgem_name]
                else:
                    unmapped.append(i)
            else:
                unmapped.append(i)

        self.unmapped_met_indices = np.array(unmapped)
        n_mapped = len(self.met_to_gnn_idx)
        n_unmapped = len(unmapped)
        print(f"  Mapped metabolites: {n_mapped}, Unmapped: {n_unmapped}")
        return n_mapped, n_unmapped

    def fit(self, preprocessed_data, graph_data, batch_info, verbose=True):
        """Train UniGraph."""
        N = preprocessed_data['N']
        J_met = preprocessed_data['J_met']
        J_rna = preprocessed_data['J_rna']
        J = preprocessed_data['J']
        K = self.latent_dim
        self.n_met = J_met
        self.n_genes = J_rna
        self.n_batch = batch_info['n_batch']
        self.start_row = batch_info['start_row']
        self.stop_row = batch_info['stop_row']

        # Prepare mappings
        n_mapped, n_unmapped = self._prepare_mappings(J_met, graph_data)

        # GNN inputs
        self.fingerprints = torch.tensor(
            graph_data['fingerprints'], dtype=torch.float32, device=self.device)
        self.edge_index = torch.tensor(
            graph_data['met_met_edges'], dtype=torch.long, device=self.device)

        # Initialize GNN
        n_features = self.fingerprints.shape[1]
        self.gnn = MetaboliteGNN(
            n_features=n_features, hidden_dim=self.hidden_dim, latent_dim=K,
            n_heads=self.n_heads, n_layers=self.n_layers, dropout=self.dropout
        ).to(self.device)

        # Prepare data tensors
        orders = torch.tensor(preprocessed_data['orders'], dtype=torch.long, device=self.device)
        n_obs = torch.tensor(preprocessed_data['n_obs'], dtype=torch.long, device=self.device)
        start_row = torch.tensor(batch_info['start_row'], dtype=torch.long, device=self.device)
        stop_row = torch.tensor(batch_info['stop_row'], dtype=torch.long, device=self.device)

        # Create index arrays for H construction
        mapped_camp_indices = torch.tensor(
            sorted(self.met_to_gnn_idx.keys()), dtype=torch.long, device=self.device)
        mapped_gnn_indices = torch.tensor(
            [self.met_to_gnn_idx[i] for i in sorted(self.met_to_gnn_idx.keys())],
            dtype=torch.long, device=self.device)
        unmapped_indices = torch.tensor(
            self.unmapped_met_indices, dtype=torch.long, device=self.device)

        # Build model
        gnn = self.gnn
        device = self.device

        def model(orders, n_obs, start_row, stop_row):
            pyro.module('gnn', gnn)

            W = pyro.sample('W', dist.Normal(
                torch.zeros(N, K, device=device),
                torch.ones(N, K, device=device)).to_event(2))

            H_gene = pyro.sample('H_gene', dist.Normal(
                torch.zeros(K, J_rna, device=device),
                torch.ones(K, J_rna, device=device)).to_event(2))

            if n_unmapped > 0:
                H_met_unmapped = pyro.sample('H_met_unmapped', dist.Normal(
                    torch.zeros(K, n_unmapped, device=device),
                    torch.ones(K, n_unmapped, device=device)).to_event(2))
            else:
                H_met_unmapped = torch.zeros(K, 0, device=device)

            # GNN output: (n_gnn_nodes, K) → transpose to (K, n_gnn_nodes)
            gnn_out = gnn(self.fingerprints, self.edge_index).T  # (K, n_gnn_nodes)

            # Build H_met (K, J_met) by placing GNN outputs and free embeddings
            H_met = torch.zeros(K, J_met, device=device)
            if n_mapped > 0:
                H_met[:, mapped_camp_indices] = gnn_out[:, mapped_gnn_indices]
            if n_unmapped > 0:
                H_met[:, unmapped_indices] = H_met_unmapped

            # Full H: [H_met | H_gene]
            H = torch.cat([H_met, H_gene], dim=1)  # (K, J)

            # Logits
            X = torch.mm(W, H)  # (N, J)

            # Plackett-Luce observation per batch
            for b in range(self.n_batch):
                X_temp = X[start_row[b]:stop_row[b], :]
                temp_order = orders[start_row[b]:stop_row[b], :]
                pyro.sample(f"R_{b}", PlackettLuce_2D(X_temp, n_obs[b, :]), obs=temp_order)

        # Guide
        expose = ['W', 'H_gene']
        if n_unmapped > 0:
            expose.append('H_met_unmapped')
        guide = AutoNormal(poutine.block(model, expose=expose))

        # SVI
        pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()
        optimizer = Adam({"lr": self.lr, "betas": [0.95, 0.999]})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        # Train
        loss_list = []
        for step in range(self.n_steps):
            loss = svi.step(orders, n_obs, start_row, stop_row)
            loss_list.append(loss)
            if verbose and step % 200 == 0:
                print(f"  Step {step}: loss = {loss:.2f}")
            if step > 200 and abs(loss - loss_list[-2]) < 1.0:
                if verbose:
                    print(f"  Converged at step {step}")
                break

        # Extract posterior parameters
        self.W_loc = pyro.param("AutoNormal.locs.W").detach().cpu().numpy()
        self.W_scale = pyro.param("AutoNormal.scales.W").detach().cpu().numpy()
        self.H_gene_loc = pyro.param("AutoNormal.locs.H_gene").detach().cpu().numpy()
        self.H_gene_scale = pyro.param("AutoNormal.scales.H_gene").detach().cpu().numpy()
        if n_unmapped > 0:
            self.H_met_unmapped_loc = pyro.param("AutoNormal.locs.H_met_unmapped").detach().cpu().numpy()
            self.H_met_unmapped_scale = pyro.param("AutoNormal.scales.H_met_unmapped").detach().cpu().numpy()

        # Compute GNN-encoded H_met_mapped
        self.gnn.eval()
        with torch.no_grad():
            self.H_met_mapped = self.gnn(self.fingerprints, self.edge_index).T.cpu().numpy()

        # Learn mapping from RNA to W for test-set prediction
        from sklearn.linear_model import RidgeCV
        rna_norm = preprocessed_data['rna_data_norm']
        self.rna_mean = preprocessed_data.get('rna_mean')
        self.rna_std = preprocessed_data.get('rna_std')
        self.rna_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.rna_model.fit(rna_norm, self.W_loc)

        return loss_list

    def predict_ranks(self, n_samples=1000, seed=42):
        """Generate posterior predicted ranks via Gumbel-Max trick."""
        np.random.seed(seed)
        K = self.latent_dim

        # Sample W
        W_draws = np.random.normal(self.W_loc, self.W_scale,
                                   size=(n_samples, *self.W_loc.shape))

        # Sample H_gene
        H_gene_draws = np.random.normal(self.H_gene_loc, self.H_gene_scale,
                                        size=(n_samples, *self.H_gene_loc.shape))

        # Sample H_met_unmapped
        if self.H_met_unmapped_loc is not None:
            H_met_unmapped_draws = np.random.normal(
                self.H_met_unmapped_loc, self.H_met_unmapped_scale,
                size=(n_samples, *self.H_met_unmapped_loc.shape))
        else:
            H_met_unmapped_draws = np.zeros((n_samples, K, 0))

        # H_met_mapped is deterministic (GNN output)
        H_met_mapped = np.expand_dims(self.H_met_mapped, 0)
        H_met_mapped_draws = np.repeat(H_met_mapped, n_samples, axis=0)

        # Build H_met (K, J_met) for each draw
        H_met_draws = np.zeros((n_samples, K, self.n_met))
        mapped_indices = sorted(self.met_to_gnn_idx.keys())
        gnn_indices = [self.met_to_gnn_idx[i] for i in mapped_indices]
        H_met_draws[:, :, mapped_indices] = H_met_mapped_draws[:, :, gnn_indices]
        if len(self.unmapped_met_indices) > 0:
            H_met_draws[:, :, self.unmapped_met_indices] = H_met_unmapped_draws

        # Full H
        H_draws = np.concatenate([H_met_draws, H_gene_draws], axis=2)

        # X = W @ H
        X_draws = np.matmul(W_draws, H_draws)

        # Gumbel-Max trick for ranking (per batch)
        rank_hat_draws = np.full(X_draws.shape, np.nan)
        for b in range(self.n_batch):
            sr, er = self.start_row[b], self.stop_row[b]
            X_temp = X_draws[:, sr:er, :]
            G = np.random.gumbel(0, 1, size=X_temp.shape)
            Z = -(X_temp + G)
            rank_hat_draws[:, sr:er, :] = np.argsort(Z, axis=1, kind='stable').argsort(
                axis=1, kind='stable')

        rank_hat_mean = np.mean(rank_hat_draws, axis=0)
        rank_hat_std = np.std(rank_hat_draws, axis=0)

        return {
            'rank_hat_draws': rank_hat_draws,
            'rank_hat_mean': rank_hat_mean,
            'rank_hat_std': rank_hat_std,
        }

    def predict_met_ranks(self, rna_data=None, batch_info=None, n_samples=1000, seed=42):
        """
        Predict metabolite ranks.
        If rna_data and batch_info are provided, predict for new samples
        by inferring W from gene expression.
        Otherwise, predict for training samples with posterior uncertainty.
        """
        if rna_data is not None and batch_info is not None:
            # Predict W for new samples from gene expression
            rna_log = log_transform_rna(rna_data)
            if hasattr(self, 'rna_mean') and self.rna_mean is not None:
                rna_norm = normalize_rna(rna_log, self.rna_mean, self.rna_std)
            else:
                rna_norm = normalize_rna(rna_log)
            W_pred = self.rna_model.predict(rna_norm)  # (N_test, K)

            # Build H_met from GNN output + unmapped embeddings
            K = self.latent_dim
            H_met = np.zeros((K, self.n_met))
            mapped_indices = sorted(self.met_to_gnn_idx.keys())
            gnn_indices = [self.met_to_gnn_idx[i] for i in mapped_indices]
            H_met[:, mapped_indices] = self.H_met_mapped[:, gnn_indices]
            if len(self.unmapped_met_indices) > 0:
                H_met[:, self.unmapped_met_indices] = self.H_met_unmapped_loc

            # Predict metabolite scores
            X_met = W_pred @ H_met  # (N_test, J_met)

            # Rank per-batch
            ranks = rank_predictions_per_batch(X_met, batch_info)
            return {'rank_hat_mean': ranks}
        else:
            # Predict for training samples with posterior uncertainty
            preds = self.predict_ranks(n_samples=n_samples, seed=seed)
            return {
                'rank_hat_mean': preds['rank_hat_mean'][:, :self.n_met],
                'rank_hat_std': preds['rank_hat_std'][:, :self.n_met],
                'rank_hat_draws': preds['rank_hat_draws'][:, :, :self.n_met],
            }
