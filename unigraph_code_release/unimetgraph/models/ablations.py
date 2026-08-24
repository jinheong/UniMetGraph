"""
Ablation study models for UniGraph.
Each variant removes one component to measure its contribution.

Variants:
1. Full: GNN + Bayesian MF + Plackett-Luce (= UniGraphModel)
2. No graph: Free embeddings for all metabolites (no GNN)
3. No rank: MSE loss instead of Plackett-Luce
4. No Bayesian: Point estimates via Adam instead of SVI
5. No chemical: Constant node features (graph topology only)
6. No graph + no chemical: = UnitedMet
"""
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
import pyro.poutine as poutine
from typing import Dict, Optional

sys.path.insert(0, '/workspace/UnitedMet')
from Performance_Benchmarking.scripts.utils import smart_perm_2D
from Performance_Benchmarking.scripts.pyro_model import run_pyro_svi

from unigraph.models.unigraph import MetaboliteGNN, PlackettLuce_2D
from unigraph.data.preprocess import (
    rank_predictions_per_batch, log_transform_rna, normalize_rna,
    tic_normalization_across
)


class UniGraphAblation:
    """
    UniGraph with configurable ablation flags.
    Supports removing: graph, rank transform, Bayesian inference, chemical features.
    """

    def __init__(self, latent_dim=30, hidden_dim=256, n_heads=4, n_layers=3,
                 dropout=0.1, lr=0.001, n_steps=1000, device='cpu', seed=42,
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
        self.use_graph = use_graph
        self.use_rank = use_rank
        self.use_bayesian = use_bayesian
        self.use_chemical = use_chemical

        self.gnn = None
        self.fingerprints = None
        self.edge_index = None
        self.met_to_gnn_idx = None
        self.unmapped_met_indices = None
        self.n_met = 0
        self.n_genes = 0
        self.n_batch = 0
        self.start_row = None
        self.stop_row = None
        self.W_loc = None
        self.W_scale = None
        self.H_met_unmapped_loc = None
        self.H_met_unmapped_scale = None
        self.H_gene_loc = None
        self.H_gene_scale = None
        self.H_met_mapped = None
        self.rna_model = None
        # For no_graph: all metabolites get free embeddings
        self.H_met_all_loc = None
        self.H_met_all_scale = None
        # For no_bayesian: point estimates
        self.W_point = None
        self.H_gene_point = None
        self.H_met_unmapped_point = None
        self.H_met_all_point = None

    def _prepare_mappings(self, J_met, graph_data):
        hgem_name_to_idx = {name: i for i, name in enumerate(graph_data['hgem_met_ids'])}
        camp_to_hgem = graph_data['camp_to_hgem']
        self.met_to_gnn_idx = {}
        unmapped = []
        for i in range(J_met):
            if i in camp_to_hgem and camp_to_hgem[i] in hgem_name_to_idx:
                self.met_to_gnn_idx[i] = hgem_name_to_idx[camp_to_hgem[i]]
            else:
                unmapped.append(i)
        self.unmapped_met_indices = np.array(unmapped)
        n_mapped = len(self.met_to_gnn_idx)
        n_unmapped = len(unmapped)
        print(f"  Mapped: {n_mapped}, Unmapped: {n_unmapped}")
        return n_mapped, n_unmapped

    def fit(self, preprocessed_data, graph_data, batch_info, verbose=True):
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

        n_mapped, n_unmapped = 0, J_met
        if self.use_graph:
            n_mapped, n_unmapped = self._prepare_mappings(J_met, graph_data)
            self.fingerprints = torch.tensor(
                graph_data['fingerprints'], dtype=torch.float32, device=self.device)
            self.edge_index = torch.tensor(
                graph_data['met_met_edges'], dtype=torch.long, device=self.device)
            if not self.use_chemical:
                # Replace fingerprints with constant vectors (degree-based)
                n_nodes = self.fingerprints.shape[0]
                # Use node degree as feature
                src, dst = self.edge_index
                degrees = torch.zeros(n_nodes, 1, device=self.device)
                degrees.scatter_add_(0, dst.unsqueeze(1).expand(-1, 1),
                                     torch.ones(dst.shape[0], 1, device=self.device))
                self.fingerprints = degrees / degrees.max()

        # Initialize GNN if using graph
        if self.use_graph:
            n_features = self.fingerprints.shape[1]
            self.gnn = MetaboliteGNN(
                n_features=n_features, hidden_dim=self.hidden_dim, latent_dim=K,
                n_heads=self.n_heads, n_layers=self.n_layers, dropout=self.dropout
            ).to(self.device)

        if self.use_bayesian:
            return self._fit_bayesian(preprocessed_data, graph_data, batch_info,
                                       N, J_met, J_rna, J, K, n_mapped, n_unmapped, verbose)
        else:
            return self._fit_point(preprocessed_data, graph_data, batch_info,
                                    N, J_met, J_rna, J, K, n_mapped, n_unmapped, verbose)

    def _fit_bayesian(self, preprocessed_data, graph_data, batch_info,
                      N, J_met, J_rna, J, K, n_mapped, n_unmapped, verbose):
        """Fit with SVI (Bayesian inference)."""
        device = self.device
        orders = torch.tensor(preprocessed_data['orders'], dtype=torch.long, device=device)
        n_obs = torch.tensor(preprocessed_data['n_obs'], dtype=torch.long, device=device)
        start_row = torch.tensor(batch_info['start_row'], dtype=torch.long, device=device)
        stop_row = torch.tensor(batch_info['stop_row'], dtype=torch.long, device=device)

        # For MSE (no rank), need TIC-normalized metabolomics
        met_tic_t = None
        if not self.use_rank:
            met_tic = preprocessed_data['met_data_tic']
            met_tic_filled = np.where(np.isnan(met_tic), 0, met_tic)
            met_tic_t = torch.tensor(met_tic_filled, dtype=torch.float32, device=device)
            obs_mask = torch.tensor(~np.isnan(met_tic), dtype=torch.float32, device=device)

        if self.use_graph:
            mapped_camp_indices = torch.tensor(
                sorted(self.met_to_gnn_idx.keys()), dtype=torch.long, device=device)
            mapped_gnn_indices = torch.tensor(
                [self.met_to_gnn_idx[i] for i in sorted(self.met_to_gnn_idx.keys())],
                dtype=torch.long, device=device)
            unmapped_indices = torch.tensor(
                self.unmapped_met_indices, dtype=torch.long, device=device)
        else:
            mapped_camp_indices = torch.zeros(0, dtype=torch.long, device=device)
            mapped_gnn_indices = torch.zeros(0, dtype=torch.long, device=device)
            unmapped_indices = torch.zeros(0, dtype=torch.long, device=device)

        gnn = self.gnn
        fingerprints = self.fingerprints
        edge_index = self.edge_index
        use_graph = self.use_graph
        use_rank = self.use_rank
        n_batch = self.n_batch

        def model(orders, n_obs, start_row, stop_row):
            if use_graph:
                pyro.module('gnn', gnn)

            W = pyro.sample('W', dist.Normal(
                torch.zeros(N, K, device=device),
                torch.ones(N, K, device=device)).to_event(2))
            H_gene = pyro.sample('H_gene', dist.Normal(
                torch.zeros(K, J_rna, device=device),
                torch.ones(K, J_rna, device=device)).to_event(2))

            if use_graph:
                if n_unmapped > 0:
                    H_met_unmapped = pyro.sample('H_met_unmapped', dist.Normal(
                        torch.zeros(K, n_unmapped, device=device),
                        torch.ones(K, n_unmapped, device=device)).to_event(2))
                else:
                    H_met_unmapped = torch.zeros(K, 0, device=device)

                gnn_out = gnn(fingerprints, edge_index).T
                H_met = torch.zeros(K, J_met, device=device)
                if n_mapped > 0:
                    H_met[:, mapped_camp_indices] = gnn_out[:, mapped_gnn_indices]
                if n_unmapped > 0:
                    H_met[:, unmapped_indices] = H_met_unmapped
            else:
                # No graph: all metabolites get free embeddings
                H_met = pyro.sample('H_met_all', dist.Normal(
                    torch.zeros(K, J_met, device=device),
                    torch.ones(K, J_met, device=device)).to_event(2))

            if use_rank:
                H = torch.cat([H_met, H_gene], dim=1)
                X = torch.mm(W, H)
                for b in range(n_batch):
                    X_temp = X[start_row[b]:stop_row[b], :]
                    temp_order = orders[start_row[b]:stop_row[b], :]
                    pyro.sample(f"R_{b}", PlackettLuce_2D(X_temp, n_obs[b, :]), obs=temp_order)
            else:
                # MSE loss on metabolites only
                X_met = torch.mm(W, H_met)
                # Use Normal observation model
                pyro.sample("met_obs", dist.Normal(X_met, 0.1).to_event(2),
                           obs=met_tic_t * obs_mask)

        # Guide
        expose = ['W', 'H_gene']
        if use_graph and n_unmapped > 0:
            expose.append('H_met_unmapped')
        if not use_graph:
            expose.append('H_met_all')
        guide = AutoNormal(poutine.block(model, expose=expose))

        pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()
        optimizer = Adam({"lr": self.lr, "betas": [0.95, 0.999]})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        loss_list = []
        for step in range(self.n_steps):
            loss = svi.step(orders, n_obs, start_row, stop_row)
            loss_list.append(loss)
            if verbose and step % 200 == 0:
                print(f"  Step {step}: loss = {loss:.2f}")

        # Extract posterior parameters
        self.W_loc = pyro.param("AutoNormal.locs.W").detach().cpu().numpy()
        self.W_scale = pyro.param("AutoNormal.scales.W").detach().cpu().numpy()
        self.H_gene_loc = pyro.param("AutoNormal.locs.H_gene").detach().cpu().numpy()
        self.H_gene_scale = pyro.param("AutoNormal.scales.H_gene").detach().cpu().numpy()

        if use_graph:
            if n_unmapped > 0:
                self.H_met_unmapped_loc = pyro.param("AutoNormal.locs.H_met_unmapped").detach().cpu().numpy()
                self.H_met_unmapped_scale = pyro.param("AutoNormal.scales.H_met_unmapped").detach().cpu().numpy()
            self.gnn.eval()
            with torch.no_grad():
                self.H_met_mapped = self.gnn(fingerprints, edge_index).T.cpu().numpy()
        else:
            self.H_met_all_loc = pyro.param("AutoNormal.locs.H_met_all").detach().cpu().numpy()
            self.H_met_all_scale = pyro.param("AutoNormal.scales.H_met_all").detach().cpu().numpy()

        # Learn RNA → W mapping
        from sklearn.linear_model import RidgeCV
        rna_norm = preprocessed_data['rna_data_norm']
        self.rna_mean = preprocessed_data.get('rna_mean')
        self.rna_std = preprocessed_data.get('rna_std')
        self.rna_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.rna_model.fit(rna_norm, self.W_loc)

        return loss_list

    def _fit_point(self, preprocessed_data, graph_data, batch_info,
                   N, J_met, J_rna, J, K, n_mapped, n_unmapped, verbose):
        """Fit with Adam (point estimates, no Bayesian inference)."""
        device = self.device
        torch.manual_seed(self.seed)

        # Prepare data
        if self.use_rank:
            orders = torch.tensor(preprocessed_data['orders'], dtype=torch.long, device=device)
            n_obs = torch.tensor(preprocessed_data['n_obs'], dtype=torch.long, device=device)
        else:
            met_tic = preprocessed_data['met_data_tic']
            met_tic_filled = np.where(np.isnan(met_tic), 0, met_tic)
            met_tic_t = torch.tensor(met_tic_filled, dtype=torch.float32, device=device)
            obs_mask = torch.tensor(~np.isnan(met_tic), dtype=torch.float32, device=device)

        start_row = batch_info['start_row']
        stop_row = batch_info['stop_row']

        # Initialize parameters
        self.W_point = nn.Parameter(torch.randn(N, K, device=device) * 0.01)
        self.H_gene_point = nn.Parameter(torch.zeros(K, J_rna, device=device))

        if self.use_graph:
            if n_unmapped > 0:
                self.H_met_unmapped_point = nn.Parameter(torch.zeros(K, n_unmapped, device=device))
            mapped_camp = sorted(self.met_to_gnn_idx.keys())
            mapped_gnn = [self.met_to_gnn_idx[i] for i in mapped_camp]
        else:
            self.H_met_all_point = nn.Parameter(torch.zeros(K, J_met, device=device))

        # Optimizer
        params = [self.W_point, self.H_gene_point]
        if self.use_graph:
            params += list(self.gnn.parameters())
            if n_unmapped > 0:
                params.append(self.H_met_unmapped_point)
        else:
            params.append(self.H_met_all_point)

        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_list = []

        for step in range(self.n_steps):
            optimizer.zero_grad()

            if self.use_graph:
                gnn_out = self.gnn(self.fingerprints, self.edge_index).T
                H_met = torch.zeros(K, J_met, device=device)
                if n_mapped > 0:
                    H_met[:, mapped_camp] = gnn_out[:, mapped_gnn]
                if n_unmapped > 0:
                    H_met[:, self.unmapped_met_indices] = self.H_met_unmapped_point
            else:
                H_met = self.H_met_all_point

            if self.use_rank:
                H = torch.cat([H_met, self.H_gene_point], dim=1)
                X = torch.mm(self.W_point, H)
                # Plackett-Luce loss (negative log prob)
                loss = 0
                for b in range(self.n_batch):
                    sr, er = start_row[b], stop_row[b]
                    X_temp = X[sr:er, :]
                    order_temp = orders[sr:er, :]
                    n_obs_b = n_obs[b, :]
                    logits = smart_perm_2D(X_temp, order_temp)
                    logp = logits - torch.flip(
                        torch.logcumsumexp(torch.flip(logits, dims=(0,)), dim=0), dims=(0,))
                    mask = torch.arange(order_temp.shape[0], device=device).unsqueeze(1).expand(
                        order_temp.shape[0], order_temp.shape[1]) < n_obs_b.view(1, -1)
                    loss -= logp[mask].sum()
            else:
                X_met = torch.mm(self.W_point, H_met)
                loss = (obs_mask * (X_met - met_tic_t) ** 2).sum() / obs_mask.sum()

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if verbose and step % 200 == 0:
                print(f"  Step {step}: loss = {loss.item():.2f}")

        # Store results
        self.W_loc = self.W_point.detach().cpu().numpy()
        self.W_scale = np.zeros_like(self.W_loc)  # No uncertainty
        self.H_gene_loc = self.H_gene_point.detach().cpu().numpy()
        self.H_gene_scale = np.zeros_like(self.H_gene_loc)

        if self.use_graph:
            if n_unmapped > 0:
                self.H_met_unmapped_loc = self.H_met_unmapped_point.detach().cpu().numpy()
                self.H_met_unmapped_scale = np.zeros_like(self.H_met_unmapped_loc)
            self.gnn.eval()
            with torch.no_grad():
                self.H_met_mapped = self.gnn(self.fingerprints, self.edge_index).T.cpu().numpy()
        else:
            self.H_met_all_loc = self.H_met_all_point.detach().cpu().numpy()
            self.H_met_all_scale = np.zeros_like(self.H_met_all_loc)

        # Learn RNA → W mapping
        from sklearn.linear_model import RidgeCV
        rna_norm = preprocessed_data['rna_data_norm']
        self.rna_mean = preprocessed_data.get('rna_mean')
        self.rna_std = preprocessed_data.get('rna_std')
        self.rna_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.rna_model.fit(rna_norm, self.W_loc)

        return loss_list

    def predict_met_ranks(self, rna_data=None, batch_info=None, n_samples=1000, seed=42):
        """Predict metabolite ranks for test samples."""
        if rna_data is not None and batch_info is not None:
            rna_log = log_transform_rna(rna_data)
            if hasattr(self, 'rna_mean') and self.rna_mean is not None:
                rna_norm = normalize_rna(rna_log, self.rna_mean, self.rna_std)
            else:
                rna_norm = normalize_rna(rna_log)
            W_pred = self.rna_model.predict(rna_norm)

            K = self.latent_dim
            H_met = np.zeros((K, self.n_met))

            if self.use_graph:
                mapped_indices = sorted(self.met_to_gnn_idx.keys())
                gnn_indices = [self.met_to_gnn_idx[i] for i in mapped_indices]
                H_met[:, mapped_indices] = self.H_met_mapped[:, gnn_indices]
                if len(self.unmapped_met_indices) > 0:
                    H_met[:, self.unmapped_met_indices] = self.H_met_unmapped_loc
            else:
                H_met = self.H_met_all_loc

            X_met = W_pred @ H_met
            ranks = rank_predictions_per_batch(X_met, batch_info)
            return {'rank_hat_mean': ranks}
        else:
            # Training prediction with uncertainty
            np.random.seed(seed)
            W_draws = np.random.normal(self.W_loc, self.W_scale,
                                       size=(n_samples, *self.W_loc.shape))
            K = self.latent_dim
            H_met = np.zeros((K, self.n_met))
            if self.use_graph:
                mapped_indices = sorted(self.met_to_gnn_idx.keys())
                gnn_indices = [self.met_to_gnn_idx[i] for i in mapped_indices]
                H_met[:, mapped_indices] = self.H_met_mapped[:, gnn_indices]
                if len(self.unmapped_met_indices) > 0:
                    H_met_draws = np.random.normal(
                        self.H_met_unmapped_loc, self.H_met_unmapped_scale,
                        size=(n_samples, K, len(self.unmapped_met_indices)))
                    for s in range(n_samples):
                        H_met_s = H_met.copy()
                        H_met_s[:, self.unmapped_met_indices] = H_met_draws[s]
                        X_met = W_draws[s] @ H_met_s
                        # Rank per batch
                        ...
            else:
                H_met_draws = np.random.normal(
                    self.H_met_all_loc, self.H_met_all_scale,
                    size=(n_samples, K, self.n_met))

            # Simplified: use point estimate
            X_met = self.W_loc @ H_met
            if batch_info is not None:
                ranks = rank_predictions_per_batch(X_met, batch_info)
            else:
                ranks = X_met
            return {'rank_hat_mean': ranks}
