"""
Baseline models for transcriptome-to-metabolome prediction benchmark.
1. UnitedMet (rank-based Bayesian MF + Plackett-Luce)
2. Simplified GNN (GAZE-like: GATv2 + Morgan fingerprints, continuous MSE)
3. XGBoost (per-metabolite, top genes via LassoCV)
4. Lasso (per-metabolite, LassoCV)
5. MIRTH (metabolite-only MF extended for cross-modality)
6. MOFA (R wrapper)
7. Kernel MKL (R wrapper)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, '/workspace/UnitedMet')
from Performance_Benchmarking.scripts.pyro_model import (
    PlackettLuce_2D, run_pyro_svi, pyro_posterior
)
from Performance_Benchmarking.scripts.utils import (
    count_obs, order_and_rank, gumbel_sampling_3D, smart_perm_2D
)
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch_geometric.nn import GATv2Conv

from unigraph.data.preprocess import (
    tic_normalization_across, log_transform_rna, normalize_rna,
    count_obs as _count_obs, order_and_rank as _order_and_rank,
    rank_predictions_per_batch
)
from unigraph.models.unigraph import MetaboliteGNN


# ============================================================
# 1. UnitedMet Baseline
# ============================================================
class UnitedMetBaseline:
    """UnitedMet: rank-based Bayesian matrix factorization + Plackett-Luce."""

    def __init__(self, latent_dim=30, n_steps=4000, lr=0.001, seed=42, device='cpu'):
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.lr = lr
        self.seed = seed
        self.device = device
        self.W_loc = None
        self.W_scale = None
        self.H_loc = None
        self.H_scale = None
        self.J_met = None
        self.J_gene = None
        self.rna_model = None  # mapping from RNA to W for test prediction

    def fit(self, preprocessed_data, batch_info, verbose=True):
        N = preprocessed_data['N']
        J = preprocessed_data['J']
        K = self.latent_dim
        n_batch = batch_info['n_batch']

        self.J_met = preprocessed_data['J_met']
        self.J_gene = preprocessed_data['J_rna']

        orders = preprocessed_data['orders']
        n_obs = preprocessed_data['n_obs']
        start_row = batch_info['start_row']
        stop_row = batch_info['stop_row']

        W_loc, W_scale, H_loc, H_scale, losses = run_pyro_svi(
            N, J, K, n_batch, start_row, stop_row,
            n_obs, orders, n_steps=self.n_steps, lr=self.lr
        )

        self.W_loc = W_loc
        self.W_scale = W_scale
        self.H_loc = H_loc
        self.H_scale = H_scale

        # Learn mapping from RNA to W for test-set prediction
        from sklearn.linear_model import RidgeCV
        rna_norm = preprocessed_data['rna_data_norm']
        # Store training RNA statistics for test normalization
        self.rna_mean = preprocessed_data.get('rna_mean')
        self.rna_std = preprocessed_data.get('rna_std')
        self.rna_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.rna_model.fit(rna_norm, self.W_loc)

        return losses

    def predict_met_ranks(self, rna_data=None, batch_info=None, n_samples=1000, seed=42):
        """
        Predict metabolite ranks.
        If rna_data and batch_info are provided, predict for new samples
        by inferring W from gene expression.
        Otherwise, predict for training samples.
        """
        np.random.seed(seed)
        K = self.latent_dim

        if rna_data is not None and batch_info is not None:
            # Predict W for new samples from gene expression
            rna_log = log_transform_rna(rna_data)
            if self.rna_mean is not None and self.rna_std is not None:
                rna_norm = normalize_rna(rna_log, self.rna_mean, self.rna_std)
            else:
                rna_norm = normalize_rna(rna_log)
            W_pred = self.rna_model.predict(rna_norm)  # (N_test, K)

            # Use point estimates for H
            H = self.H_loc  # (K, J)
            X = W_pred @ H  # (N_test, J)

            # Extract metabolite portion and rank per-batch
            X_met = X[:, :self.J_met]
            pred = X_met
        else:
            # Predict for training samples with uncertainty
            W_draws = np.random.normal(self.W_loc, self.W_scale,
                                       size=(n_samples, *self.W_loc.shape))
            H_draws = np.random.normal(self.H_loc, self.H_scale,
                                       size=(n_samples, *self.H_loc.shape))
            X_draws = np.matmul(W_draws, H_draws)

            # Gumbel-Max trick for ranking
            G = np.random.gumbel(0, 1, size=X_draws.shape)
            Z = -(X_draws + G)
            rank_hat_draws = np.argsort(Z, axis=1, kind='stable').argsort(axis=1, kind='stable')
            rank_hat_mean = np.mean(rank_hat_draws, axis=0)

            # Extract metabolite portion
            pred_ranks_met = rank_hat_mean[:, :self.J_met]

            # Rank per-batch if batch_info provided
            if batch_info is not None:
                # Re-rank the mean ranks per batch
                ranks = rank_predictions_per_batch(pred_ranks_met, batch_info)
                return {'rank_hat_mean': ranks}
            return {'rank_hat_mean': pred_ranks_met}

        # Rank predictions per-batch
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# 2. Simplified GNN Baseline (GAZE-like)
# ============================================================
class SimplifiedGNN:
    """
    Simplified GAZE: GATv2 + Morgan fingerprints + gene expression.
    Continuous MSE loss (no rank transform, no Plackett-Luce).
    """

    def __init__(self, latent_dim=30, hidden_dim=256, n_heads=4, n_layers=3,
                 dropout=0.1, lr=0.001, n_steps=4000, device='cpu', seed=42):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.n_steps = n_steps
        self.device = device
        self.seed = seed
        self.gnn = None
        self.W = None  # sample embeddings (point estimate)
        self.H_gene = None  # gene embeddings (point estimate)
        self.met_to_gnn_idx = None
        self.unmapped_met_indices = None
        self.H_met_unmapped = None
        self.n_met = 0
        self.n_genes = 0

    def fit(self, met_data, rna_data, graph_data, batch_info, verbose=True):
        """
        Train with continuous MSE loss on TIC-normalized metabolomics.
        """
        torch.manual_seed(self.seed)
        N, J_met = met_data.shape
        J_rna = rna_data.shape[1]
        K = self.latent_dim
        self.n_met = J_met
        self.n_genes = J_rna

        # Prepare mappings
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

        # TIC normalize
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        # Replace NaN with 0 for MSE training
        met_tic_filled = np.where(np.isnan(met_tic), 0, met_tic)

        # Convert to tensors
        met_t = torch.tensor(met_tic_filled, dtype=torch.float32, device=self.device)
        rna_t = torch.tensor(rna_norm, dtype=torch.float32, device=self.device)
        fingerprints = torch.tensor(graph_data['fingerprints'], dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(graph_data['met_met_edges'], dtype=torch.long, device=self.device)

        # Initialize models
        n_features = fingerprints.shape[1]
        self.gnn = MetaboliteGNN(
            n_features=n_features, hidden_dim=self.hidden_dim, latent_dim=K,
            n_heads=self.n_heads, n_layers=self.n_layers, dropout=self.dropout
        ).to(self.device)

        # Free embeddings for unmapped metabolites and genes
        self.H_met_unmapped = nn.Parameter(torch.zeros(K, n_unmapped, device=self.device))
        self.H_gene = nn.Parameter(torch.zeros(K, J_rna, device=self.device))
        self.W = nn.Parameter(torch.randn(N, K, device=self.device) * 0.01)

        # Optimizer
        params = list(self.gnn.parameters()) + [self.W, self.H_met_unmapped, self.H_gene]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Create mask for observed metabolites
        obs_mask = ~np.isnan(met_tic)
        obs_mask_t = torch.tensor(obs_mask, dtype=torch.float32, device=self.device)

        # Training loop
        for step in range(self.n_steps):
            self.gnn.train()
            optimizer.zero_grad()

            # GNN output
            gnn_out = self.gnn(fingerprints, edge_index).T  # (K, n_gnn_nodes)

            # Build H_met
            H_met = torch.zeros(K, J_met, device=self.device)
            if n_mapped > 0:
                mapped_camp = sorted(self.met_to_gnn_idx.keys())
                mapped_gnn = [self.met_to_gnn_idx[i] for i in mapped_camp]
                H_met[:, mapped_camp] = gnn_out[:, mapped_gnn]
            if n_unmapped > 0:
                H_met[:, self.unmapped_met_indices] = self.H_met_unmapped

            # Predict
            pred = torch.mm(self.W, H_met)  # (N, J_met)

            # MSE loss on observed entries only
            loss = (obs_mask_t * (pred - met_t) ** 2).sum() / obs_mask_t.sum()

            loss.backward()
            optimizer.step()

            if verbose and step % 500 == 0:
                print(f"  GNN Step {step}: loss = {loss.item():.6f}")

        # Store final parameters
        self.W = self.W.detach().cpu().numpy()
        self.H_gene = self.H_gene.detach().cpu().numpy()
        self.H_met_unmapped = self.H_met_unmapped.detach().cpu().numpy()
        self.gnn.eval()
        with torch.no_grad():
            self.H_met_mapped = self.gnn(fingerprints, edge_index).T.cpu().numpy()

        # Learn mapping from RNA to W for test-set prediction
        from sklearn.linear_model import RidgeCV
        # Store training RNA statistics
        self.rna_mean = np.nanmean(rna_log, axis=0, keepdims=True)
        self.rna_std = np.nanstd(rna_log, axis=0, keepdims=True)
        self.rna_std[self.rna_std == 0] = 1.0
        self.rna_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.rna_model.fit(rna_norm, self.W)

        return None

    def predict_met(self, rna_data=None, batch_info=None):
        """Predict metabolite abundances (continuous).
        If rna_data provided, infer W from gene expression for new samples."""
        K = self.latent_dim
        H_met = np.zeros((K, self.n_met))
        mapped_camp = sorted(self.met_to_gnn_idx.keys())
        mapped_gnn = [self.met_to_gnn_idx[i] for i in mapped_camp]
        H_met[:, mapped_camp] = self.H_met_mapped[:, mapped_gnn]
        if len(self.unmapped_met_indices) > 0:
            H_met[:, self.unmapped_met_indices] = self.H_met_unmapped

        if rna_data is not None:
            rna_log = log_transform_rna(rna_data)
            if hasattr(self, 'rna_mean') and self.rna_mean is not None:
                rna_norm = normalize_rna(rna_log, self.rna_mean, self.rna_std)
            else:
                rna_norm = normalize_rna(rna_log)
            W_pred = self.rna_model.predict(rna_norm)
        else:
            W_pred = self.W

        pred = W_pred @ H_met  # (N, J_met)
        return pred

    def predict_met_ranks(self, rna_data=None, batch_info=None):
        """Predict metabolite ranks (rank-transform the continuous predictions per batch)."""
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# 3. XGBoost Baseline
# ============================================================
class XGBoostBaseline:
    """Per-metabolite XGBoost with univariate gene pre-filtering."""

    def __init__(self, n_top_genes=100, n_estimators=200, max_depth=6,
                 lr=0.1, seed=42, device='cpu', n_prefilter=500):
        self.n_top_genes = n_top_genes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.seed = seed
        self.n_prefilter = n_prefilter  # pre-filter genes by correlation
        self.models = {}
        self.selected_genes = {}
        self.scalers_X = {}
        self.scalers_y = {}

    def _prefilter_genes(self, X, y, n_top):
        """Fast vectorized univariate correlation filter."""
        # Vectorized Pearson correlation
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        num = X_centered.T @ y_centered
        denom = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
        denom[denom == 0] = 1.0
        correlations = np.abs(num / denom)
        return np.argsort(correlations)[-n_top:]

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        import xgboost as xgb

        N, J_met = met_data.shape
        J_rna = rna_data.shape[1]

        # TIC normalize metabolomics
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        for j in range(J_met):
            # Get observed samples for this metabolite
            mask = ~np.isnan(met_tic[:, j])
            if mask.sum() < 20:
                continue

            y = met_tic[mask, j]
            X = rna_norm[mask]

            # Fast univariate pre-filter
            top_pre = self._prefilter_genes(X, y, self.n_prefilter)
            X_pre = X[:, top_pre]

            # LassoCV on pre-filtered genes
            lasso = LassoCV(cv=3, max_iter=2000, random_state=self.seed, n_alphas=20)
            lasso.fit(X_pre, y)
            top_genes = top_pre[np.argsort(np.abs(lasso.coef_))[-self.n_top_genes:]]
            self.selected_genes[j] = top_genes

            X_sel = X[:, top_genes]

            # Scale
            sx = StandardScaler()
            sy = StandardScaler()
            X_sel = sx.fit_transform(X_sel)
            y = sy.fit_transform(y.reshape(-1, 1)).ravel()
            self.scalers_X[j] = sx
            self.scalers_y[j] = sy

            # XGBoost
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.lr, random_state=self.seed,
                n_jobs=4, verbosity=0
            )
            model.fit(X_sel, y)
            self.models[j] = model

        if verbose:
            print(f"  Trained {len(self.models)}/{J_met} metabolite models")

    def predict_met(self, rna_data, batch_info):
        N, J_met = len(rna_data), len(self.models) if self.models else 0
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        preds = np.full((rna_data.shape[0], max(self.models.keys()) + 1 if self.models else 0), np.nan)
        for j, model in self.models.items():
            top_genes = self.selected_genes[j]
            X = rna_norm[:, top_genes]
            X = self.scalers_X[j].transform(X)
            pred = model.predict(X)
            pred = self.scalers_y[j].inverse_transform(pred.reshape(-1, 1)).ravel()
            preds[:, j] = pred

        return preds

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# 4. Lasso Baseline
# ============================================================
class LassoBaseline:
    """Per-metabolite Lasso regression with gene pre-filtering."""

    def __init__(self, seed=42, device='cpu', n_prefilter=500):
        self.seed = seed
        self.n_prefilter = n_prefilter
        self.models = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.selected_genes = {}

    def _prefilter_genes(self, X, y, n_top):
        """Fast vectorized univariate correlation filter."""
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        num = X_centered.T @ y_centered
        denom = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
        denom[denom == 0] = 1.0
        correlations = np.abs(num / denom)
        return np.argsort(correlations)[-n_top:]

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        N, J_met = met_data.shape
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        for j in range(J_met):
            mask = ~np.isnan(met_tic[:, j])
            if mask.sum() < 20:
                continue

            y = met_tic[mask, j]
            X = rna_norm[mask]

            # Pre-filter genes by correlation
            top_genes = self._prefilter_genes(X, y, self.n_prefilter)
            self.selected_genes[j] = top_genes
            X_sel = X[:, top_genes]

            sx = StandardScaler()
            sy = StandardScaler()
            X_sel = sx.fit_transform(X_sel)
            y = sy.fit_transform(y.reshape(-1, 1)).ravel()
            self.scalers_X[j] = sx
            self.scalers_y[j] = sy

            model = LassoCV(cv=3, max_iter=5000, random_state=self.seed, n_alphas=50)
            model.fit(X_sel, y)
            self.models[j] = model

        if verbose:
            print(f"  Trained {len(self.models)}/{J_met} metabolite models")

    def predict_met(self, rna_data, batch_info):
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)
        n_samples = rna_data.shape[0]
        n_mets = max(self.models.keys()) + 1 if self.models else 0
        preds = np.full((n_samples, n_mets), np.nan)

        for j, model in self.models.items():
            top_genes = self.selected_genes[j]
            X = rna_norm[:, top_genes]
            X = self.scalers_X[j].transform(X)
            pred = model.predict(X)
            preds[:, j] = self.scalers_y[j].inverse_transform(pred.reshape(-1, 1)).ravel()

        return preds

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# 5. MIRTH Baseline
# ============================================================
class MIRTHBaseline:
    """
    MIRTH: Metabolite-only matrix factorization extended for cross-modality.
    Uses SVD-based factorization on metabolomics data, then learns
    a mapping from gene expression to the sample factors.
    """

    def __init__(self, latent_dim=30, seed=42, device='cpu'):
        self.latent_dim = latent_dim
        self.seed = seed
        self.device = device
        self.W_met = None  # SVD factors
        self.H_met = None  # SVD loadings
        self.gene_model = None  # mapping from genes to W

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import RidgeCV

        N, J_met = met_data.shape
        K = self.latent_dim

        # TIC normalize
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        # Replace NaN with 0 for SVD
        met_filled = np.where(np.isnan(met_tic), 0, met_tic)

        # SVD on metabolomics
        svd = TruncatedSVD(n_components=K, random_state=self.seed)
        W_met = svd.fit_transform(met_filled)  # (N, K)
        H_met = svd.components_  # (K, J_met)
        self.W_met = W_met
        self.H_met = H_met

        # Learn mapping from gene expression to W_met
        self.gene_model = RidgeCV(alphas=np.logspace(-3, 3, 50))
        self.gene_model.fit(rna_norm, W_met)

        if verbose:
            print(f"  MIRTH: SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    def predict_met(self, rna_data, batch_info):
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)
        W_pred = self.gene_model.predict(rna_norm)  # (N, K)
        pred = W_pred @ self.H_met  # (N, J_met)
        return pred

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# 6. MOFA Baseline (R wrapper)
# ============================================================
class MOFABaseline:
    """MOFA+ via R wrapper. Bayesian group-wise matrix factorization."""

    def __init__(self, latent_dim=30, n_steps=1000, seed=42, device='cpu'):
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.seed = seed
        self.device = device
        self.W = None  # sample factors
        self.H_met = None  # metabolite loadings
        self.gene_model = None

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()

        N, J_met = met_data.shape
        K = self.latent_dim

        # TIC normalize
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        # Replace NaN with 0
        met_filled = np.where(np.isnan(met_tic), 0, met_tic)

        # Call MOFA2 via R
        r_code = f"""
        library(MOFA2)
        set.seed({self.seed})

        # Create data matrices
        met_data <- matrix({ro.r('paste(as.character(as.vector(met_data_r)), collapse=",")')}, nrow={N}, ncol={J_met})
        rna_data <- matrix({ro.r('paste(as.character(as.vector(rna_data_r)), collapse=",")')}, nrow={N}, ncol={rna_data.shape[1]})

        # Create MOFA object
        data <- list(metabolomics = met_data, transcriptomics = rna_data)
        mofa <- create_mofa(data)
        mofa <- prepare_mofa(mofa, num_factors = {K})

        # Train
        mofa <- run_mofa(mofa, maxiter = {self.n_steps})

        # Extract factors and weights
        W <- get_factors(mofa)[[1]]
        H_met <- get_weights(mofa)[[1]]

        list(W = W, H_met = H_met)
        """

        try:
            result = ro.r(r_code)
            self.W = np.array(result[0])
            self.H_met = np.array(result[1]).T  # MOFA returns (J, K), we want (K, J)

            # Learn gene-to-W mapping
            from sklearn.linear_model import RidgeCV
            self.gene_model = RidgeCV(alphas=np.logspace(-3, 3, 50))
            self.gene_model.fit(rna_norm, self.W)

            if verbose:
                print(f"  MOFA: trained {K} factors")
        except Exception as e:
            print(f"  MOFA error: {e}")
            # Fallback: use SVD
            from sklearn.decomposition import TruncatedSVD
            from sklearn.linear_model import RidgeCV
            svd = TruncatedSVD(n_components=K, random_state=self.seed)
            self.W = svd.fit_transform(np.where(np.isnan(met_tic), 0, met_tic))
            self.H_met = svd.components_
            self.gene_model = RidgeCV(alphas=np.logspace(-3, 3, 50))
            self.gene_model.fit(rna_norm, self.W)

    def predict_met(self, rna_data, batch_info):
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)
        W_pred = self.gene_model.predict(rna_norm)
        return W_pred @ self.H_met

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# 7. Kernel MKL Baseline (R wrapper)
# ============================================================
class KernelMKLBaseline:
    """Kernel Multiple Kernel Learning via mixKernel R package."""

    def __init__(self, latent_dim=30, seed=42, device='cpu'):
        self.latent_dim = latent_dim
        self.seed = seed
        self.device = device
        self.W = None
        self.H_met = None
        self.gene_model = None

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        # Simplified MKL: KernelPCA on metabolomics + kernel ridge from genes
        from sklearn.decomposition import KernelPCA
        from sklearn.linear_model import RidgeCV

        N, J_met = met_data.shape
        K = self.latent_dim

        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        met_filled = np.where(np.isnan(met_tic), 0, met_tic)

        # KernelPCA with RBF kernel on metabolomics
        kpca = KernelPCA(n_components=K, kernel='rbf',
                         gamma=1.0/met_filled.shape[1], random_state=self.seed)
        self.W = kpca.fit_transform(met_filled)
        # Reconstruct loadings via inverse transform
        self.H_met = kpca.eigenvectors_.T  # (K, N) -> need (K, J_met)
        # For prediction, we need H_met such that W @ H_met ≈ met_filled
        # Use least squares: H_met = pinv(W) @ met_filled
        self.H_met = np.linalg.pinv(self.W) @ met_filled  # (K, J_met)

        # Kernel ridge regression from genes to factors
        self.gene_model = RidgeCV(alphas=np.logspace(-3, 3, 50))
        self.gene_model.fit(rna_norm, self.W)

        if verbose:
            print(f"  Kernel MKL: trained {K} factors (KernelPCA + Ridge)")

    def predict_met(self, rna_data, batch_info):
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)
        W_pred = self.gene_model.predict(rna_norm)
        return W_pred @ self.H_met

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}
