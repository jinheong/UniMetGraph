"""
Fast baseline models for transcriptome-to-metabolome prediction.
Optimized versions that use vectorized operations and pre-filtering.
"""
import numpy as np
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from unigraph.data.preprocess import (
    tic_normalization_across, log_transform_rna, normalize_rna,
    count_obs, order_and_rank, rank_predictions_per_batch
)


def _get_valid_metabolites(met_data, min_obs=20):
    """Get indices of metabolites with enough observations."""
    obs_count = np.sum(~np.isnan(met_data), axis=0)
    return np.where(obs_count >= min_obs)[0]


def _vectorized_correlation_filter(X, Y, n_top):
    """
    Compute correlations between all columns of X and all columns of Y.
    Returns top gene indices for each metabolite.
    """
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Compute correlation matrix: (J_x, J_y)
    num = X_centered.T @ Y_centered
    x_norm = np.sqrt((X_centered ** 2).sum(axis=0))
    y_norm = np.sqrt((Y_centered ** 2).sum(axis=0))
    denom = np.outer(x_norm, y_norm)
    denom[denom == 0] = 1.0
    corr_matrix = np.abs(num / denom)  # (J_x, J_y)

    # Top n_top genes for each metabolite
    top_indices = np.argsort(corr_matrix, axis=0)[-n_top:].T  # (J_y, n_top)
    return top_indices


# ============================================================
# Fast Lasso Baseline
# ============================================================
class FastLassoBaseline:
    """Per-metabolite Lasso with vectorized gene pre-filtering."""

    def __init__(self, n_top_genes=200, alpha=0.01, seed=42, device='cpu'):
        self.n_top_genes = n_top_genes
        self.alpha = alpha
        self.seed = seed
        self.models = {}
        self.selected_genes = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.valid_mets = None

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        N, J_met = met_data.shape
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        # Get valid metabolites
        self.valid_mets = _get_valid_metabolites(met_tic, min_obs=20)

        # Replace NaN with 0 for correlation computation
        met_filled = np.where(np.isnan(met_tic), 0, met_tic)

        # Vectorized correlation filter for all metabolites at once
        if verbose:
            print(f"  Computing gene-metabolite correlations...")
        top_genes_all = _vectorized_correlation_filter(rna_norm, met_filled, self.n_top_genes)

        # Fit Lasso for each valid metabolite
        for j in self.valid_mets:
            mask = ~np.isnan(met_tic[:, j])
            if mask.sum() < 20:
                continue

            y = met_tic[mask, j]
            gene_idx = top_genes_all[j]
            X = rna_norm[mask][:, gene_idx]

            sx = StandardScaler()
            sy = StandardScaler()
            X_s = sx.fit_transform(X)
            y_s = sy.fit_transform(y.reshape(-1, 1)).ravel()
            self.scalers_X[j] = sx
            self.scalers_y[j] = sy

            model = Lasso(alpha=self.alpha, max_iter=5000, random_state=self.seed)
            model.fit(X_s, y_s)
            self.models[j] = model
            self.selected_genes[j] = gene_idx

        if verbose:
            print(f"  Trained {len(self.models)}/{J_met} metabolite models")

    def predict_met(self, rna_data, batch_info):
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)
        n_samples = rna_data.shape[0]
        n_mets = max(self.models.keys()) + 1 if self.models else 0
        preds = np.full((n_samples, n_mets), np.nan)

        for j, model in self.models.items():
            gene_idx = self.selected_genes[j]
            X = rna_norm[:, gene_idx]
            X_s = self.scalers_X[j].transform(X)
            pred = model.predict(X_s)
            preds[:, j] = self.scalers_y[j].inverse_transform(pred.reshape(-1, 1)).ravel()

        return preds

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# Fast XGBoost Baseline
# ============================================================
class FastXGBoostBaseline:
    """Per-metabolite XGBoost with vectorized gene pre-filtering."""

    def __init__(self, n_top_genes=100, n_estimators=100, max_depth=4,
                 lr=0.1, seed=42, device='cpu', n_prefilter=500):
        self.n_top_genes = n_top_genes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.seed = seed
        self.n_prefilter = n_prefilter
        self.models = {}
        self.selected_genes = {}
        self.scalers_X = {}
        self.scalers_y = {}
        self.valid_mets = None

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        import xgboost as xgb

        N, J_met = met_data.shape
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        self.valid_mets = _get_valid_metabolites(met_tic, min_obs=20)
        met_filled = np.where(np.isnan(met_tic), 0, met_tic)

        # Vectorized correlation filter
        if verbose:
            print(f"  Computing gene-metabolite correlations...")
        top_genes_all = _vectorized_correlation_filter(rna_norm, met_filled, self.n_prefilter)

        for j in self.valid_mets:
            mask = ~np.isnan(met_tic[:, j])
            if mask.sum() < 20:
                continue

            y = met_tic[mask, j]
            pre_genes = top_genes_all[j]
            X_pre = rna_norm[mask][:, pre_genes]

            # Quick Lasso for final selection
            from sklearn.linear_model import LassoCV
            lasso = LassoCV(cv=3, max_iter=2000, n_alphas=10, random_state=self.seed)
            lasso.fit(X_pre, y)
            top_genes = pre_genes[np.argsort(np.abs(lasso.coef_))[-self.n_top_genes:]]
            self.selected_genes[j] = top_genes

            X_sel = rna_norm[mask][:, top_genes]
            sx = StandardScaler()
            sy = StandardScaler()
            X_sel = sx.fit_transform(X_sel)
            y_s = sy.fit_transform(y.reshape(-1, 1)).ravel()
            self.scalers_X[j] = sx
            self.scalers_y[j] = sy

            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.lr, random_state=self.seed,
                n_jobs=2, verbosity=0
            )
            model.fit(X_sel, y_s)
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
            gene_idx = self.selected_genes[j]
            X = rna_norm[:, gene_idx]
            X_s = self.scalers_X[j].transform(X)
            pred = model.predict(X_s)
            preds[:, j] = self.scalers_y[j].inverse_transform(pred.reshape(-1, 1)).ravel()

        return preds

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}


# ============================================================
# Fast Ridge Baseline (replaces Lasso for speed)
# ============================================================
class FastRidgeBaseline:
    """Multivariate Ridge regression: predict all metabolites at once."""

    def __init__(self, n_top_genes=500, seed=42, device='cpu'):
        self.n_top_genes = n_top_genes
        self.seed = seed
        self.model = None
        self.selected_genes = None
        self.scaler_X = None
        self.scaler_y = None
        self.valid_mets = None

    def fit(self, met_data, rna_data, batch_info, verbose=False):
        N, J_met = met_data.shape
        met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)

        self.valid_mets = _get_valid_metabolites(met_tic, min_obs=20)

        # Select top genes by variance
        gene_var = np.var(rna_norm, axis=0)
        self.selected_genes = np.argsort(gene_var)[-self.n_top_genes:]
        X = rna_norm[:, self.selected_genes]

        # Use only valid metabolites
        Y = met_tic[:, self.valid_mets]
        # Replace NaN with 0 for Ridge
        Y_filled = np.where(np.isnan(Y), 0, Y)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_s = self.scaler_X.fit_transform(X)
        Y_s = self.scaler_y.fit_transform(Y_filled)

        self.model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.model.fit(X_s, Y_s)

        if verbose:
            print(f"  Ridge: alpha={self.model.alpha_:.4f}, "
                  f"genes={self.n_top_genes}, mets={len(self.valid_mets)}")

    def predict_met(self, rna_data, batch_info):
        rna_log = log_transform_rna(rna_data)
        rna_norm = normalize_rna(rna_log)
        X = rna_norm[:, self.selected_genes]
        X_s = self.scaler_X.transform(X)
        Y_pred_s = self.model.predict(X_s)
        Y_pred = self.scaler_y.inverse_transform(Y_pred_s)

        n_samples = rna_data.shape[0]
        n_mets = max(self.valid_mets) + 1 if len(self.valid_mets) > 0 else 0
        preds = np.full((n_samples, n_mets), np.nan)
        preds[:, self.valid_mets] = Y_pred
        return preds

    def predict_met_ranks(self, rna_data, batch_info):
        pred = self.predict_met(rna_data, batch_info)
        ranks = rank_predictions_per_batch(pred, batch_info)
        return {'rank_hat_mean': ranks}
