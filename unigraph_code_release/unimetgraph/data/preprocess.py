"""
Data preprocessing: TIC normalization, rank transformation, gene expression normalization.
Adapted from UnitedMet's data_processing.py.
"""
import numpy as np
from typing import Tuple, Dict


def tic_normalization_across(data: np.ndarray, batch_index_vector: np.ndarray) -> np.ndarray:
    """
    TIC (Total Ion Current) normalization for metabolomics data, per batch.
    Each row is normalized so that the sum of observed values equals 1.
    """
    normalized_data = np.copy(data).astype(float)
    n_batches = batch_index_vector.max() + 1
    for bidx in range(n_batches):
        batch_rows = np.arange(data.shape[0])[batch_index_vector == bidx]
        batch = data[batch_rows]
        nan_mask = np.isnan(batch)
        missing = np.all(nan_mask, axis=0)
        min_batch = np.nanmin(batch) if not np.all(np.isnan(batch)) else 0.0

        for row in range(batch.shape[0]):
            n_censored = np.sum(nan_mask[row]) - np.sum(missing)
            row_tic = np.nansum(batch[row]) + 0.5 * n_censored * min_batch
            if row_tic > 0:
                batch[row, :] = batch[row, :] / row_tic

        # Normalize each row to sum to 1
        row_sums = np.nansum(batch, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        batch = batch / row_sums
        normalized_data[batch_rows] = batch

    return normalized_data


def count_obs(data: np.ndarray, n_batch: int, J: int, batch_index_vector: np.ndarray) -> np.ndarray:
    """Count non-NaN values per batch per metabolite."""
    n_obs = np.zeros(shape=[n_batch, J]).astype(int)
    for b in range(n_batch):
        n_obs[b, :] = np.sum(~np.isnan(data[batch_index_vector == b]), axis=0)
    return n_obs


def order_and_rank(data: np.ndarray, n_obs: np.ndarray, N: int, J: int,
                   n_batch: int, batch_index_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous data to orders and ranks within each batch.
    Order: indices of items from largest to smallest.
    Rank: rank of each item (0 = largest, N-1 = smallest).
    """
    orders = np.zeros([N, J]).astype(int)
    ranks = np.zeros([N, J]).astype(int)

    for b in range(n_batch):
        batch = data[batch_index_vector == b]
        batch = np.where(np.isnan(batch), 0, batch)
        indices = batch.argsort(axis=0, kind='stable')
        orders[batch_index_vector == b] = indices[::-1]
        ranks_temp = batch.argsort(axis=0, kind='stable')[::-1].argsort(axis=0, kind='stable')
        ranks[batch_index_vector == b] = np.where(ranks_temp > n_obs[b, :], n_obs[b, :], ranks_temp)

    return orders, ranks


def rank_predictions_per_batch(pred: np.ndarray, batch_info: Dict) -> np.ndarray:
    """
    Rank-transform predictions per batch using boolean mask indexing.
    Handles non-contiguous batch samples correctly.
    Highest prediction -> rank 0, lowest -> rank N-1.
    """
    biv = batch_info['batch_index_vector']
    n_samples, n_mets = pred.shape
    ranks = np.full((n_samples, n_mets), np.nan, dtype=float)
    for b in np.unique(biv):
        mask = biv == b
        batch_pred = pred[mask]
        valid = ~np.isnan(batch_pred).all(axis=0)
        if valid.any():
            batch_ranks = batch_pred[:, valid].argsort(
                axis=0, kind='stable')[::-1].argsort(axis=0, kind='stable')
            ranks[np.ix_(mask, valid)] = batch_ranks
    return ranks


def log_transform_rna(rna_data: np.ndarray, pseudo_count: float = 1.0) -> np.ndarray:
    """Log-transform RNA-seq TPM data: log2(TPM + pseudo_count)."""
    return np.log2(rna_data + pseudo_count)


def normalize_rna(rna_data: np.ndarray, mean=None, std=None) -> np.ndarray:
    """Z-score normalize RNA-seq data per gene.
    If mean/std provided, use them (for test set normalization)."""
    if mean is None:
        mean = np.nanmean(rna_data, axis=0, keepdims=True)
    if std is None:
        std = np.nanstd(rna_data, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (rna_data - mean) / std


def preprocess_camp(met_data: np.ndarray, rna_data: np.ndarray,
                    batch_info: Dict) -> Dict:
    """
    Full preprocessing pipeline for CAMP data.

    Returns dict with:
        met_data_tic: TIC-normalized metabolomics
        rna_data_log: log-transformed + z-scored RNA-seq
        orders: rank orders for Plackett-Luce
        ranks: rank values
        n_obs: observation counts per batch per metabolite
        met_data_raw: original metabolomics (for evaluation)
    """
    batch_index_vector = batch_info['batch_index_vector']
    n_batch = batch_info['n_batch']
    N, J_met = met_data.shape
    J_rna = rna_data.shape[1]
    J = J_met + J_rna

    # TIC normalize metabolomics
    print("TIC normalizing metabolomics...")
    met_data_tic = tic_normalization_across(met_data, batch_index_vector)

    # Log-transform and normalize RNA-seq
    print("Normalizing RNA-seq...")
    rna_data_log = log_transform_rna(rna_data)
    rna_mean = np.nanmean(rna_data_log, axis=0, keepdims=True)
    rna_std = np.nanstd(rna_data_log, axis=0, keepdims=True)
    rna_std[rna_std == 0] = 1.0
    rna_data_norm = (rna_data_log - rna_mean) / rna_std

    # Concatenate met + rna for rank transformation
    # (UnitedMet ranks both metabolites and genes together)
    data_combined = np.concatenate([met_data_tic, rna_data_norm], axis=1)

    # Count observations and compute orders/ranks
    print("Computing ranks and orders...")
    n_obs = count_obs(data_combined, n_batch, J, batch_index_vector)
    orders, ranks = order_and_rank(data_combined, n_obs, N, J, n_batch, batch_index_vector)

    print(f"Preprocessed data: met={met_data_tic.shape}, rna={rna_data_norm.shape}")
    print(f"Combined for ranking: {data_combined.shape}")

    return {
        'met_data_tic': met_data_tic,
        'rna_data_norm': rna_data_norm,
        'rna_mean': rna_mean,
        'rna_std': rna_std,
        'data_combined': data_combined,
        'orders': orders,
        'ranks': ranks,
        'n_obs': n_obs,
        'N': N,
        'J_met': J_met,
        'J_rna': J_rna,
        'J': J,
    }


def preprocess_for_prediction(met_data: np.ndarray, rna_data: np.ndarray,
                               batch_info: Dict, met_only: bool = False) -> Dict:
    """
    Preprocess data for prediction (rank transform only metabolites or met+gene).
    Used for evaluation where we predict metabolite ranks from gene expression.

    Args:
        met_only: If True, only rank-transform metabolites (for evaluation).
                  If False, rank-transform met+gene (for training).
    """
    batch_index_vector = batch_info['batch_index_vector']
    n_batch = batch_info['n_batch']
    N, J_met = met_data.shape

    # TIC normalize metabolomics
    met_data_tic = tic_normalization_across(met_data, batch_index_vector)

    # Log-transform and normalize RNA-seq
    rna_data_log = log_transform_rna(rna_data)
    rna_data_norm = normalize_rna(rna_data_log)

    if met_only:
        # Only rank metabolites
        data = met_data_tic
        J = J_met
    else:
        # Rank met + gene
        data = np.concatenate([met_data_tic, rna_data_norm], axis=1)
        J = J_met + rna_data.shape[1]

    n_obs = count_obs(data, n_batch, J, batch_index_vector)
    orders, ranks = order_and_rank(data, n_obs, N, J, n_batch, batch_index_vector)

    return {
        'met_data_tic': met_data_tic,
        'rna_data_norm': rna_data_norm,
        'orders': orders,
        'ranks': ranks,
        'n_obs': n_obs,
        'N': N,
        'J_met': J_met,
        'J': J,
    }
