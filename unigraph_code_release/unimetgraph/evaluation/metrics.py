"""
Evaluation metrics for transcriptome-to-metabolome prediction.
"""
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple


def spearman_per_metabolite(true_ranks: np.ndarray, pred_ranks: np.ndarray,
                             min_obs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-metabolite Spearman correlation.
    Returns (rhos, valid_mask) where valid_mask indicates metabolites with enough observations.
    """
    n_mets = true_ranks.shape[1]
    rhos = np.full(n_mets, np.nan)
    valid = np.zeros(n_mets, dtype=bool)

    for j in range(n_mets):
        mask = ~np.isnan(true_ranks[:, j]) & ~np.isnan(pred_ranks[:, j])
        if mask.sum() >= min_obs:
            rho, pval = spearmanr(true_ranks[mask, j], pred_ranks[mask, j])
            if not np.isnan(rho):
                rhos[j] = rho
                valid[j] = True

    return rhos, valid


def mae_per_metabolite(true_ranks: np.ndarray, pred_ranks: np.ndarray,
                       min_obs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-metabolite Mean Absolute Error."""
    n_mets = true_ranks.shape[1]
    maes = np.full(n_mets, np.nan)
    valid = np.zeros(n_mets, dtype=bool)

    for j in range(n_mets):
        mask = ~np.isnan(true_ranks[:, j]) & ~np.isnan(pred_ranks[:, j])
        if mask.sum() >= min_obs:
            maes[j] = np.mean(np.abs(true_ranks[mask, j] - pred_ranks[mask, j]))
            valid[j] = True

    return maes, valid


def r2_per_metabolite(true_ranks: np.ndarray, pred_ranks: np.ndarray,
                      min_obs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-metabolite R² score."""
    n_mets = true_ranks.shape[1]
    r2s = np.full(n_mets, np.nan)
    valid = np.zeros(n_mets, dtype=bool)

    for j in range(n_mets):
        mask = ~np.isnan(true_ranks[:, j]) & ~np.isnan(pred_ranks[:, j])
        if mask.sum() >= min_obs:
            y_true = true_ranks[mask, j]
            y_pred = pred_ranks[mask, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot > 0:
                r2s[j] = 1 - ss_res / ss_tot
                valid[j] = True

    return r2s, valid


def summarize_metrics(rhos: np.ndarray, maes: np.ndarray = None,
                      r2s: np.ndarray = None) -> Dict:
    """Summarize metrics across metabolites."""
    valid_rhos = rhos[~np.isnan(rhos)]
    result = {
        'spearman_mean': np.mean(valid_rhos),
        'spearman_median': np.median(valid_rhos),
        'spearman_std': np.std(valid_rhos),
        'spearman_n_valid': len(valid_rhos),
    }
    if maes is not None:
        valid_maes = maes[~np.isnan(maes)]
        result['mae_mean'] = np.mean(valid_maes)
        result['mae_median'] = np.median(valid_maes)
    if r2s is not None:
        valid_r2s = r2s[~np.isnan(r2s)]
        result['r2_mean'] = np.mean(valid_r2s)
        result['r2_median'] = np.median(valid_r2s)
    return result


def evaluate_predictions(true_ranks: np.ndarray, pred_ranks: np.ndarray,
                         metrics: List[str] = ['spearman', 'mae']) -> Dict:
    """Full evaluation of predictions."""
    results = {}

    if 'spearman' in metrics:
        rhos, valid = spearman_per_metabolite(true_ranks, pred_ranks)
        results['spearman'] = rhos
        results['spearman_valid'] = valid

    if 'mae' in metrics:
        maes, valid = mae_per_metabolite(true_ranks, pred_ranks)
        results['mae'] = maes
        results['mae_valid'] = valid

    if 'r2' in metrics:
        r2s, valid = r2_per_metabolite(true_ranks, pred_ranks)
        results['r2'] = r2s
        results['r2_valid'] = valid

    results['summary'] = summarize_metrics(
        results.get('spearman', np.array([np.nan])),
        results.get('mae'),
        results.get('r2')
    )

    return results
