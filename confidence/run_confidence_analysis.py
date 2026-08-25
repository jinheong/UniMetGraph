#!/usr/bin/env python3
"""
Confidence-Filtered Metabolite Prediction Analysis.

Runs k-fold CV for all 8 models, saves per-metabolite predictions and
confidence measures, then analyzes whether filtering to high-confidence
metabolites improves aggregate Spearman correlation.

Usage:
  # Run fast models (Lasso, Ridge, XGBoost, MIRTH, Kernel MKL)
  python run_confidence_analysis.py --models fast --n_folds 3

  # Run slow models (UnitedMet, UniGraph, GNN)
  python run_confidence_analysis.py --models slow --n_folds 2 --n_steps 50 --n_top_genes 5000

  # Run all models
  python run_confidence_analysis.py --models all

  # Run analysis only (after predictions are saved)
  python run_confidence_analysis.py --analyze
"""
import sys
import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/UnitedMet')

from unigraph.data.load_camp import load_camp_data
from unigraph.data.graph import construct_graph
from unigraph.data.preprocess import (
    preprocess_camp, tic_normalization_across, log_transform_rna, normalize_rna,
    count_obs, order_and_rank, rank_predictions_per_batch
)
from unigraph.evaluation.metrics import spearman_per_metabolite, mae_per_metabolite
from unigraph.models.baselines import (
    UnitedMetBaseline, SimplifiedGNN, MIRTHBaseline, KernelMKLBaseline
)
from unigraph.models.baselines_fast import (
    FastLassoBaseline, FastXGBoostBaseline, FastRidgeBaseline
)
from unigraph.models.ablations import UniGraphAblation

OUTPUT_DIR = "/mnt/results/confidence"
PRED_DIR = "/workspace/confidence_predictions"  # Local disk for npz files
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

FAST_MODELS = ['lasso', 'ridge', 'xgboost', 'mirth', 'kernel_mkl']
SLOW_MODELS = ['unitedmet', 'unigraph', 'gnn']

MODEL_DISPLAY = {
    'lasso': 'Lasso', 'ridge': 'Ridge', 'xgboost': 'XGBoost',
    'mirth': 'MIRTH', 'kernel_mkl': 'Kernel MKL',
    'unitedmet': 'UnitedMet', 'unigraph': 'UniGraph', 'gnn': 'Simplified GNN',
}


# ============================================================
# Helper functions
# ============================================================

def get_true_ranks(met_data, batch_info):
    met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
    n_batch = batch_info['n_batch']
    N, J_met = met_tic.shape
    n_obs = count_obs(met_tic, n_batch, J_met, batch_info['batch_index_vector'])
    _, ranks = order_and_rank(met_tic, n_obs, N, J_met, n_batch, batch_info['batch_index_vector'])
    return ranks


def reindex_batches(batch_info, sample_indices):
    old_biv = batch_info['batch_index_vector']
    new_biv = old_biv[sample_indices]
    unique_batches = np.unique(new_biv)
    batch_map = {old: new for new, old in enumerate(unique_batches)}
    new_biv_reindexed = np.array([batch_map[b] for b in new_biv])
    n_batch = len(unique_batches)
    start_row, stop_row, batch_names = [], [], []
    for new_b in range(n_batch):
        old_b = unique_batches[new_b]
        mask = new_biv_reindexed == new_b
        indices = np.where(mask)[0]
        start_row.append(indices[0])
        stop_row.append(indices[-1] + 1)
        batch_names.append(batch_info['batch_names'][old_b])
    return {
        'batch_index_vector': new_biv_reindexed,
        'start_row': np.array(start_row),
        'stop_row': np.array(stop_row),
        'batch_names': batch_names,
        'n_batch': n_batch,
    }


def subsample_genes(rna_data, n_top_genes=5000):
    gene_var = np.var(rna_data, axis=0)
    selected = np.argsort(gene_var)[-n_top_genes:]
    return selected


def compute_universal_confidence(train_met, train_rna, train_batch, met_anno, graph_data):
    """Compute model-independent confidence measures on training data."""
    met_tic = tic_normalization_across(train_met, train_batch['batch_index_vector'])
    J_met = met_tic.shape[1]

    # 1. Observation rate
    obs_rate = np.sum(~np.isnan(met_tic), axis=0) / met_tic.shape[0]

    # 2. Coefficient of variation
    with np.errstate(divide='ignore', invalid='ignore'):
        met_mean = np.nanmean(met_tic, axis=0)
        met_std = np.nanstd(met_tic, axis=0)
        coef_var = np.where(np.abs(met_mean) > 1e-10, met_std / np.abs(met_mean), 0)

    # 3. Max gene-metabolite correlation (on top 5000 genes by variance for speed)
    rna_log = log_transform_rna(train_rna)
    rna_norm = normalize_rna(rna_log)
    gene_var = np.var(rna_norm, axis=0)
    top_genes = np.argsort(gene_var)[-5000:]
    rna_sub = rna_norm[:, top_genes]

    met_filled = np.where(np.isnan(met_tic), 0, met_tic)
    # Vectorized Pearson correlation
    X_c = rna_sub - rna_sub.mean(axis=0)
    Y_c = met_filled - met_filled.mean(axis=0)
    num = X_c.T @ Y_c  # (J_rna, J_met)
    x_norm = np.sqrt((X_c ** 2).sum(axis=0))
    y_norm = np.sqrt((Y_c ** 2).sum(axis=0))
    denom = np.outer(x_norm, y_norm)
    denom[denom == 0] = 1.0
    corr_matrix = np.abs(num / denom)  # (J_rna, J_met)
    max_gene_corr = corr_matrix.max(axis=0)  # (J_met,)

    # 4. Network mapping
    network_mapped = np.zeros(J_met, dtype=float)
    if graph_data is not None and 'camp_to_hgem' in graph_data:
        hgem_name_to_idx = {name: i for i, name in enumerate(graph_data['hgem_met_ids'])}
        camp_to_hgem = graph_data['camp_to_hgem']
        for i in range(J_met):
            if i in camp_to_hgem and camp_to_hgem[i] in hgem_name_to_idx:
                network_mapped[i] = 1.0

    return {
        'obs_rate': obs_rate,
        'coef_var': coef_var,
        'max_gene_corr': max_gene_corr,
        'network_mapped': network_mapped,
    }


def compute_train_r2(model, model_name, train_met, train_rna, train_batch):
    """Compute per-metabolite training R² for different model types."""
    J_met = train_met.shape[1]
    train_r2 = np.full(J_met, np.nan)

    met_tic = tic_normalization_across(train_met, train_batch['batch_index_vector'])

    if model_name in ['lasso', 'xgboost']:
        # Per-metabolite models: predict on training data
        preds = model.predict_met(train_rna, train_batch)
        for j in model.models.keys():
            mask = ~np.isnan(met_tic[:, j])
            if mask.sum() < 5:
                continue
            y_true = met_tic[mask, j]
            y_pred = preds[mask, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot > 0:
                train_r2[j] = 1 - ss_res / ss_tot

    elif model_name in ['ridge', 'mirth', 'kernel_mkl']:
        # Multivariate models: predict on training data
        preds = model.predict_met(train_rna, train_batch)
        valid_mets = model.valid_mets if hasattr(model, 'valid_mets') and model.valid_mets is not None else range(J_met)
        for j in valid_mets:
            if j >= preds.shape[1]:
                continue
            mask = ~np.isnan(met_tic[:, j]) & ~np.isnan(preds[:, j])
            if mask.sum() < 5:
                continue
            y_true = met_tic[mask, j]
            y_pred = preds[mask, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot > 0:
                train_r2[j] = 1 - ss_res / ss_tot

    elif model_name in ['unitedmet', 'unigraph', 'gnn']:
        # MF models: R² of RNA→W RidgeCV mapping (same for all metabolites)
        if hasattr(model, 'rna_model') and model.rna_model is not None:
            rna_log = log_transform_rna(train_rna)
            if hasattr(model, 'rna_mean') and model.rna_mean is not None:
                rna_norm = normalize_rna(rna_log, model.rna_mean, model.rna_std)
            else:
                rna_norm = normalize_rna(rna_log)
            W_pred = model.rna_model.predict(rna_norm)
            W_true = model.W_loc if hasattr(model, 'W_loc') else model.W
            r2_w = 1 - np.sum((W_true - W_pred) ** 2) / np.sum((W_true - np.mean(W_true)) ** 2)
            train_r2[:] = r2_w  # Same R² for all metabolites

    return train_r2


def compute_posterior_var(model, model_name, rna_data, batch_info, n_samples=100, seed=42):
    """Compute posterior predictive variance for Bayesian models."""
    J_met = model.n_met if hasattr(model, 'n_met') else 0
    if J_met == 0:
        return np.full(1, np.nan)

    if model_name == 'unitedmet':
        np.random.seed(seed)
        N = rna_data.shape[0]
        K = model.W_loc.shape[1]

        rna_log = log_transform_rna(rna_data)
        if hasattr(model, 'rna_mean') and model.rna_mean is not None:
            rna_norm = normalize_rna(rna_log, model.rna_mean, model.rna_std)
        else:
            rna_norm = normalize_rna(rna_log)
        W_pred = model.rna_model.predict(rna_norm)

        # Sample W and H from posterior
        W_draws = np.random.normal(model.W_loc, model.W_scale, size=(n_samples, *model.W_loc.shape))
        H_draws = np.random.normal(model.H_loc, model.H_scale, size=(n_samples, *model.H_loc.shape))

        pred_ranks_samples = []
        for s in range(n_samples):
            X_met = W_draws[s] @ H_draws[:model.W_loc.shape[1], :]  # Use training W draws
            # Actually, use predicted W for test samples
            X_met = W_pred @ H_draws[s]
            ranks = rank_predictions_per_batch(X_met, batch_info)
            pred_ranks_samples.append(ranks)

        pred_ranks_samples = np.array(pred_ranks_samples)  # (n_samples, N, J_met)
        posterior_var = np.var(pred_ranks_samples, axis=0).mean(axis=0)  # (J_met,)
        return posterior_var

    elif model_name == 'unigraph':
        np.random.seed(seed)
        N = rna_data.shape[0]
        K = model.latent_dim

        rna_log = log_transform_rna(rna_data)
        if hasattr(model, 'rna_mean') and model.rna_mean is not None:
            rna_norm = normalize_rna(rna_log, model.rna_mean, model.rna_std)
        else:
            rna_norm = normalize_rna(rna_log)
        W_pred = model.rna_model.predict(rna_norm)

        # Build H_met from GNN + unmapped embeddings
        H_met = np.zeros((K, model.n_met))
        if model.use_graph:
            mapped_indices = sorted(model.met_to_gnn_idx.keys())
            gnn_indices = [model.met_to_gnn_idx[i] for i in mapped_indices]
            H_met[:, mapped_indices] = model.H_met_mapped[:, gnn_indices]
            if len(model.unmapped_met_indices) > 0:
                H_met_draws = np.random.normal(
                    model.H_met_unmapped_loc, model.H_met_unmapped_scale,
                    size=(n_samples, K, len(model.unmapped_met_indices)))
        else:
            H_met_draws = np.random.normal(
                model.H_met_all_loc, model.H_met_all_scale,
                size=(n_samples, K, model.n_met))

        pred_ranks_samples = []
        for s in range(n_samples):
            if model.use_graph:
                H_met_s = H_met.copy()
                if len(model.unmapped_met_indices) > 0:
                    H_met_s[:, model.unmapped_met_indices] = H_met_draws[s]
            else:
                H_met_s = H_met_draws[s]
            X_met = W_pred @ H_met_s
            ranks = rank_predictions_per_batch(X_met, batch_info)
            pred_ranks_samples.append(ranks)

        pred_ranks_samples = np.array(pred_ranks_samples)
        posterior_var = np.var(pred_ranks_samples, axis=0).mean(axis=0)
        return posterior_var

    else:
        return np.full(J_met, np.nan)


# ============================================================
# Model runners
# ============================================================

def run_fast_model(model_name, train_met, train_rna, train_batch,
                   test_rna, test_batch, seed):
    """Train and predict for a fast model."""
    if model_name == 'lasso':
        model = FastLassoBaseline(n_top_genes=200, alpha=0.01, seed=seed)
    elif model_name == 'ridge':
        model = FastRidgeBaseline(n_top_genes=500, seed=seed)
    elif model_name == 'xgboost':
        model = FastXGBoostBaseline(n_top_genes=50, n_estimators=50,
                                    n_prefilter=200, seed=seed)
    elif model_name == 'mirth':
        model = MIRTHBaseline(latent_dim=30, seed=seed)
    elif model_name == 'kernel_mkl':
        model = KernelMKLBaseline(latent_dim=30, seed=seed)
    else:
        raise ValueError(f"Unknown fast model: {model_name}")

    model.fit(train_met, train_rna, train_batch, verbose=False)
    preds = model.predict_met_ranks(test_rna, test_batch)
    return model, preds['rank_hat_mean']


def run_slow_model(model_name, train_met, train_rna, train_batch,
                   test_rna, test_batch, graph_data, seed,
                   n_steps=50, n_top_genes=5000):
    """Train and predict for a slow model."""
    # Subsample genes
    gene_idx = subsample_genes(train_rna, n_top_genes)
    train_rna_sub = train_rna[:, gene_idx]
    test_rna_sub = test_rna[:, gene_idx]

    # Preprocess for MF models
    preprocessed = preprocess_camp(train_met, train_rna_sub, train_batch)

    if model_name == 'unitedmet':
        model = UnitedMetBaseline(latent_dim=30, n_steps=n_steps, lr=0.001,
                                   seed=seed, device='cpu')
        model.fit(preprocessed, train_batch, verbose=False)
        preds = model.predict_met_ranks(rna_data=test_rna_sub, batch_info=test_batch)

    elif model_name == 'unigraph':
        model = UniGraphAblation(
            latent_dim=30, n_steps=n_steps, lr=0.001,
            device='cpu', seed=seed,
            use_graph=True, use_rank=True, use_bayesian=True, use_chemical=True)
        model.fit(preprocessed, graph_data, train_batch, verbose=False)
        preds = model.predict_met_ranks(rna_data=test_rna_sub, batch_info=test_batch)

    elif model_name == 'gnn':
        model = SimplifiedGNN(latent_dim=30, hidden_dim=256, n_heads=4,
                               n_layers=3, dropout=0.1, lr=0.001,
                               n_steps=n_steps, device='cpu', seed=seed)
        model.fit(train_met, train_rna_sub, graph_data, train_batch, verbose=False)
        preds = model.predict_met_ranks(rna_data=test_rna_sub, batch_info=test_batch)

    else:
        raise ValueError(f"Unknown slow model: {model_name}")

    return model, preds['rank_hat_mean'], gene_idx


# ============================================================
# Main runner
# ============================================================

def run_models(model_list, n_folds, seed=42, n_steps=50, n_top_genes=5000):
    """Run CV for specified models, saving predictions and confidence."""
    os.makedirs(PRED_DIR, exist_ok=True)

    print("Loading CAMP data...")
    met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)
    print(f"  met_data: {met_data.shape}, rna_data: {rna_data.shape}")

    print("Loading graph...")
    gene_names = batch_info.get('gene_names', None)
    if gene_names is None:
        rna_df = pd.read_csv(
            '/workspace/data/pancancer_metabolomics/data/transcriptomics_processed/Cornell_DLBCL.tpm.gene_symbol.csv',
            index_col=0)
        gene_names = list(rna_df.index)
    graph_data = construct_graph(met_anno, gene_names, rna_data)
    print(f"  Graph: {graph_data['fingerprints'].shape[0]} nodes")

    N = met_data.shape[0]
    np.random.seed(seed)
    indices = np.arange(N)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for model_name in model_list:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_DISPLAY.get(model_name, model_name)}")
        print(f"{'='*60}")

        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

            print(f"\n  Fold {fold_idx+1}/{n_folds} (train={len(train_idx)}, test={len(test_idx)})")

            train_met = met_data[train_idx]
            train_rna = rna_data[train_idx]
            test_met = met_data[test_idx]
            test_rna = rna_data[test_idx]
            train_batch = reindex_batches(batch_info, train_idx)
            test_batch = reindex_batches(batch_info, test_idx)

            t0 = time.time()

            # Run model
            if model_name in FAST_MODELS:
                model, pred_ranks = run_fast_model(
                    model_name, train_met, train_rna, train_batch,
                    test_rna, test_batch, seed=seed * 100 + fold_idx)
                gene_idx = None
            else:
                model, pred_ranks, gene_idx = run_slow_model(
                    model_name, train_met, train_rna, train_batch,
                    test_rna, test_batch, graph_data, seed=seed * 100 + fold_idx,
                    n_steps=n_steps, n_top_genes=n_top_genes)

            # True ranks
            true_ranks = get_true_ranks(test_met, test_batch)

            # Compute confidence measures
            print(f"  Computing confidence measures...", end=" ")
            universal_conf = compute_universal_confidence(
                train_met, train_rna, train_batch, met_anno, graph_data)

            train_r2 = compute_train_r2(
                model, model_name, train_met,
                train_rna[:, gene_idx] if gene_idx is not None else train_rna,
                train_batch)

            if model_name in ['unitedmet', 'unigraph']:
                posterior_var = compute_posterior_var(
                    model, model_name,
                    test_rna[:, gene_idx] if gene_idx is not None else test_rna,
                    test_batch, n_samples=50, seed=42)
            else:
                J_met = met_data.shape[1]
                posterior_var = np.full(J_met, np.nan)

            elapsed = time.time() - t0
            print(f"done ({elapsed:.0f}s)")

            # Save predictions and confidence
            save_path = os.path.join(PRED_DIR, f"{model_name}_fold{fold_idx}.npz")
            np.savez_compressed(
                save_path,
                pred_ranks=pred_ranks,
                true_ranks=true_ranks,
                test_indices=test_idx,
                train_r2=train_r2,
                obs_rate=universal_conf['obs_rate'],
                coef_var=universal_conf['coef_var'],
                max_gene_corr=universal_conf['max_gene_corr'],
                network_mapped=universal_conf['network_mapped'],
                posterior_var=posterior_var,
                met_names=np.array(met_names, dtype=object),
            )
            print(f"  Saved: {save_path}")

            # Quick metric
            rhos, valid = spearman_per_metabolite(true_ranks, pred_ranks)
            valid_rhos = rhos[valid]
            print(f"  ρ = {np.mean(valid_rhos):.4f} (n={int(valid.sum())})")

    print(f"\nAll models complete. Predictions saved to {PRED_DIR}/")


# ============================================================
# Analysis
# ============================================================

def analyze():
    """Analyze predictions: compute per-metabolite ρ, filter, generate figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams as rcParams
    from scipy.stats import spearmanr

    rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    rcParams['svg.fonttype'] = 'none'

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Phylo color palette
    COLORS = ['#0279EE', '#FF9400', '#75A025', '#FD9BED', '#E9ED4C',
              '#000000', '#ECE9E2', '#FAF9F3']

    # Load all prediction files
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('.npz')])
    if not pred_files:
        print("No prediction files found!")
        return

    print(f"Found {len(pred_files)} prediction files")

    # Group by model
    model_folds = {}
    met_names = None
    for f in pred_files:
        parts = f.replace('.npz', '').split('_fold')
        model_name = parts[0]
        fold_idx = int(parts[1])
        if model_name not in model_folds:
            model_folds[model_name] = []
        model_folds[model_name].append(fold_idx)

    print(f"Models: {list(model_folds.keys())}")

    # Load metabolite annotations
    met_data, rna_data, met_anno, met_names_raw, sample_info, batch_info = load_camp_data(tumor_only=True)
    J_met = met_data.shape[1]

    # ============================================================
    # 1. Pool predictions across folds and compute per-metabolite ρ
    # ============================================================
    print("\n--- Pooling predictions and computing per-metabolite ρ ---")

    per_met_results = {}  # model -> {rhos, valid, confidence measures}
    pooled_data = {}  # model -> {pred_pooled, true_pooled}

    for model_name in sorted(model_folds.keys()):
        folds = sorted(model_folds[model_name])
        all_pred = []
        all_true = []
        all_test_idx = []

        # Confidence measures (average across folds)
        conf_accum = {}

        for fold_idx in folds:
            fpath = os.path.join(PRED_DIR, f"{model_name}_fold{fold_idx}.npz")
            data = np.load(fpath, allow_pickle=True)
            all_pred.append(data['pred_ranks'])
            all_true.append(data['true_ranks'])
            all_test_idx.append(data['test_indices'])

            # Accumulate confidence (average across folds)
            for key in ['train_r2', 'obs_rate', 'coef_var', 'max_gene_corr',
                        'network_mapped', 'posterior_var']:
                if key not in conf_accum:
                    conf_accum[key] = []
                val = data[key]
                if not np.all(np.isnan(val)):
                    conf_accum[key].append(val)

        # Pool predictions
        pred_pooled = np.vstack(all_pred)
        true_pooled = np.vstack(all_true)
        test_idx_pooled = np.concatenate(all_test_idx)

        # Reorder by original sample index
        order = np.argsort(test_idx_pooled)
        pred_pooled = pred_pooled[order]
        true_pooled = true_pooled[order]

        pooled_data[model_name] = {
            'pred': pred_pooled,
            'true': true_pooled,
        }

        # Per-metabolite ρ
        rhos, valid = spearman_per_metabolite(true_pooled, pred_pooled, min_obs=5)

        # Average confidence across folds
        conf_avg = {}
        for key, vals in conf_accum.items():
            if vals:
                stacked = np.array(vals)
                conf_avg[key] = np.nanmean(stacked, axis=0)
            else:
                conf_avg[key] = np.full(J_met, np.nan)

        per_met_results[model_name] = {
            'rhos': rhos,
            'valid': valid,
            'confidence': conf_avg,
        }

        valid_rhos = rhos[valid]
        print(f"  {MODEL_DISPLAY.get(model_name, model_name):15s}: "
              f"ρ_mean={np.mean(valid_rhos):.4f}, ρ_median={np.median(valid_rhos):.4f}, "
              f"n_valid={int(valid.sum())}")

    # ============================================================
    # 2. Build per-metabolite table
    # ============================================================
    print("\n--- Building per-metabolite table ---")

    table_data = {'metabolite': met_names_raw}
    if met_anno is not None:
        if 'H_SUPER_PATHWAY' in met_anno.columns:
            table_data['super_pathway'] = met_anno['H_SUPER_PATHWAY'].values
        if 'H_SUB_PATHWAY' in met_anno.columns:
            table_data['sub_pathway'] = met_anno['H_SUB_PATHWAY'].values

    # Add per-metabolite ρ for each model
    for model_name in sorted(per_met_results.keys()):
        rhos = per_met_results[model_name]['rhos']
        table_data[f'rho_{model_name}'] = rhos

    # Add confidence measures (from first model that has them)
    for key in ['train_r2', 'obs_rate', 'coef_var', 'max_gene_corr',
                'network_mapped', 'posterior_var']:
        for model_name in sorted(per_met_results.keys()):
            conf = per_met_results[model_name]['confidence'].get(key)
            if conf is not None and not np.all(np.isnan(conf)):
                table_data[f'{key}_{model_name}'] = conf
                break  # Use first available

    # Also add universal measures (same across models, use lasso as reference)
    ref_model = 'lasso' if 'lasso' in per_met_results else list(per_met_results.keys())[0]
    for key in ['obs_rate', 'coef_var', 'max_gene_corr', 'network_mapped']:
        conf = per_met_results[ref_model]['confidence'].get(key)
        if conf is not None:
            table_data[key] = conf

    per_met_df = pd.DataFrame(table_data)
    per_met_df.to_csv(os.path.join(TABLES_DIR, 'per_metabolite_rho.csv'), index=False)
    print(f"  Saved per_metabolite_rho.csv ({len(per_met_df)} metabolites)")

    # ============================================================
    # 3. Post-hoc filtering (circular: use cv_rho to filter)
    # ============================================================
    print("\n--- Post-hoc filtering (upper bound) ---")

    thresholds = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    posthoc_results = []

    for model_name in sorted(per_met_results.keys()):
        rhos = per_met_results[model_name]['rhos']
        valid = per_met_results[model_name]['valid']

        valid_rhos = rhos[valid]
        n_valid = len(valid_rhos)

        for thresh in thresholds:
            n_keep = max(1, int(n_valid * thresh))
            # Sort by rho descending
            sorted_rhos = np.sort(valid_rhos)[::-1]
            kept_rhos = sorted_rhos[:n_keep]

            posthoc_results.append({
                'model': model_name,
                'model_display': MODEL_DISPLAY.get(model_name, model_name),
                'threshold': thresh,
                'n_metabolites': n_keep,
                'spearman_mean': np.mean(kept_rhos),
                'spearman_median': np.median(kept_rhos),
            })

    posthoc_df = pd.DataFrame(posthoc_results)
    posthoc_df.to_csv(os.path.join(TABLES_DIR, 'filtered_performance_posthoc.csv'), index=False)
    print(posthoc_df.pivot(index='threshold', columns='model_display', values='spearman_mean').to_string())

    # ============================================================
    # 4. Non-circular filtering (train-set confidence → test-set ρ)
    # ============================================================
    print("\n--- Non-circular filtering (train-set confidence) ---")

    confidence_measures = ['train_r2', 'max_gene_corr', 'obs_rate', 'coef_var', 'posterior_var']
    noncircular_results = []

    for model_name in sorted(per_met_results.keys()):
        rhos = per_met_results[model_name]['rhos']
        valid = per_met_results[model_name]['valid']
        conf = per_met_results[model_name]['confidence']

        valid_rhos = rhos[valid]
        n_valid = len(valid_rhos)

        for conf_name in confidence_measures:
            conf_vals = conf.get(conf_name)
            if conf_vals is None or np.all(np.isnan(conf_vals)):
                continue

            # Get confidence for valid metabolites only
            conf_valid = conf_vals[valid]
            # Remove NaN confidence
            conf_mask = ~np.isnan(conf_valid)
            if conf_mask.sum() < 10:
                continue

            conf_clean = conf_valid[conf_mask]
            rhos_clean = valid_rhos[conf_mask]

            for thresh in thresholds:
                n_keep = max(1, int(len(conf_clean) * thresh))
                # Sort by confidence descending
                sorted_idx = np.argsort(conf_clean)[::-1]
                kept_idx = sorted_idx[:n_keep]
                kept_rhos = rhos_clean[kept_idx]

                noncircular_results.append({
                    'model': model_name,
                    'model_display': MODEL_DISPLAY.get(model_name, model_name),
                    'confidence_measure': conf_name,
                    'threshold': thresh,
                    'n_metabolites': n_keep,
                    'spearman_mean': np.mean(kept_rhos),
                    'spearman_median': np.median(kept_rhos),
                })

    noncircular_df = pd.DataFrame(noncircular_results)
    noncircular_df.to_csv(os.path.join(TABLES_DIR, 'filtered_performance_noncircular.csv'), index=False)

    # Find best confidence measure per model
    best_conf = {}
    for model_name in sorted(per_met_results.keys()):
        model_df = noncircular_df[noncircular_df['model'] == model_name]
        if len(model_df) == 0:
            continue
        # Use threshold=0.10 to compare measures
        thresh_df = model_df[model_df['threshold'] == 0.10]
        if len(thresh_df) > 0:
            best = thresh_df.loc[thresh_df['spearman_mean'].idxmax()]
            best_conf[model_name] = best['confidence_measure']
            print(f"  {MODEL_DISPLAY.get(model_name, model_name):15s}: "
                  f"best confidence = {best['confidence_measure']}, "
                  f"ρ@10% = {best['spearman_mean']:.4f}")

    # ============================================================
    # 5. Confidence measure correlation with cv_rho
    # ============================================================
    print("\n--- Confidence measure correlations with cv_rho ---")

    conf_corr_results = []
    ref_rhos = per_met_results[ref_model]['rhos']
    ref_valid = per_met_results[ref_model]['valid']

    for conf_name in confidence_measures:
        # Use average confidence across models
        all_conf = []
        all_rho = []
        for model_name in sorted(per_met_results.keys()):
            conf = per_met_results[model_name]['confidence'].get(conf_name)
            rhos = per_met_results[model_name]['rhos']
            valid = per_met_results[model_name]['valid']
            if conf is None:
                continue
            conf_valid = conf[valid]
            rhos_valid = rhos[valid]
            mask = ~np.isnan(conf_valid) & ~np.isnan(rhos_valid)
            if mask.sum() > 10:
                all_conf.extend(conf_valid[mask])
                all_rho.extend(rhos_valid[mask])

        if len(all_conf) > 10:
            corr, pval = spearmanr(all_conf, all_rho)
            conf_corr_results.append({
                'confidence_measure': conf_name,
                'correlation_with_cv_rho': corr,
                'pvalue': pval,
                'n_samples': len(all_conf),
            })

    conf_corr_df = pd.DataFrame(conf_corr_results)
    conf_corr_df.to_csv(os.path.join(TABLES_DIR, 'confidence_measure_correlations.csv'), index=False)
    print(conf_corr_df.to_string(index=False))

    # ============================================================
    # 6. Top predictable metabolites
    # ============================================================
    print("\n--- Top predictable metabolites ---")

    # Find best model by mean ρ
    best_model = max(per_met_results.keys(),
                     key=lambda m: np.nanmean(per_met_results[m]['rhos'][per_met_results[m]['valid']]))
    best_rhos = per_met_results[best_model]['rhos']
    best_valid = per_met_results[best_model]['valid']

    # Sort by ρ
    valid_indices = np.where(best_valid)[0]
    valid_rhos = best_rhos[valid_indices]
    sorted_order = np.argsort(valid_rhos)[::-1]
    top_indices = valid_indices[sorted_order[:50]]

    top_data = {
        'rank': range(1, 51),
        'metabolite': [met_names_raw[i] for i in top_indices],
        'super_pathway': [met_anno['H_SUPER_PATHWAY'].iloc[i] if met_anno is not None else '' for i in top_indices],
        'sub_pathway': [met_anno['H_SUB_PATHWAY'].iloc[i] if met_anno is not None else '' for i in top_indices],
    }
    # Add ρ for all models
    for model_name in sorted(per_met_results.keys()):
        top_data[f'rho_{model_name}'] = [per_met_results[model_name]['rhos'][i] for i in top_indices]
    # Add confidence
    for key in ['obs_rate', 'max_gene_corr', 'coef_var', 'network_mapped']:
        conf = per_met_results[ref_model]['confidence'].get(key)
        if conf is not None:
            top_data[key] = [conf[i] for i in top_indices]

    top_df = pd.DataFrame(top_data)
    top_df.to_csv(os.path.join(TABLES_DIR, 'top_predictable_metabolites.csv'), index=False)
    print(f"  Top 5 ({MODEL_DISPLAY.get(best_model, best_model)}):")
    print(top_df[['rank', 'metabolite', 'super_pathway', f'rho_{best_model}']].head(5).to_string(index=False))

    # ============================================================
    # 7. Generate figures
    # ============================================================
    print("\n--- Generating figures ---")

    model_order = [m for m in ['unigraph', 'unitedmet', 'gnn', 'xgboost', 'lasso',
                                'ridge', 'mirth', 'kernel_mkl'] if m in per_met_results]
    n_models = len(model_order)
    model_colors = COLORS[:n_models]

    # Fig 1: Per-metabolite ρ distribution (violin plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    violin_data = []
    violin_labels = []
    violin_colors = []
    for i, m in enumerate(model_order):
        rhos = per_met_results[m]['rhos']
        valid = per_met_results[m]['valid']
        valid_rhos = rhos[valid]
        valid_rhos = valid_rhos[~np.isnan(valid_rhos)]
        violin_data.append(valid_rhos)
        violin_labels.append(MODEL_DISPLAY.get(m, m))
        violin_colors.append(model_colors[i])

    parts = ax.violinplot(violin_data, positions=range(n_models), showmeans=True,
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_alpha(0.6)
    for key in ['cmeans', 'cmedians']:
        if key in parts:
            parts[key].set_color('black')
            parts[key].set_linewidth(1.5)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(violin_labels, rotation=30, ha='right')
    ax.set_ylabel('Per-metabolite Spearman ρ')
    ax.set_title('Distribution of Per-metabolite Prediction Accuracy')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.3, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'per_met_rho_distribution.svg'), format='svg')
    plt.savefig(os.path.join(FIGURES_DIR, 'per_met_rho_distribution.png'), format='png', dpi=150)
    plt.close()
    print("  Saved: per_met_rho_distribution")

    # Fig 2: Performance vs fraction retained (post-hoc)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(model_order):
        model_df = posthoc_df[posthoc_df['model'] == m]
        ax.plot(model_df['threshold'] * 100, model_df['spearman_mean'],
                'o-', color=model_colors[i], label=MODEL_DISPLAY.get(m, m), linewidth=2, markersize=6)
    ax.set_xlabel('Fraction of Metabolites Retained (%)')
    ax.set_ylabel('Aggregate Spearman ρ (mean)')
    ax.set_title('Performance vs. Metabolite Filtering (Post-hoc, Upper Bound)')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-2, 102)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_vs_fraction_posthoc.svg'), format='svg')
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_vs_fraction_posthoc.png'), format='png', dpi=150)
    plt.close()
    print("  Saved: performance_vs_fraction_posthoc")

    # Fig 3: Performance vs fraction retained (non-circular, best measure per model)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(model_order):
        if m not in best_conf:
            continue
        conf_name = best_conf[m]
        model_df = noncircular_df[(noncircular_df['model'] == m) &
                                   (noncircular_df['confidence_measure'] == conf_name)]
        if len(model_df) == 0:
            continue
        ax.plot(model_df['threshold'] * 100, model_df['spearman_mean'],
                'o-', color=model_colors[i],
                label=f"{MODEL_DISPLAY.get(m, m)} ({conf_name})",
                linewidth=2, markersize=6)
    ax.set_xlabel('Fraction of Metabolites Retained (%)')
    ax.set_ylabel('Aggregate Spearman ρ (mean)')
    ax.set_title('Performance vs. Metabolite Filtering (Non-circular, Best Train-set Confidence)')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-2, 102)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_vs_fraction_noncircular.svg'), format='svg')
    plt.savefig(os.path.join(FIGURES_DIR, 'performance_vs_fraction_noncircular.png'), format='png', dpi=150)
    plt.close()
    print("  Saved: performance_vs_fraction_noncircular")

    # Fig 4: Confidence measure comparison (bar plot)
    if len(conf_corr_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(conf_corr_df['confidence_measure'],
                        conf_corr_df['correlation_with_cv_rho'],
                        color=[COLORS[i % len(COLORS)] for i in range(len(conf_corr_df))])
        ax.set_xlabel('Spearman Correlation with Per-metabolite ρ')
        ax.set_title('Confidence Measure Predictive Power')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, conf_corr_df['correlation_with_cv_rho']):
            ax.text(val + 0.01 if val > 0 else val - 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9,
                    ha='left' if val > 0 else 'right')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'confidence_measure_comparison.svg'), format='svg')
        plt.savefig(os.path.join(FIGURES_DIR, 'confidence_measure_comparison.png'), format='png', dpi=150)
        plt.close()
        print("  Saved: confidence_measure_comparison")

    # Fig 5: Top predictable metabolites heatmap
    top_n = 30
    top_indices_heat = top_indices[:top_n]
    heat_data = np.zeros((top_n, len(model_order)))
    for j, m in enumerate(model_order):
        for i, idx in enumerate(top_indices_heat):
            heat_data[i, j] = per_met_results[m]['rhos'][idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=-0.2, vmax=0.5)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_order], rotation=30, ha='right')
    # Y-axis: metabolite names (truncated)
    met_labels = []
    for idx in top_indices_heat:
        name = met_names_raw[idx]
        if len(name) > 35:
            name = name[:32] + '...'
        sp = met_anno['H_SUPER_PATHWAY'].iloc[idx] if met_anno is not None else ''
        met_labels.append(f"{name} [{sp}]" if sp and str(sp) != 'nan' else name)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(met_labels, fontsize=7)
    plt.colorbar(im, ax=ax, label='Spearman ρ')
    ax.set_title(f'Top {top_n} Most Predictable Metabolites')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'top_predictable_metabolites.svg'), format='svg')
    plt.savefig(os.path.join(FIGURES_DIR, 'top_predictable_metabolites.png'), format='png', dpi=150)
    plt.close()
    print("  Saved: top_predictable_metabolites")

    # Fig 6: Pathway analysis (bar plot of mean ρ by super pathway)
    if met_anno is not None and 'H_SUPER_PATHWAY' in met_anno.columns:
        pathway_data = []
        for pathway in met_anno['H_SUPER_PATHWAY'].dropna().unique():
            if str(pathway) == 'nan':
                continue
            mask = (met_anno['H_SUPER_PATHWAY'] == pathway).values
            for m in model_order:
                rhos = per_met_results[m]['rhos']
                valid = per_met_results[m]['valid']
                pathway_rhos = rhos[mask & valid]
                pathway_rhos = pathway_rhos[~np.isnan(pathway_rhos)]
                if len(pathway_rhos) > 5:
                    pathway_data.append({
                        'pathway': pathway,
                        'model': m,
                        'model_display': MODEL_DISPLAY.get(m, m),
                        'mean_rho': np.mean(pathway_rhos),
                        'median_rho': np.median(pathway_rhos),
                        'n_mets': len(pathway_rhos),
                    })

        pathway_df = pd.DataFrame(pathway_data)
        pathway_df.to_csv(os.path.join(TABLES_DIR, 'pathway_analysis.csv'), index=False)

        # Plot for best model
        best_pathway = pathway_df[pathway_df['model'] == best_model].sort_values('mean_rho', ascending=True)
        if len(best_pathway) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(best_pathway['pathway'], best_pathway['mean_rho'],
                           color=model_colors[model_order.index(best_model)])
            ax.set_xlabel('Mean Spearman ρ')
            ax.set_title(f'Metabolite Predictability by Super Pathway ({MODEL_DISPLAY.get(best_model, best_model)})')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            for bar, val, n in zip(bars, best_pathway['mean_rho'], best_pathway['n_mets']):
                ax.text(val + 0.001 if val > 0 else val - 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f} (n={n})', va='center', fontsize=8,
                        ha='left' if val > 0 else 'right')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'pathway_analysis.svg'), format='svg')
            plt.savefig(os.path.join(FIGURES_DIR, 'pathway_analysis.png'), format='png', dpi=150)
            plt.close()
            print("  Saved: pathway_analysis")

    print(f"\n=== Analysis complete ===")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"Tables: {TABLES_DIR}/")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-Filtered Metabolite Prediction Analysis")
    parser.add_argument('--models', type=str, default='all',
                        help='fast, slow, all, or comma-separated names')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='SVI steps for slow models')
    parser.add_argument('--n_top_genes', type=int, default=5000,
                        help='Gene subsampling for slow models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis only (after predictions are saved)')
    args = parser.parse_args()

    if args.analyze:
        analyze()
    else:
        if args.models == 'fast':
            model_list = FAST_MODELS
        elif args.models == 'slow':
            model_list = SLOW_MODELS
        elif args.models == 'all':
            model_list = FAST_MODELS + SLOW_MODELS
        else:
            model_list = args.models.split(',')

        run_models(model_list, args.n_folds, seed=args.seed,
                   n_steps=args.n_steps, n_top_genes=args.n_top_genes)
