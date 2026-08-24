#!/usr/bin/env python3
"""
Comprehensive benchmark: UniGraph vs all baselines.
Runs in-distribution CV, zero-shot LOMO, and cross-dataset validation.
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import warnings
import traceback
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/UnitedMet')

from unigraph.data.load_camp import load_camp_data
from unigraph.data.load_ccrcc import load_ccrcc_data
from unigraph.data.graph import construct_graph
from unigraph.data.preprocess import (
    preprocess_camp, preprocess_for_prediction,
    tic_normalization_across, log_transform_rna, normalize_rna,
    count_obs, order_and_rank
)
from unigraph.evaluation.metrics import evaluate_predictions, spearman_per_metabolite, mae_per_metabolite, r2_per_metabolite
from unigraph.models.unigraph import UniGraphModel
from unigraph.models.baselines import UnitedMetBaseline, SimplifiedGNN, MIRTHBaseline, KernelMKLBaseline
from unigraph.models.baselines_fast import FastLassoBaseline, FastXGBoostBaseline, FastRidgeBaseline

RESULTS_DIR = "/mnt/results/benchmark"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_true_ranks(met_data, batch_info):
    """Compute true ranks from metabolomics data."""
    met_tic = tic_normalization_across(met_data, batch_info['batch_index_vector'])
    n_batch = batch_info['n_batch']
    N, J_met = met_tic.shape
    n_obs = count_obs(met_tic, n_batch, J_met, batch_info['batch_index_vector'])
    _, ranks = order_and_rank(met_tic, n_obs, N, J_met, n_batch, batch_info['batch_index_vector'])
    return ranks


def reindex_batches(batch_info, sample_indices):
    """Reindex batch_info for a subset of samples."""
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


def run_model(model_name, met_data, rna_data, batch_info, graph_data,
              train_idx=None, test_idx=None, met_mask_idx=None,
              latent_dim=30, n_steps=1000, seed=42, device='cpu'):
    """Train and predict with a single model."""
    # Split data
    if train_idx is not None:
        train_met = met_data[train_idx]
        train_rna = rna_data[train_idx]
        test_met = met_data[test_idx]
        test_rna = rna_data[test_idx]
        train_batch = reindex_batches(batch_info, train_idx)
    else:
        train_met = met_data
        train_rna = rna_data
        test_met = met_data
        test_rna = rna_data
        train_batch = batch_info

    # Mask metabolites for LOMO
    if met_mask_idx is not None:
        train_met_masked = train_met.copy()
        train_met_masked[:, met_mask_idx] = np.nan
    else:
        train_met_masked = train_met

    # Create model
    if model_name == 'unigraph':
        model = UniGraphModel(latent_dim=latent_dim, n_steps=n_steps, lr=0.001, device=device, seed=seed)
    elif model_name == 'unitedmet':
        model = UnitedMetBaseline(latent_dim=latent_dim, n_steps=n_steps, lr=0.001, seed=seed, device=device)
    elif model_name == 'gnn':
        model = SimplifiedGNN(latent_dim=latent_dim, n_steps=n_steps, lr=0.001, device=device, seed=seed)
    elif model_name == 'xgboost':
        model = FastXGBoostBaseline(seed=seed, n_top_genes=100, n_estimators=100, max_depth=4)
    elif model_name == 'lasso':
        model = FastLassoBaseline(seed=seed, n_top_genes=200, alpha=0.01)
    elif model_name == 'ridge':
        model = FastRidgeBaseline(seed=seed, n_top_genes=500)
    elif model_name == 'mirth':
        model = MIRTHBaseline(latent_dim=latent_dim, seed=seed, device=device)
    elif model_name == 'kernel_mkl':
        model = KernelMKLBaseline(latent_dim=latent_dim, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    if model_name in ['unigraph', 'unitedmet']:
        preprocessed = preprocess_camp(train_met_masked, train_rna, train_batch)
        if model_name == 'unigraph':
            model.fit(preprocessed, graph_data, train_batch, verbose=False)
        else:
            model.fit(preprocessed, train_batch, verbose=False)
        # Predict for test samples using gene expression
        test_batch = reindex_batches(batch_info, test_idx) if test_idx is not None else batch_info
        preds = model.predict_met_ranks(rna_data=test_rna, batch_info=test_batch, n_samples=500, seed=42)
        pred_ranks = preds['rank_hat_mean']
        true_ranks = get_true_ranks(test_met, test_batch)

    elif model_name == 'gnn':
        model.fit(train_met_masked, train_rna, graph_data, train_batch, verbose=False)
        test_batch = reindex_batches(batch_info, test_idx) if test_idx is not None else batch_info
        preds = model.predict_met_ranks(test_rna, test_batch)
        pred_ranks = preds['rank_hat_mean']
        true_ranks = get_true_ranks(test_met, test_batch)

    else:
        model.fit(train_met_masked, train_rna, train_batch, verbose=False)
        test_batch = reindex_batches(batch_info, test_idx) if test_idx is not None else batch_info
        preds = model.predict_met_ranks(test_rna, test_batch)
        pred_ranks = preds['rank_hat_mean']
        true_ranks = get_true_ranks(test_met, test_batch)

    return true_ranks, pred_ranks


# ============================================================
# Protocol 1: In-distribution CV
# ============================================================
def run_indist_cv(models, met_data, rna_data, batch_info, graph_data,
                  latent_dim=30, n_steps=1000, n_folds=5, n_seeds=3, device='cpu'):
    """Run 5-fold stratified CV on samples for all models."""
    N = met_data.shape[0]
    all_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        indices = np.arange(N)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

            for model_name in models:
                print(f"  [{model_name}] Seed {seed}, Fold {fold_idx+1}/{n_folds}", end="... ")
                t0 = time.time()
                try:
                    true_ranks, pred_ranks = run_model(
                        model_name, met_data, rna_data, batch_info, graph_data,
                        train_idx=train_idx, test_idx=test_idx,
                        latent_dim=latent_dim, n_steps=n_steps,
                        seed=seed*100+fold_idx, device=device
                    )
                    rhos, valid = spearman_per_metabolite(true_ranks, pred_ranks)
                    maes, _ = mae_per_metabolite(true_ranks, pred_ranks)
                    valid_rhos = rhos[valid]
                    result = {
                        'model': model_name, 'seed': seed, 'fold': fold_idx,
                        'spearman_mean': np.mean(valid_rhos),
                        'spearman_median': np.median(valid_rhos),
                        'spearman_std': np.std(valid_rhos),
                        'n_valid_mets': int(valid.sum()),
                        'mae_mean': np.mean(maes[valid]),
                        'mae_median': np.median(maes[valid]),
                        'runtime_sec': time.time() - t0,
                    }
                    all_results.append(result)
                    print(f"ρ={result['spearman_mean']:.3f} ({result['runtime_sec']:.0f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        'model': model_name, 'seed': seed, 'fold': fold_idx,
                        'error': str(e), 'runtime_sec': time.time() - t0
                    })

                # Save incrementally
                pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/indist_cv.csv', index=False)

    return pd.DataFrame(all_results)


# ============================================================
# Protocol 2: Zero-shot LOMO
# ============================================================
def run_lomo(models, met_data, rna_data, batch_info, graph_data,
             latent_dim=30, n_steps=1000, n_seeds=3, device='cpu'):
    """Run leave-one-metabolite-out zero-shot evaluation."""
    J_met = met_data.shape[1]
    all_results = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        met_obs_count = np.sum(~np.isnan(met_data), axis=0)
        valid_mets = np.where(met_obs_count >= 20)[0]

        for model_name in models:
            n_hold = 50 if model_name in ['unigraph', 'gnn'] else 10
            held_out = np.random.choice(valid_mets, size=min(n_hold, len(valid_mets)), replace=False)

            print(f"  [{model_name}] Seed {seed}, LOMO ({len(held_out)} mets)", end="... ")
            t0 = time.time()
            try:
                # Zero-shot methods: mask held-out metabolites during training
                # Non-zero-shot methods: train on all metabolites, evaluate on held-out subset
                is_zero_shot = model_name in ['unigraph', 'gnn']
                mask_idx = held_out if is_zero_shot else None

                true_ranks, pred_ranks = run_model(
                    model_name, met_data, rna_data, batch_info, graph_data,
                    met_mask_idx=mask_idx,
                    latent_dim=latent_dim, n_steps=n_steps,
                    seed=seed, device=device
                )
                # Evaluate only on held-out metabolites
                true_held = true_ranks[:, held_out]
                pred_held = pred_ranks[:, held_out]
                rhos, valid = spearman_per_metabolite(true_held, pred_held)
                r2s, _ = r2_per_metabolite(true_held, pred_held)
                valid_rhos = rhos[valid]
                valid_r2s = r2s[valid]
                result = {
                    'model': model_name, 'seed': seed,
                    'n_held_out': len(held_out),
                    'spearman_mean': np.mean(valid_rhos) if len(valid_rhos) > 0 else np.nan,
                    'spearman_median': np.median(valid_rhos) if len(valid_rhos) > 0 else np.nan,
                    'r2_mean': np.mean(valid_r2s) if len(valid_r2s) > 0 else np.nan,
                    'r2_median': np.median(valid_r2s) if len(valid_r2s) > 0 else np.nan,
                    'n_valid': int(valid.sum()),
                    'runtime_sec': time.time() - t0,
                }
                all_results.append(result)
                print(f"ρ={result['spearman_mean']:.3f}, R²={result['r2_mean']:.3f} ({result['runtime_sec']:.0f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                all_results.append({
                    'model': model_name, 'seed': seed,
                    'error': str(e), 'runtime_sec': time.time() - t0
                })

            pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/lomo.csv', index=False)

    return pd.DataFrame(all_results)


# ============================================================
# Protocol 3: Cross-dataset
# ============================================================
def run_cross_dataset(models, camp_data, ccrcc_data, graph_data,
                      camp_met_names=None, ccrcc_met_names=None,
                      latent_dim=30, n_steps=1000, n_seeds=3, device='cpu'):
    """Train on CAMP, test on ccRCC with proper gene and metabolite alignment."""
    camp_met, camp_rna, camp_batch = camp_data
    ccrcc_met, ccrcc_rna, ccrcc_batch = ccrcc_data

    # --- Align genes by name ---
    camp_genes = camp_batch.get('gene_names', None)
    ccrcc_genes = ccrcc_batch.get('gene_names', None)
    if camp_genes is not None and ccrcc_genes is not None:
        common_genes = sorted(set(camp_genes) & set(ccrcc_genes))
        camp_gene_idx = np.array([camp_genes.index(g) for g in common_genes])
        ccrcc_gene_idx = np.array([ccrcc_genes.index(g) for g in common_genes])
        camp_rna_aligned = camp_rna[:, camp_gene_idx]
        ccrcc_rna_aligned = ccrcc_rna[:, ccrcc_gene_idx]
        print(f"Gene alignment: CAMP {len(camp_genes)} ∩ ccRCC {len(ccrcc_genes)} "
              f"= {len(common_genes)} common genes")
    else:
        camp_rna_aligned = camp_rna
        ccrcc_rna_aligned = ccrcc_rna
        common_genes = None

    # --- Align metabolites by name (clean ccRCC asterisk suffixes) ---
    if camp_met_names is not None and ccrcc_met_names is not None:
        ccrcc_clean = [n.rstrip('*') for n in ccrcc_met_names]
        common_mets = sorted(set(camp_met_names) & set(ccrcc_clean))
        camp_met_idx = np.array([camp_met_names.index(m) for m in common_mets])
        ccrcc_met_idx = np.array([ccrcc_clean.index(m) for m in common_mets])
        print(f"Metabolite alignment: CAMP {len(camp_met_names)} ∩ ccRCC {len(ccrcc_met_names)} "
              f"= {len(common_mets)} common metabolites")
    else:
        camp_met_idx = None
        ccrcc_met_idx = None
        common_mets = None

    all_results = []

    for seed in range(n_seeds):
        for model_name in models:
            print(f"  [{model_name}] Seed {seed}, Cross-dataset", end="... ")
            t0 = time.time()
            try:
                # Train on CAMP with aligned genes
                if model_name in ['unigraph', 'unitedmet']:
                    preprocessed = preprocess_camp(camp_met, camp_rna_aligned, camp_batch)
                    if model_name == 'unigraph':
                        model = UniGraphModel(latent_dim=latent_dim, n_steps=n_steps, lr=0.001, device=device, seed=seed)
                        model.fit(preprocessed, graph_data, camp_batch, verbose=False)
                    else:
                        model = UnitedMetBaseline(latent_dim=latent_dim, n_steps=n_steps, lr=0.001, seed=seed, device=device)
                        model.fit(preprocessed, camp_batch, verbose=False)
                    preds = model.predict_met_ranks(rna_data=ccrcc_rna_aligned, batch_info=ccrcc_batch, n_samples=500, seed=42)
                    pred_ranks = preds['rank_hat_mean']
                elif model_name == 'gnn':
                    model = SimplifiedGNN(latent_dim=latent_dim, n_steps=n_steps, lr=0.001, device=device, seed=seed)
                    model.fit(camp_met, camp_rna_aligned, graph_data, camp_batch, verbose=False)
                    preds = model.predict_met_ranks(ccrcc_rna_aligned, ccrcc_batch)
                    pred_ranks = preds['rank_hat_mean']
                else:
                    if model_name == 'xgboost':
                        model = FastXGBoostBaseline(seed=seed, n_top_genes=100, n_estimators=100, max_depth=4)
                    elif model_name == 'lasso':
                        model = FastLassoBaseline(seed=seed, n_top_genes=200, alpha=0.01)
                    elif model_name == 'ridge':
                        model = FastRidgeBaseline(seed=seed, n_top_genes=500)
                    elif model_name == 'mirth':
                        model = MIRTHBaseline(latent_dim=latent_dim, seed=seed, device=device)
                    elif model_name == 'kernel_mkl':
                        model = KernelMKLBaseline(latent_dim=latent_dim, seed=seed, device=device)
                    model.fit(camp_met, camp_rna_aligned, camp_batch, verbose=False)
                    preds = model.predict_met_ranks(ccrcc_rna_aligned, ccrcc_batch)
                    pred_ranks = preds['rank_hat_mean']

                # True ranks on ccRCC
                true_ranks = get_true_ranks(ccrcc_met, ccrcc_batch)

                # Align metabolites by name
                if camp_met_idx is not None and ccrcc_met_idx is not None:
                    pred_aligned = pred_ranks[:, camp_met_idx]
                    true_aligned = true_ranks[:, ccrcc_met_idx]
                else:
                    n_overlap = min(true_ranks.shape[1], pred_ranks.shape[1])
                    pred_aligned = pred_ranks[:, :n_overlap]
                    true_aligned = true_ranks[:, :n_overlap]

                rhos, valid = spearman_per_metabolite(true_aligned, pred_aligned)
                valid_rhos = rhos[valid]
                result = {
                    'model': model_name, 'seed': seed,
                    'spearman_mean': np.mean(valid_rhos) if len(valid_rhos) > 0 else np.nan,
                    'spearman_median': np.median(valid_rhos) if len(valid_rhos) > 0 else np.nan,
                    'n_valid': int(valid.sum()),
                    'n_common_genes': len(common_genes) if common_genes is not None else camp_rna.shape[1],
                    'n_common_mets': len(common_mets) if common_mets is not None else n_overlap,
                    'runtime_sec': time.time() - t0,
                }
                all_results.append(result)
                print(f"ρ={result['spearman_mean']:.3f} ({result['runtime_sec']:.0f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    'model': model_name, 'seed': seed,
                    'error': str(e), 'runtime_sec': time.time() - t0
                })

            pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/crossds.csv', index=False)

    return pd.DataFrame(all_results)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='all',
                        choices=['all', 'indist', 'lomo', 'crossds'])
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model names or "all"')
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ALL_MODELS = ['unigraph', 'unitedmet', 'gnn', 'xgboost', 'lasso', 'ridge', 'mirth', 'kernel_mkl']
    FAST_MODELS = ['ridge', 'lasso', 'mirth', 'kernel_mkl']
    SLOW_MODELS = ['unigraph', 'unitedmet', 'gnn', 'xgboost']

    if args.models == 'all':
        models = ALL_MODELS
    elif args.models == 'fast':
        models = FAST_MODELS
    elif args.models == 'slow':
        models = SLOW_MODELS
    else:
        models = args.models.split(',')

    print(f"=== Benchmark: {args.protocol} ===")
    print(f"Models: {models}")
    print(f"Latent dim: {args.latent_dim}, Steps: {args.n_steps}")
    print(f"Folds: {args.n_folds}, Seeds: {args.n_seeds}")
    print()

    # Load data
    print("Loading CAMP data...")
    met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)

    # Load graph
    print("Loading graph...")
    gene_names = batch_info.get('gene_names', None)
    if gene_names is None:
        import pandas as pd
        rna_df = pd.read_csv('data/pancancer_metabolomics/data/transcriptomics_processed/Cornell_DLBCL.tpm.gene_symbol.csv', index_col=0)
        gene_names = list(rna_df.index)
    graph_data = construct_graph(met_anno, gene_names, rna_data)

    if args.protocol in ['all', 'indist']:
        print("\n=== Protocol 1: In-distribution CV ===")
        indist_results = run_indist_cv(
            models, met_data, rna_data, batch_info, graph_data,
            latent_dim=args.latent_dim, n_steps=args.n_steps,
            n_folds=args.n_folds, n_seeds=args.n_seeds, device=args.device
        )
        print("\nIn-distribution CV results:")
        cols = [c for c in ['spearman_mean', 'spearman_median', 'mae_mean'] if c in indist_results.columns]
        if cols:
            print(indist_results.groupby('model')[cols].mean().to_string())
        else:
            print(indist_results.to_string())

    if args.protocol in ['all', 'lomo']:
        print("\n=== Protocol 2: Zero-shot LOMO ===")
        lomo_results = run_lomo(
            models, met_data, rna_data, batch_info, graph_data,
            latent_dim=args.latent_dim, n_steps=args.n_steps,
            n_seeds=args.n_seeds, device=args.device
        )
        print("\nLOMO results:")
        cols = [c for c in ['spearman_mean', 'r2_mean'] if c in lomo_results.columns]
        if cols:
            print(lomo_results.groupby('model')[cols].mean().to_string())
        else:
            print(lomo_results.to_string())

    if args.protocol in ['all', 'crossds']:
        print("\n=== Protocol 3: Cross-dataset ===")
        print("Loading ccRCC data...")
        ccrcc_met, ccrcc_rna, ccrcc_met_names, ccrcc_gene_names, ccrcc_sample_info, ccrcc_batch_info = load_ccrcc_data()
        camp_data = (met_data, rna_data, batch_info)
        ccrcc_data = (ccrcc_met, ccrcc_rna, ccrcc_batch_info)
        crossds_results = run_cross_dataset(
            models, camp_data, ccrcc_data, graph_data,
            camp_met_names=met_names, ccrcc_met_names=ccrcc_met_names,
            latent_dim=args.latent_dim, n_steps=args.n_steps,
            n_seeds=args.n_seeds, device=args.device
        )
        print("\nCross-dataset results:")
        cols = [c for c in ['spearman_mean'] if c in crossds_results.columns]
        if cols:
            print(crossds_results.groupby('model')[cols].mean().to_string())
        else:
            print(crossds_results.to_string())

    print("\n=== Benchmark complete ===")
    print(f"Results saved to {RESULTS_DIR}/")
