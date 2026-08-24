#!/usr/bin/env python3
"""
Slow models benchmark: UnitedMet, UniGraph, GNN, XGBoost.
Uses gene subsampling (top N by variance) to make matrix factorization feasible.
Writes to separate files to avoid conflicts with fast models benchmark.
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
import traceback
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/UnitedMet')

from unigraph.data.load_camp import load_camp_data
from unigraph.data.graph import construct_graph
from unigraph.data.preprocess import (
    preprocess_camp, tic_normalization_across, log_transform_rna, normalize_rna,
    count_obs, order_and_rank, rank_predictions_per_batch
)
from unigraph.evaluation.metrics import spearman_per_metabolite, mae_per_metabolite, r2_per_metabolite
from unigraph.models.unigraph import UniGraphModel
from unigraph.models.baselines import UnitedMetBaseline, SimplifiedGNN
from unigraph.models.baselines_fast import FastXGBoostBaseline

RESULTS_DIR = "/mnt/results/benchmark"
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    """Select top N genes by variance."""
    gene_var = np.var(rna_data, axis=0)
    selected = np.argsort(gene_var)[-n_top_genes:]
    return selected


def run_slow_indist(models, met_data, rna_data, batch_info, graph_data,
                    latent_dim=30, n_steps=100, n_folds=2, n_seeds=1,
                    device='cpu', n_top_genes=5000):
    """In-distribution CV for slow models with gene subsampling."""
    N = met_data.shape[0]
    all_results = []
    outfile = f'{RESULTS_DIR}/indist_cv_slow.csv'

    for seed in range(n_seeds):
        np.random.seed(seed)
        indices = np.arange(N)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

            # Subsample genes based on training data variance
            gene_idx = subsample_genes(rna_data[train_idx], n_top_genes)
            train_rna_sub = rna_data[train_idx][:, gene_idx]
            test_rna_sub = rna_data[test_idx][:, gene_idx]
            train_met = met_data[train_idx]
            test_met = met_data[test_idx]
            train_batch = reindex_batches(batch_info, train_idx)
            test_batch = reindex_batches(batch_info, test_idx)

            for model_name in models:
                print(f"  [{model_name}] Seed {seed}, Fold {fold_idx+1}/{n_folds}", end="... ")
                t0 = time.time()
                try:
                    if model_name == 'unigraph':
                        preprocessed = preprocess_camp(train_met, train_rna_sub, train_batch)
                        model = UniGraphModel(latent_dim=latent_dim, n_steps=n_steps,
                                              lr=0.001, device=device, seed=seed*100+fold_idx)
                        model.fit(preprocessed, graph_data, train_batch, verbose=False)
                        preds = model.predict_met_ranks(rna_data=test_rna_sub, batch_info=test_batch,
                                                         n_samples=500, seed=42)
                    elif model_name == 'unitedmet':
                        preprocessed = preprocess_camp(train_met, train_rna_sub, train_batch)
                        model = UnitedMetBaseline(latent_dim=latent_dim, n_steps=n_steps,
                                                  lr=0.001, seed=seed*100+fold_idx, device=device)
                        model.fit(preprocessed, train_batch, verbose=False)
                        preds = model.predict_met_ranks(rna_data=test_rna_sub, batch_info=test_batch,
                                                         n_samples=500, seed=42)
                    elif model_name == 'gnn':
                        model = SimplifiedGNN(latent_dim=latent_dim, n_steps=n_steps,
                                              lr=0.001, device=device, seed=seed*100+fold_idx)
                        model.fit(train_met, train_rna_sub, graph_data, train_batch, verbose=False)
                        preds = model.predict_met_ranks(test_rna_sub, test_batch)
                    elif model_name == 'xgboost':
                        model = FastXGBoostBaseline(seed=seed*100+fold_idx, n_top_genes=100,
                                                     n_estimators=100, max_depth=4)
                        model.fit(train_met, train_rna_sub, train_batch, verbose=False)
                        preds = model.predict_met_ranks(test_rna_sub, test_batch)

                    pred_ranks = preds['rank_hat_mean']
                    true_ranks = get_true_ranks(test_met, test_batch)

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
                        'n_genes': len(gene_idx),
                    }
                    all_results.append(result)
                    print(f"ρ={result['spearman_mean']:.3f} ({result['runtime_sec']:.0f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    all_results.append({
                        'model': model_name, 'seed': seed, 'fold': fold_idx,
                        'error': str(e), 'runtime_sec': time.time() - t0
                    })

                pd.DataFrame(all_results).to_csv(outfile, index=False)

    return pd.DataFrame(all_results)


def run_slow_lomo(models, met_data, rna_data, batch_info, graph_data,
                  latent_dim=30, n_steps=100, n_seeds=1,
                  device='cpu', n_top_genes=5000):
    """LOMO zero-shot evaluation for slow models with gene subsampling."""
    all_results = []
    outfile = f'{RESULTS_DIR}/lomo_slow.csv'

    # Subsample genes based on full data variance
    gene_idx = subsample_genes(rna_data, n_top_genes)
    rna_sub = rna_data[:, gene_idx]

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
                is_zero_shot = model_name in ['unigraph', 'gnn']
                train_met = met_data
                train_rna = rna_sub
                train_batch = batch_info
                test_batch = batch_info

                if is_zero_shot:
                    train_met_masked = train_met.copy()
                    train_met_masked[:, held_out] = np.nan
                else:
                    train_met_masked = train_met

                if model_name == 'unigraph':
                    preprocessed = preprocess_camp(train_met_masked, train_rna, train_batch)
                    model = UniGraphModel(latent_dim=latent_dim, n_steps=n_steps,
                                          lr=0.001, device=device, seed=seed)
                    model.fit(preprocessed, graph_data, train_batch, verbose=False)
                    preds = model.predict_met_ranks(rna_data=train_rna, batch_info=test_batch,
                                                     n_samples=500, seed=42)
                elif model_name == 'unitedmet':
                    preprocessed = preprocess_camp(train_met_masked, train_rna, train_batch)
                    model = UnitedMetBaseline(latent_dim=latent_dim, n_steps=n_steps,
                                              lr=0.001, seed=seed, device=device)
                    model.fit(preprocessed, train_batch, verbose=False)
                    preds = model.predict_met_ranks(rna_data=train_rna, batch_info=test_batch,
                                                     n_samples=500, seed=42)
                elif model_name == 'gnn':
                    model = SimplifiedGNN(latent_dim=latent_dim, n_steps=n_steps,
                                          lr=0.001, device=device, seed=seed)
                    model.fit(train_met_masked, train_rna, graph_data, train_batch, verbose=False)
                    preds = model.predict_met_ranks(train_rna, test_batch)
                elif model_name == 'xgboost':
                    model = FastXGBoostBaseline(seed=seed, n_top_genes=100,
                                                 n_estimators=100, max_depth=4)
                    model.fit(train_met_masked, train_rna, train_batch, verbose=False)
                    preds = model.predict_met_ranks(train_rna, test_batch)

                pred_ranks = preds['rank_hat_mean']
                true_ranks = get_true_ranks(met_data, test_batch)

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
                    'n_genes': len(gene_idx),
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

            pd.DataFrame(all_results).to_csv(outfile, index=False)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='all',
                        choices=['all', 'indist', 'lomo'])
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model names or "all"')
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_folds', type=int, default=2)
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('--n_top_genes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ALL_SLOW = ['unigraph', 'unitedmet', 'gnn', 'xgboost']
    if args.models == 'all':
        models = ALL_SLOW
    else:
        models = args.models.split(',')

    print(f"=== Slow Models Benchmark: {args.protocol} ===")
    print(f"Models: {models}")
    print(f"Latent dim: {args.latent_dim}, Steps: {args.n_steps}")
    print(f"Folds: {args.n_folds}, Seeds: {args.n_seeds}")
    print(f"Top genes: {args.n_top_genes}")
    print()

    # Load data
    print("Loading CAMP data...")
    met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)

    print("Loading graph...")
    gene_names = batch_info.get('gene_names', None)
    if gene_names is None:
        rna_df = pd.read_csv('data/pancancer_metabolomics/data/transcriptomics_processed/Cornell_DLBCL.tpm.gene_symbol.csv', index_col=0)
        gene_names = list(rna_df.index)
    graph_data = construct_graph(met_anno, gene_names, rna_data)

    if args.protocol in ['all', 'indist']:
        print("\n=== In-distribution CV ===")
        results = run_slow_indist(
            models, met_data, rna_data, batch_info, graph_data,
            latent_dim=args.latent_dim, n_steps=args.n_steps,
            n_folds=args.n_folds, n_seeds=args.n_seeds,
            device=args.device, n_top_genes=args.n_top_genes
        )
        print("\nResults:")
        cols = [c for c in ['spearman_mean', 'spearman_median', 'mae_mean'] if c in results.columns]
        if cols:
            print(results.groupby('model')[cols].mean().to_string())

    if args.protocol in ['all', 'lomo']:
        print("\n=== LOMO ===")
        results = run_slow_lomo(
            models, met_data, rna_data, batch_info, graph_data,
            latent_dim=args.latent_dim, n_steps=args.n_steps,
            n_seeds=args.n_seeds,
            device=args.device, n_top_genes=args.n_top_genes
        )
        print("\nResults:")
        cols = [c for c in ['spearman_mean', 'r2_mean'] if c in results.columns]
        if cols:
            print(results.groupby('model')[cols].mean().to_string())

    print("\n=== Slow benchmark complete ===")
