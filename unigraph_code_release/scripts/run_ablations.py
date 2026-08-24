#!/usr/bin/env python3
"""
Ablation study: Run UniGraph variants with components removed.
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/UnitedMet')

from unigraph.data.load_camp import load_camp_data
from unigraph.data.graph import construct_graph
from unigraph.data.preprocess import (
    preprocess_camp, tic_normalization_across, log_transform_rna, normalize_rna,
    count_obs, order_and_rank
)
from unigraph.evaluation.metrics import spearman_per_metabolite, mae_per_metabolite
from unigraph.models.ablations import UniGraphAblation
from unigraph.models.baselines import UnitedMetBaseline

RESULTS_DIR = "/mnt/results/benchmark"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ablation variants
ABLATIONS = {
    'full': dict(use_graph=True, use_rank=True, use_bayesian=True, use_chemical=True),
    'no_graph': dict(use_graph=False, use_rank=True, use_bayesian=True, use_chemical=True),
    'no_rank': dict(use_graph=True, use_rank=False, use_bayesian=True, use_chemical=True),
    'no_bayesian': dict(use_graph=True, use_rank=True, use_bayesian=False, use_chemical=True),
    'no_chemical': dict(use_graph=True, use_rank=True, use_bayesian=True, use_chemical=False),
    # no_graph + no_chemical = UnitedMet (handled separately)
}


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


def run_ablation_indist(ablation_name, ablation_config, met_data, rna_data,
                         batch_info, graph_data, latent_dim=30, n_steps=500,
                         n_folds=3, n_seeds=2, device='cpu', n_top_genes=5000):
    """Run in-distribution CV for one ablation variant."""
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

            print(f"  [{ablation_name}] Seed {seed}, Fold {fold_idx+1}/{n_folds}", end="... ")
            t0 = time.time()
            try:
                # Subsample genes based on training data variance
                gene_idx = subsample_genes(rna_data[train_idx], n_top_genes)
                train_met = met_data[train_idx]
                train_rna = rna_data[train_idx][:, gene_idx]
                test_met = met_data[test_idx]
                test_rna = rna_data[test_idx][:, gene_idx]
                train_batch = reindex_batches(batch_info, train_idx)
                test_batch = reindex_batches(batch_info, test_idx)

                preprocessed = preprocess_camp(train_met, train_rna, train_batch)

                if ablation_name == 'unitedmet':
                    model = UnitedMetBaseline(latent_dim=latent_dim, n_steps=n_steps,
                                              lr=0.001, seed=seed*100+fold_idx, device=device)
                    model.fit(preprocessed, train_batch, verbose=False)
                else:
                    model = UniGraphAblation(
                        latent_dim=latent_dim, n_steps=n_steps, lr=0.001,
                        device=device, seed=seed*100+fold_idx, **ablation_config)
                    model.fit(preprocessed, graph_data, train_batch, verbose=False)

                preds = model.predict_met_ranks(rna_data=test_rna, batch_info=test_batch,
                                                 n_samples=500, seed=42)
                pred_ranks = preds['rank_hat_mean']
                true_ranks = get_true_ranks(test_met, test_batch)

                rhos, valid = spearman_per_metabolite(true_ranks, pred_ranks)
                maes, _ = mae_per_metabolite(true_ranks, pred_ranks)
                valid_rhos = rhos[valid]
                result = {
                    'model': ablation_name, 'seed': seed, 'fold': fold_idx,
                    'spearman_mean': np.mean(valid_rhos),
                    'spearman_median': np.median(valid_rhos),
                    'spearman_std': np.std(valid_rhos),
                    'n_valid_mets': int(valid.sum()),
                    'mae_mean': np.mean(maes[valid]),
                    'runtime_sec': time.time() - t0,
                }
                all_results.append(result)
                print(f"ρ={result['spearman_mean']:.3f} ({result['runtime_sec']:.0f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    'model': ablation_name, 'seed': seed, 'fold': fold_idx,
                    'error': str(e), 'runtime_sec': time.time() - t0
                })

            pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/ablation_indist.csv', index=False)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--n_seeds', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_top_genes', type=int, default=5000)
    parser.add_argument('--variants', type=str, default='all',
                        help='Comma-separated variant names or "all"')
    args = parser.parse_args()

    print("=== Ablation Study ===")
    print(f"Steps: {args.n_steps}, Folds: {args.n_folds}, Seeds: {args.n_seeds}, Top genes: {args.n_top_genes}")

    # Load data
    print("Loading CAMP data...")
    met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)

    print("Loading graph...")
    gene_names = batch_info.get('gene_names', None)
    if gene_names is None:
        rna_df = pd.read_csv('data/pancancer_metabolomics/data/transcriptomics_processed/Cornell_DLBCL.tpm.gene_symbol.csv', index_col=0)
        gene_names = list(rna_df.index)
    graph_data = construct_graph(met_anno, gene_names, rna_data)

    # Determine variants to run
    if args.variants == 'all':
        variants = list(ABLATIONS.keys()) + ['unitedmet']
    else:
        variants = args.variants.split(',')

    all_results = []
    for vname in variants:
        print(f"\n--- {vname} ---")
        if vname == 'unitedmet':
            df = run_ablation_indist('unitedmet', {}, met_data, rna_data,
                                      batch_info, graph_data,
                                      latent_dim=args.latent_dim, n_steps=args.n_steps,
                                      n_folds=args.n_folds, n_seeds=args.n_seeds,
                                      device=args.device, n_top_genes=args.n_top_genes)
        else:
            df = run_ablation_indist(vname, ABLATIONS[vname], met_data, rna_data,
                                      batch_info, graph_data,
                                      latent_dim=args.latent_dim, n_steps=args.n_steps,
                                      n_folds=args.n_folds, n_seeds=args.n_seeds,
                                      device=args.device, n_top_genes=args.n_top_genes)
        all_results.append(df)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(f'{RESULTS_DIR}/ablation_indist.csv', index=False)
    print("\n=== Ablation results ===")
    cols = [c for c in ['spearman_mean', 'spearman_median', 'mae_mean'] if c in final.columns]
    if cols:
        print(final.groupby('model')[cols].mean().to_string())
    print(f"\nSaved to {RESULTS_DIR}/ablation_indist.csv")
