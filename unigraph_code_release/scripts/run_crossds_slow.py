#!/usr/bin/env python3
"""
Cross-dataset benchmark for slow models (UnitedMet, UniGraph, GNN, XGBoost)
with gene subsampling. Trains on CAMP, tests on ccRCC.
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
from unigraph.data.load_ccrcc import load_ccrcc_data
from unigraph.data.graph import construct_graph
from unigraph.data.preprocess import (
    preprocess_camp, normalize_rna, rank_predictions_per_batch
)
from unigraph.evaluation.metrics import spearman_per_metabolite
from unigraph.models.unigraph import UniGraphModel
from unigraph.models.baselines import (
    UnitedMetBaseline, SimplifiedGNN, MIRTHBaseline, KernelMKLBaseline
)
from unigraph.models.baselines_fast import FastXGBoostBaseline

RESULTS_DIR = "/mnt/results/benchmark"
os.makedirs(RESULTS_DIR, exist_ok=True)


def subsample_genes(rna_data, gene_names, n_top_genes=5000):
    """Subsample to top N genes by variance."""
    gene_var = np.var(rna_data, axis=0)
    selected = np.argsort(gene_var)[-n_top_genes:]
    selected_names = [gene_names[i] for i in selected]
    return rna_data[:, selected], selected_names


def get_true_ranks(met_data, batch_info):
    """Compute true per-batch ranks."""
    n_samples, n_mets = met_data.shape
    true_ranks = np.zeros_like(met_data, dtype=float)
    batch_ids = batch_info['batch_ids']
    unique_batches = batch_info['unique_batches']
    for b in unique_batches:
        mask = batch_ids == b
        if mask.sum() < 2:
            continue
        for j in range(n_mets):
            vals = met_data[mask, j]
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                true_ranks[mask, j] = np.nan
                continue
            ranks = pd.Series(vals[valid]).rank().values
            true_ranks[np.ix_(mask, [j])][valid] = ranks
    return true_ranks


def run_crossds_slow(models, n_steps=100, n_seeds=3, n_top_genes=5000,
                     latent_dim=30, device='cpu'):
    """Run cross-dataset benchmark for slow models."""
    print(f"=== Cross-dataset Slow Models ===")
    print(f"Models: {models}")
    print(f"Steps: {n_steps}, Seeds: {n_seeds}, Top genes: {n_top_genes}")

    # Load CAMP
    print("\nLoading CAMP data...")
    camp_met, camp_rna, camp_batch, camp_met_names = load_camp_data()
    print(f"  CAMP: {camp_met.shape[0]} samples, {camp_met.shape[1]} metabolites, {camp_rna.shape[1]} genes")

    # Subsample CAMP genes
    camp_gene_names = camp_batch['gene_names']
    camp_rna_sub, camp_gene_names_sub = subsample_genes(camp_rna, camp_gene_names, n_top_genes)
    camp_batch_sub = {**camp_batch, 'gene_names': camp_gene_names_sub}
    print(f"  CAMP subsampled: {camp_rna_sub.shape[1]} genes")

    # Load ccRCC
    print("\nLoading ccRCC data...")
    ccrcc_met, ccrcc_rna, ccrcc_batch, ccrcc_met_names = load_ccrcc_data()
    print(f"  ccRCC: {ccrcc_met.shape[0]} samples, {ccrcc_met.shape[1]} metabolites, {ccrcc_rna.shape[1]} genes")

    # Align genes
    ccrcc_gene_names = ccrcc_batch['gene_names']
    common_genes = sorted(set(camp_gene_names_sub) & set(ccrcc_gene_names))
    camp_gene_idx = np.array([camp_gene_names_sub.index(g) for g in common_genes])
    ccrcc_gene_idx = np.array([ccrcc_gene_names.index(g) for g in common_genes])
    camp_rna_aligned = camp_rna_sub[:, camp_gene_idx]
    ccrcc_rna_aligned = ccrcc_rna[:, ccrcc_gene_idx]
    print(f"  Gene alignment: {len(common_genes)} common genes")

    # Align metabolites
    ccrcc_clean = [n.rstrip('*') for n in ccrcc_met_names]
    common_mets = sorted(set(camp_met_names) & set(ccrcc_clean))
    camp_met_idx = np.array([camp_met_names.index(m) for m in common_mets])
    ccrcc_met_idx = np.array([ccrcc_clean.index(m) for m in common_mets])
    print(f"  Metabolite alignment: {len(common_mets)} common metabolites")

    # Load graph
    print("\nLoading graph...")
    graph_data = construct_graph()

    all_results = []

    for seed in range(n_seeds):
        for model_name in models:
            print(f"\n  [{model_name}] Seed {seed}, Cross-dataset", end="... ")
            t0 = time.time()
            try:
                # Train on CAMP with aligned genes
                if model_name in ['unigraph', 'unitedmet']:
                    preprocessed = preprocess_camp(camp_met, camp_rna_aligned, camp_batch_sub)
                    if model_name == 'unigraph':
                        model = UniGraphModel(latent_dim=latent_dim, n_steps=n_steps,
                                              lr=0.001, device=device, seed=seed)
                        model.fit(preprocessed, graph_data, camp_batch_sub, verbose=False)
                    else:
                        model = UnitedMetBaseline(latent_dim=latent_dim, n_steps=n_steps,
                                                  lr=0.001, seed=seed, device=device)
                        model.fit(preprocessed, camp_batch_sub, verbose=False)
                    preds = model.predict_met_ranks(rna_data=ccrcc_rna_aligned,
                                                    batch_info=ccrcc_batch, n_samples=500, seed=42)
                    pred_ranks = preds['rank_hat_mean']

                elif model_name == 'gnn':
                    model = SimplifiedGNN(latent_dim=latent_dim, n_steps=n_steps,
                                          lr=0.001, device=device, seed=seed)
                    model.fit(camp_met, camp_rna_aligned, graph_data, camp_batch_sub, verbose=False)
                    preds = model.predict_met_ranks(ccrcc_rna_aligned, ccrcc_batch)
                    pred_ranks = preds['rank_hat_mean']

                elif model_name == 'xgboost':
                    model = FastXGBoostBaseline(seed=seed, n_top_genes=100,
                                                n_estimators=100, max_depth=4)
                    model.fit(camp_met, camp_rna_aligned, camp_batch_sub, verbose=False)
                    preds = model.predict_met_ranks(ccrcc_rna_aligned, ccrcc_batch)
                    pred_ranks = preds['rank_hat_mean']

                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # True ranks on ccRCC
                true_ranks = get_true_ranks(ccrcc_met, ccrcc_batch)

                # Align metabolites
                pred_aligned = pred_ranks[:, camp_met_idx]
                true_aligned = true_ranks[:, ccrcc_met_idx]

                rhos, valid = spearman_per_metabolite(true_aligned, pred_aligned)
                valid_rhos = rhos[valid]
                result = {
                    'model': model_name, 'seed': seed,
                    'spearman_mean': np.mean(valid_rhos) if len(valid_rhos) > 0 else np.nan,
                    'spearman_median': np.median(valid_rhos) if len(valid_rhos) > 0 else np.nan,
                    'n_valid': int(valid.sum()),
                    'n_common_genes': len(common_genes),
                    'n_common_mets': len(common_mets),
                    'runtime_sec': time.time() - t0,
                }
                all_results.append(result)
                print(f"ρ={result['spearman_mean']:.3f} ({result['runtime_sec']:.0f}s)")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"ERROR: {e}")
                all_results.append({
                    'model': model_name, 'seed': seed,
                    'error': str(e), 'runtime_sec': time.time() - t0
                })

            # Save incrementally
            pd.DataFrame(all_results).to_csv(f'{RESULTS_DIR}/crossds_slow.csv', index=False)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated model names or "all"')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--n_top_genes', type=int, default=5000)
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.models == 'all':
        models = ['unitedmet', 'unigraph', 'gnn', 'xgboost']
    else:
        models = args.models.split(',')

    run_crossds_slow(models, n_steps=args.n_steps, n_seeds=args.n_seeds,
                     n_top_genes=args.n_top_genes, latent_dim=args.latent_dim,
                     device=args.device)
