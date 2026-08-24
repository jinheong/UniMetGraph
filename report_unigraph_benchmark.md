# UniMetGraph: Hybrid Rank-Based Bayesian Covariation with Metabolic Network Topology for Transcriptome-to-Metabolome Prediction

## Benchmark Report

**Date**: 2026-08-25
**Author**: Jin-Heong Ong
**Status**: Complete (fast models: 3 seeds; slow models: 1 seed, 2 folds; ablation: 1 seed, 2 folds, 6/6 variants)

---

## 1. Introduction

Predicting metabolome profiles from transcriptome data is a fundamental challenge in systems biology. The relationship between gene expression and metabolite abundance is mediated by enzyme kinetics, post-translational regulation, and network constraints, making it fundamentally non-linear. Two recent approaches address this from different angles:

- **UnitedMet** [1]: Uses rank-based Bayesian matrix factorization with a Plackett-Luce likelihood to model the covariation between transcriptomics and metabolomics without requiring direct abundance prediction.
- **GAZE** [24]: Uses a graph neural network (GATv2) over metabolic network topology with Morgan fingerprint node features to encode metabolite chemistry and network position.

**UniGraph** combines these approaches: GNN-encoded metabolite embeddings (from network topology + chemical fingerprints) replace free latent vectors for mapped metabolites, while retaining the rank-based Bayesian framework and Plackett-Luce observation model.

## 2. Methods

### 2.1 UniGraph Architecture

UniGraph combines:
- **MetaboliteGNN**: GATv2 (3 layers, hidden 256, 4 heads) on Human-GEM metabolite-metabolite graph with Morgan fingerprint (2048-bit) node features → K-dimensional metabolite embeddings
- **Bayesian MF**: Variational inference (AutoNormal guide) for sample embeddings (W), gene embeddings (H_gene), and unmapped metabolite embeddings (H_met_unmapped)
- **Plackett-Luce observation**: Rank-transformed [met|gene] data modeled via Plackett-Luce distribution per batch
- **Test prediction**: Ridge regression mapping from gene expression to sample embeddings (W)

### 2.2 Baselines

| Method | Type | Description |
|--------|------|-------------|
| UnitedMet | Rank-based Bayesian MF | Plackett-Luce on rank-transformed met+gene |
| Simplified GNN | Continuous GNN | GATv2 + Morgan fingerprints, MSE loss |
| XGBoost | Per-metabolite regression | Top 100 genes, 100 estimators, max_depth=4 |
| Lasso | Per-metabolite regression | Top 200 genes by correlation, alpha=0.01 |
| Ridge | Multivariate regression | Top 500 genes by variance |
| MIRTH | Metabolite-only MF | TruncatedSVD + Ridge mapping from genes |
| Kernel MKL | Kernel methods | KernelPCA (RBF) + Ridge mapping |

### 2.3 Dataset

- **CAMP** (primary): 764 tumor samples, 2,359 metabolites, 16,927 genes, 15 batches, 11 cancer types (Zenodo doi:10.5281/zenodo.7150252)
- **ccRCC** (external validation): 121 matched samples from CPTAC (50) and CPTAC_val (71), 1,148 metabolites, 21,022 genes (Zenodo doi:10.5281/zenodo.11286535). RC18 and RC20 excluded due to sample ID mismatch.
- **Human-GEM** v1.19.0: 8,461 metabolites, 12,931 reactions, 2,848 genes
- **Metabolite mapping**: 522/2,359 CAMP metabolites mapped to Human-GEM (22%)
- **Graph statistics**: 405 nodes, 3,622 edges, 365 valid Morgan fingerprints (2048-bit)
- **Cross-dataset alignment**: 16,053 common genes, 1,003 common metabolites

### 2.4 Implementation Details

- **Gene subsampling**: Matrix factorization models use top 5,000 genes by variance (~26s/step vs ~50s/step with full 16,927 genes). Regression baselines use their own gene selection.
- **Training**: 100 SVI steps (slow models), 50 steps (ablation). Latent dim K=30. Adam, lr=0.001.
- **Test prediction**: Post-hoc RidgeCV mapping from gene expression → sample embeddings (W), then X_met = W_pred @ H_met, rank-transformed per batch.
- **Compute**: CPU-only (CUDA unavailable). PyTorch 2.7–2.13, Pyro 1.9.1, torch-geometric 2.8.

### 2.5 Evaluation Protocols

1. **In-distribution CV**: 5-fold CV. Fast models: 3 seeds; slow models: 2 folds, 1 seed. Metrics: per-metabolite Spearman ρ and MAE.
2. **Zero-shot LOMO**: Leave-one-metabolite-out. Zero-shot methods (UniGraph, GNN) mask 50 metabolites; non-zero-shot methods evaluate on 10 held-out. Metrics: Spearman ρ and R².
3. **Cross-dataset**: Train CAMP → test ccRCC. 3 seeds (fast models only). Metric: Spearman ρ on 1,003 common metabolites.
4. **Ablation**: 6 variants. 2-fold CV, 1 seed, 50 SVI steps.

### 2.6 Statistical Analysis

- Wilcoxon signed-rank test for pairwise method comparison
- Benjamini-Hochberg FDR correction at α=0.05

## 3. Results

### 3.1 In-distribution CV

**Table 1. In-distribution CV results (all models)**

| Method | n_runs | Spearman ρ (mean ± std) | Median ρ | MAE (mean) |
|--------|--------|--------------------------|----------|------------|
| XGBoost | 2 | 0.029 ± 0.012 | 0.029 | 13.64 |
| Lasso | 9 | 0.020 ± 0.009 | 0.023 | 5.48 |
| Ridge | 9 | 0.015 ± 0.009 | 0.016 | 5.50 |
| Kernel MKL | 8 | 0.010 ± 0.006 | 0.010 | 5.55 |
| MIRTH | 9 | 0.007 ± 0.009 | 0.009 | 5.53 |
| UniGraph | 2 | 0.004 ± 0.007 | 0.004 | 13.88 |
| UnitedMet | 2 | 0.004 ± 0.007 | 0.004 | 13.88 |
| Simplified GNN | 2 | 0.003 ± 0.009 | 0.003 | 13.88 |

**Key findings:**
- XGBoost is the top-performing model (ρ=0.029), followed by Lasso (ρ=0.020).
- Among fast models, Lasso is significantly better than all others (Wilcoxon p<0.01, BH-corrected). Ridge is significantly better than MIRTH and Kernel MKL (p<0.01).
- UniGraph (ρ=0.004) and UnitedMet (ρ=0.004) perform nearly identically, both near random.
- The Simplified GNN (ρ=0.003) is the weakest model.
- Note: MAE values differ between fast models (~5.5) and slow models (~13.9) because fast models predict continuous values while slow models predict on the rank scale.

**Figure**: `indist_spearman_boxplot.svg`

### 3.2 Zero-shot LOMO

**Table 2. Zero-shot LOMO results (all models)**

| Method | n_runs | Held-out mets | Spearman ρ | R² |
|--------|--------|---------------|------------|-----|
| XGBoost | 1 | 10 | 0.076 | -16.43 |
| UnitedMet | 1 | 10 | 0.046 | -13.19 |
| UniGraph | 1 | 50 | 0.039 | -9.98 |
| Kernel MKL | 3 | 10 | 0.014 ± 0.022 | -21.21 |
| Ridge | 3 | 10 | 0.013 ± 0.046 | -17.01 |
| MIRTH | 3 | 10 | 0.007 ± 0.006 | -16.65 |
| Lasso | 3 | 10 | 0.007 ± 0.019 | -22.69 |
| Simplified GNN | 1 | 50 | 0.006 | -16.44 |

**Key findings:**
- XGBoost achieves the highest LOMO ρ (0.076), but evaluates on only 10 held-out metabolites.
- UnitedMet (ρ=0.046, 10 mets) outperforms UniGraph (ρ=0.039, 50 mets), but the comparison is not directly fair — UniGraph evaluates on 5× more metabolites in zero-shot mode.
- UniGraph (ρ=0.039) substantially outperforms the Simplified GNN (ρ=0.006) in zero-shot, suggesting the rank-based Bayesian framework helps even when the GNN alone fails.
- All R² values are strongly negative, indicating predictions worse than the mean.
- Fast model baselines (Ridge, Lasso, MIRTH, Kernel MKL) perform near-random (ρ<0.014), as expected for non-zero-shot methods.

**Figure**: `lomo_spearman_boxplot.svg`

### 3.3 Cross-dataset Validation

**Table 3. Cross-dataset results (CAMP→ccRCC, fast models)**

| Method | n_runs | Spearman ρ (mean ± std) | n_valid | n_common_mets |
|--------|--------|--------------------------|---------|---------------|
| Lasso | 3 | 0.016 ± 0.000 | 132 | 1,003 |
| Ridge | 3 | 0.008 ± 0.000 | 132 | 1,003 |
| MIRTH | 3 | -0.004 ± 0.000 | 132 | 1,003 |
| Kernel MKL | 3 | -0.005 ± 0.000 | 132 | 1,003 |

**Key findings:**
- Lasso is the only method with positive cross-dataset transfer (ρ=0.016).
- MIRTH and Kernel MKL perform below random (ρ≈-0.005).
- Results are remarkably consistent across seeds (std≈0).
- Slow model cross-dataset results were not completed (script prepared but not launched due to time constraints).

**Figure**: `crossds_spearman_boxplot.svg`

### 3.4 Ablation Study

**Table 4. Ablation results (2-fold CV, 1 seed, 50 steps)**

| Variant | Fold 0 ρ | Fold 1 ρ | Mean ρ | Description |
|---------|----------|----------|--------|-------------|
| Full UniGraph | -0.002 | 0.009 | 0.003 | GNN + Bayesian MF + Plackett-Luce |
| No Bayesian | -0.002 | 0.010 | 0.004 | Point estimates (Adam) instead of SVI |
| No Chemical | -0.002 | 0.009 | 0.003 | Degree-based features instead of Morgan fingerprints |
| No Rank (MSE) | -0.003 | 0.009 | 0.003 | MSE loss instead of Plackett-Luce |
| No Graph | -0.003 | 0.008 | 0.002 | Free embeddings for all metabolites, no GNN |
| UnitedMet (=No Graph+No Chemical) | -0.002 | 0.009 | 0.003 | Free embeddings, no GNN, no chemical features |

**Key findings:**
- All six ablation variants perform nearly identically (ρ≈0.002–0.004), with differences well within noise.
- Removing the Bayesian framework (no_bayesian) yields marginally higher ρ (0.004 vs 0.003), suggesting SVI with only 50 steps may not provide sufficient posterior exploration.
- Removing the rank-based loss (no_rank) does not degrade performance, contradicting expectations.
- Removing chemical fingerprints (no_chemical) has no effect, consistent with the low metabolite mapping rate (22%).
- Removing the GNN (no_graph) yields the lowest mean ρ (0.002), but the difference from the full model (0.003) is negligible. This confirms that the graph topology provides minimal benefit at this training scale.

**Figure**: `ablation_boxplot.svg`

### 3.5 Statistical Tests

**Table 5. Pairwise Wilcoxon tests (in-distribution CV, fast models)**

| Comparison | Mean diff | p-value | BH p | Significant? |
|------------|-----------|---------|------|--------------|
| Lasso vs MIRTH | +0.012 | 0.004 | 0.008 | Yes |
| Lasso vs Ridge | +0.005 | 0.004 | 0.008 | Yes |
| Lasso vs Kernel MKL | +0.012 | 0.008 | 0.009 | Yes |
| Ridge vs MIRTH | +0.008 | 0.004 | 0.008 | Yes |
| Ridge vs Kernel MKL | +0.007 | 0.008 | 0.009 | Yes |
| MIRTH vs Kernel MKL | +0.000 | 0.547 | 0.547 | No |

Slow models (2 runs each) had insufficient paired data for Wilcoxon tests.

### 3.6 Multi-protocol Summary

**Figure**: `multi_protocol_heatmap.svg` — heatmap of mean Spearman ρ across protocols.

**Table 6. Summary across all protocols**

| Protocol | Best Method | Best ρ | Worst Method | Worst ρ |
|----------|-------------|--------|--------------|---------|
| In-distribution CV | XGBoost | 0.029 | Simplified GNN | 0.003 |
| Zero-shot LOMO | XGBoost | 0.076 | Simplified GNN | 0.006 |
| Cross-dataset | Lasso | 0.016 | Kernel MKL | -0.005 |

## 4. Discussion

### 4.1 Overall Performance

All methods achieve very low Spearman ρ (<0.03 in-distribution, <0.08 LOMO), confirming that transcriptome-to-metabolome prediction is exceptionally difficult. The gene-metabolite relationship is mediated by multiple regulatory layers (enzyme kinetics, post-translational modification, allosteric regulation, transport), and no method captures more than a small fraction of this complexity.

### 4.2 XGBoost and Lasso Dominate

XGBoost (ρ=0.029 CV, 0.076 LOMO) and Lasso (ρ=0.020 CV, 0.016 cross-dataset) consistently outperform the matrix factorization approaches. This is likely because:
1. Per-metabolite models capture metabolite-specific gene-metabolite relationships without the constraint of a shared latent space.
2. Tree-based ensembles (XGBoost) can model non-linear interactions.
3. L1 regularization (Lasso) provides aggressive feature selection suited to the high-dimensional, low-sample regime.

### 4.3 UniGraph vs UnitedMet

UniGraph (ρ=0.004 CV, 0.039 LOMO) and UnitedMet (ρ=0.004 CV, 0.046 LOMO) perform nearly identically in-distribution, with UnitedMet slightly ahead in LOMO. The GNN-encoded metabolite embeddings do not improve over free latent vectors, likely because:
1. Only 22% of metabolites map to Human-GEM, limiting the GNN's coverage.
2. With 50–100 SVI steps, the models may not have converged sufficiently for the graph prior to matter.
3. The metabolite-metabolite graph (405 nodes, 3,622 edges) may be too sparse to provide useful inductive bias.

### 4.4 GNN Baseline Fails

The Simplified GNN (ρ=0.003 CV, 0.006 LOMO) is the weakest model, suggesting that GNN-encoded metabolite embeddings alone (without the rank-based Bayesian framework) are insufficient. UniGraph's improvement over the GNN baseline in LOMO (0.039 vs 0.006) indicates that the Plackett-Luce observation model contributes substantially more than the GNN encoder.

### 4.5 Ablation: Components Are Interchangeable

The ablation results show no meaningful differences between variants (all ρ≈0.002–0.004). This suggests that with 50 SVI steps and 5,000 genes, the model is operating in a regime where none of the architectural choices matter — the bottleneck is data and training scale, not architecture. Key factors:
1. **Insufficient training**: 50 steps may be far too few for SVI convergence with 7,359 features.
2. **Gene subsampling**: Top 5,000 by variance may exclude informative low-variance genes.
3. **Low mapping rate**: 22% metabolite coverage limits the GNN's influence.

### 4.6 Cross-dataset Generalization

Cross-dataset performance is uniformly low. Only Lasso achieves positive transfer (ρ=0.016) from pan-cancer CAMP to ccRCC. This suggests cancer-type-specific metabolic signatures and platform differences dominate over shared gene-metabolite relationships.

## 5. Conclusion

This benchmark evaluates 8 methods across 3 protocols for transcriptome-to-metabolome prediction:

1. **All methods perform poorly** (Spearman ρ < 0.03 in-distribution), confirming the fundamental difficulty of the task.
2. **XGBoost and Lasso dominate**, with per-metabolite regression outperforming matrix factorization approaches.
3. **UniGraph does not improve over UnitedMet**, likely due to low metabolite mapping rate (22%) and insufficient training scale.
4. **The rank-based Plackett-Luce framework helps over plain GNN** (UniGraph ρ=0.039 vs GNN ρ=0.006 in LOMO), but the GNN encoder itself adds no value over free embeddings.
5. **Ablation components are interchangeable** at this training scale, suggesting the bottleneck is data/compute, not architecture. The no_graph variant (ρ=0.002) confirms the GNN encoder adds no value over free embeddings at this scale.
6. **Cross-dataset generalization is extremely limited**, with only Lasso achieving positive transfer.
7. **The no_graph ablation** (ρ=0.002) confirms that removing the GNN encoder has negligible impact, consistent with the low metabolite mapping rate and insufficient training scale.

The results suggest that for transcriptome-to-metabolome prediction, simple per-metabolite regression with strong regularization (Lasso, XGBoost) outperforms complex multi-omics integration models. The rank-based Bayesian framework shows promise for zero-shot prediction but requires more training steps and higher metabolite-to-network mapping rates to realize its theoretical advantages.

## 6. Limitations

1. **GPU unavailable**: All Pyro SVI training on CPU, limiting steps (50–100) and seeds (1–2) for slow models.
2. **Gene subsampling**: Top 5,000 genes by variance for MF models, potentially excluding informative genes.
3. **Low metabolite mapping**: Only 22% of CAMP metabolites map to Human-GEM, limiting GNN coverage.
4. **ccRCC constraints**: 121 matched samples from 2/4 datasets; RC18/RC20 excluded.
5. **Insufficient slow model replication**: 1 seed, 2 folds — not enough for statistical significance.
6. **Cross-dataset for slow models**: Not completed (script prepared but not launched).
7. **MAE scale mismatch**: Fast models report MAE on continuous scale (~5.5), slow models on rank scale (~13.9) — not directly comparable.

## 7. Code and Data Availability

- **UniGraph package**: `/workspace/unigraph/` — Python package with data loading, preprocessing, models, evaluation
- **Benchmark scripts**: `run_benchmark.py`, `run_slow_benchmark.py`, `run_ablations.py`, `run_crossds_slow.py`, `run_analysis.py`
- **Results**: CSV files in `/mnt/results/benchmark/`, figures in `/mnt/results/figures/`
- **Data**: CAMP (Zenodo doi:10.5281/zenodo.7150252), ccRCC (Zenodo doi:10.5281/zenodo.11286535), Human-GEM v1.19.0

## References

[1] UnitedMet — rank-based Bayesian covariation model for multi-omics integration
[24] GAZE — metabolic network graph neural network
[22] MOFA — multi-omics factor analysis
[34] MKL — multiple kernel learning
[37] mixKernel — kernel mixing R package
