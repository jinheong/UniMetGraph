# Confidence-Filtered Metabolite Prediction Analysis

## Overview

This module analyzes whether a subset of metabolites can be predicted more confidently than others from transcriptome data, and whether filtering to high-confidence predictions improves aggregate prediction performance measured by Spearman's rank correlation.

The analysis evaluates 8 models for transcriptome-to-metabolome prediction across 2,359 metabolites in 764 pan-cancer tumor samples (CAMP dataset). While aggregate performance is low (Spearman ρ ≈ 0.01–0.02 across all metabolites), per-metabolite analysis reveals substantial heterogeneity: a subset of metabolites achieves ρ > 0.20 when identified using confidence measures.

**Key result**: Filtering to the top 10% of metabolites by `max_gene_corr` (max gene-metabolite correlation, computed on training data only) lifts aggregate ρ from ~0.01 to ~0.10–0.11 across all 8 models — a 6–14× improvement that is scientifically rigorous (non-circular).

## Requirements

### Python packages

```
torch>=2.0
torch-geometric>=2.3
pyro-ppl>=1.8
xgboost>=1.7
cobra>=0.25
rdkit>=2022.03
numpy>=1.23
pandas>=1.5
scipy>=1.9
scikit-learn>=1.2
matplotlib>=3.6
```

Install with:

```bash
uv pip install torch pyro-ppl xgboost cobra rdkit numpy pandas scipy scikit-learn matplotlib
```

For torch-geometric, follow the official installation guide at https://pytorch-geometric.readthedocs.io/

### Data

The analysis uses three data sources:

1. **CAMP (Pan-cancer metabolomics)** — 764 tumor samples, 2,359 metabolites, 16,927 genes, 15 datasets covering 11 cancer types. Available from Zenodo: `doi:10.5281/zenodo.7150252`
2. **Human-GEM v1.19.0** — Genome-scale metabolic model used for metabolite network graph construction. 8,461 metabolites, 12,931 reactions, 2,848 genes. Available from: https://github.com/SysBioChalmers/Human-GEM
3. **Graph cache** — Precomputed metabolite graph (405 nodes, 3,622 edges) with Morgan fingerprints (365 valid). Built from Human-GEM and cached as `graph_data.npz` + `smiles_map.json`.

Place data under `data/pancancer_metabolomics/` and `Human-GEM/model/` respectively. The graph cache goes under `data/graph_cache/`.

### Dependencies

The UnitedMet baseline requires the UnitedMet package. Clone and place under `UnitedMet/`:

```bash
git clone https://github.com/UnitedMet/UnitedMet.git UnitedMet
```

The UniGraph and baseline models are in the `unigraph/` package (included in this repository).

## Usage

### Step 1: Run model predictions with cross-validation

Run fast models (Lasso, Ridge, XGBoost, MIRTH, Kernel MKL) with 3-fold CV:

```bash
python run_confidence_analysis.py --models lasso,ridge,xgboost,mirth,kernel_mkl --n_folds 3 --seed 42
```

Run slow models (UnitedMet, UniGraph, Simplified GNN) with 2-fold CV:

```bash
python run_confidence_analysis.py --models unitedmet,unigraph,gnn --n_folds 2 --n_steps 50 --n_top_genes 5000 --seed 42
```

Run all models at once:

```bash
python run_confidence_analysis.py --models all --n_folds 3 --seed 42
```

Predictions and per-metabolite confidence measures are saved as `.npz` files in `confidence_predictions/`.

### Step 2: Run analysis and generate figures

After all model predictions are saved:

```bash
python run_confidence_analysis.py --analyze
```

This generates:
- 6 figures (SVG + PNG) in `results/confidence/figures/`
- 6 CSV tables in `results/confidence/tables/`

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | `all` | `fast`, `slow`, `all`, or comma-separated model names |
| `--n_folds` | `3` | Number of cross-validation folds |
| `--n_steps` | `50` | SVI steps for Bayesian models (UnitedMet, UniGraph) |
| `--n_top_genes` | `5000` | Gene subsampling count for slow models |
| `--seed` | `42` | Random seed |
| `--analyze` | `False` | Run analysis only (skip model training) |

### Running on multiple machines

Fast and slow models can be run in parallel on separate machines. After both complete, copy all `.npz` prediction files into a single `confidence_predictions/` directory, then run `--analyze`.

## Models

| Model | Type | Description |
|-------|------|-------------|
| Lasso | Per-metabolite regression | Lasso with top-200 correlated genes per metabolite |
| Ridge | Multivariate regression | Ridge regression with top-500 variance genes |
| XGBoost | Per-metabolite gradient boosting | XGBoost with top-50 genes, 50 estimators, univariate pre-filtering |
| MIRTH | SVD + Ridge | Truncated SVD on metabolomics, Ridge mapping from genes |
| Kernel MKL | Kernel PCA + Ridge | RBF Kernel PCA on metabolomics, Ridge mapping from genes |
| UnitedMet | Bayesian matrix factorization | SVI with Plackett-Luce rank likelihood, K=30 |
| UniGraph | GNN + Bayesian MF | GATv2 metabolite encoder + Bayesian MF + Plackett-Luce |
| Simplified GNN | GNN + point estimate | GATv2 encoder with Morgan fingerprints, continuous MSE loss |

## Confidence Measures

### Train-set measures (non-circular, computed on training data only)

| Measure | Description | Model-Independent? |
|---------|-------------|-------------------|
| `max_gene_corr` | Max \|Pearson r\| between metabolite and any gene (top 5,000 by variance) | Yes |
| `obs_rate` | Fraction of training samples with non-NaN values | Yes |
| `coef_var` | Coefficient of variation (std/\|mean\|) on training data | Yes |
| `network_mapped` | Binary: metabolite maps to Human-GEM network | Yes |
| `train_r2` | Training R² per metabolite (model-specific) | No |
| `posterior_var` | Posterior predictive variance (Bayesian models only) | No |

### Post-hoc measure (circular)

| Measure | Description |
|---------|-------------|
| `cv_rho` | Per-metabolite Spearman ρ on pooled CV test predictions |

## Filtering Strategy

For each confidence measure and model:
1. Rank metabolites from highest to lowest confidence
2. Evaluate at 7 thresholds: top 5%, 10%, 25%, 50%, 75%, 90%, 100%
3. Compute aggregate Spearman ρ (mean of per-metabolite ρ across retained metabolites)

**Post-hoc filtering** uses `cv_rho` (test-set performance) to select metabolites — this shows the theoretical upper bound but is circular (uses test performance to select metabolites).

**Non-circular filtering** uses train-set confidence measures only, applied to held-out test data — this shows practically achievable improvement without information leakage.

## Results Summary

### Aggregate performance (all metabolites)

| Model | ρ (mean) |
|-------|----------|
| Lasso | 0.019 |
| XGBoost | 0.017 |
| Ridge | 0.015 |
| Kernel MKL | 0.011 |
| MIRTH | 0.011 |
| Simplified GNN | 0.008 |
| UnitedMet | 0.007 |
| UniGraph | 0.007 |

### Non-circular filtering (top 10% by max_gene_corr)

| Model | ρ @ 10% | Improvement |
|-------|---------|-------------|
| Lasso | 0.114 | 6.0× |
| Ridge | 0.108 | 7.0× |
| XGBoost | 0.108 | 6.3× |
| UniGraph | 0.103 | 15.7× |
| UnitedMet | 0.102 | 14.4× |
| Simplified GNN | 0.102 | 12.9× |
| MIRTH | 0.099 | 9.0× |
| Kernel MKL | 0.099 | 9.0× |

### Confidence measure correlations with test ρ

| Measure | r | p-value |
|---------|---|---------|
| **max_gene_corr** | **0.373** | < 1e-300 |
| obs_rate | 0.156 | 6.6e-103 |
| coef_var | -0.153 | 9.7e-99 |
| posterior_var | -0.099 | 1.3e-6 |
| train_r2 | 0.093 | 8.0e-38 |

### Top 5 predictable metabolites

| Metabolite | Pathway | ρ (Lasso) |
|-----------|---------|-----------|
| glucose | Carbohydrate / Glycolysis | 0.264 |
| guanosine 5'-diphospho-fucose | Nucleotide / Purine | 0.249 |
| maltose | Carbohydrate / Starch metabolism | 0.247 |
| fructose-6-phosphate | Carbohydrate / Glycolysis | 0.238 |
| 1-arachidoylglycerophosphocholine | Lipid / Lysolipid | 0.238 |

## Outputs

### Figures (`results/confidence/figures/`)

| File | Description |
|------|-------------|
| `per_met_rho_distribution.svg/png` | Violin plot of per-metabolite ρ distribution per model |
| `performance_vs_fraction_posthoc.svg/png` | Aggregate ρ vs. fraction retained (post-hoc, upper bound) |
| `performance_vs_fraction_noncircular.svg/png` | Aggregate ρ vs. fraction retained (non-circular, best confidence measure) |
| `confidence_measure_comparison.svg/png` | Correlation of each confidence measure with per-metabolite ρ |
| `top_predictable_metabolites.svg/png` | Heatmap of top-30 metabolites across all models |
| `pathway_analysis.svg/png` | Mean ρ by super pathway for best model |

### Tables (`results/confidence/tables/`)

| File | Description |
|------|-------------|
| `per_metabolite_rho.csv` | All 2,359 metabolites with ρ per model + confidence measures + annotations |
| `filtered_performance_posthoc.csv` | Aggregate ρ at each filtering threshold (post-hoc) |
| `filtered_performance_noncircular.csv` | Aggregate ρ at each filtering threshold (non-circular, all measures) |
| `confidence_measure_correlations.csv` | Correlation between each confidence measure and per-metabolite ρ |
| `top_predictable_metabolites.csv` | Top-50 metabolites with ρ, confidence, and pathway annotations |
| `pathway_analysis.csv` | Mean ρ by super pathway for each model |

## Methodology

### Cross-validation protocol

- **Fast models**: 3-fold CV, 1 seed. Each metabolite gets predictions from all 764 samples (pooled across folds).
- **Slow models**: 2-fold CV, 1 seed, 50 SVI steps, top 5,000 genes by variance.
- Per-metabolite Spearman ρ computed on pooled predictions vs true ranks (minimum 5 observations).

### Confidence measure computation

Train-set confidence measures are computed on training data only within each CV fold, then averaged across folds. This ensures no information leakage from test data.

### Filtering evaluation

- **Post-hoc**: Pool predictions → compute per-metabolite ρ → rank by ρ → filter → report aggregate ρ on filtered set.
- **Non-circular**: For each fold, compute confidence on training data → rank metabolites → filter test predictions → compute aggregate ρ → average across folds.

## Code Structure

```
run_confidence_analysis.py    # Main script (model running + analysis)
unigraph/
  data/
    load_camp.py              # CAMP data loading
    preprocess.py             # TIC normalization, rank transform, RNA normalization
    graph.py                  # Human-GEM graph construction
  models/
    unigraph.py               # UniGraph hybrid model (GNN + Bayesian MF)
    baselines.py              # UnitedMet, SimplifiedGNN, MIRTH, KernelMKL
    baselines_fast.py         # FastLasso, FastXGBoost, FastRidge
  evaluation/
    metrics.py                # Per-metabolite Spearman, MAE, R²
```

## References

1. CAMP dataset: Zenodo `doi:10.5281/zenodo.7150252`
2. Human-GEM: Robinson et al., 2020. `doi:10.1016/j.ymben.2020.03.006`
3. UnitedMet: Rank-based Bayesian covariation model for metabolome prediction
4. GAZE: Metabolic network topology-based GNN for metabolite prediction
