# UniMetGraph: Hybrid Rank-Based Bayesian Covariation with Metabolic Network Topology for Transcriptome-to-Metabolome Prediction

This package was developed using Biomni [1], an on-line agentic AI platform specialized for biomedical study. 
The idea of this tool was inspired by **UnitedMet** [2], which uses rank-based Bayesian matrix factorization with a Plackett-Luce likelihood to model the covariation between transcriptomics and metabolomics without requiring direct abundance prediction.
With the idea to integrate UnitedMet with graph network-based approach, Biomni suggested the plan and implement this package, as well as benchmark. 
This repository can serve as a showcase of how Biomni can help to develop a new computational tool for biomedical research. All generated files are provided here.

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

## Step 1: Run model predictions with cross-validation

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

Results are placed in the [report/](./report).

## References
[1] Huang K, Zhang S, Wang H, Qu Y, Lu Y, Roohani Y, Li R, Qiu L, Li G, Zhang J, Yin D, Marwaha S, Carter JN, Zhou X, Wheeler M, Bernstein JA, Wang M, He P, Zhou J, Snyder M, Cong L, Regev A, Leskovec J. Biomni: A General-Purpose Biomedical AI Agent. bioRxiv [Preprint]. 2025 Jun 2:2025.05.30.656746. doi: 10.1101/2025.05.30.656746. PMID: 40501924; PMCID: PMC12157518.

[2] Xie AX, Tansey W, Reznik E. UnitedMet harnesses RNA-metabolite covariation to impute metabolite levels in clinical samples. medRxiv [Preprint]. 2024 Nov 21:2024.05.24.24307903. doi: 10.1101/2024.05.24.24307903. Update in: Nat Cancer. 2025 May;6(5):892-906. doi: 10.1038/s43018-025-00943-0. PMID: 38826234; PMCID: PMC11142294.

## References of datasets
1. CAMP dataset: Zenodo `doi:10.5281/zenodo.7150252`
2. Human-GEM: Robinson et al., 2020. `doi:10.1016/j.ymben.2020.03.006`

