# UniGraph: Hybrid Rank-Based Bayesian Covariation with Metabolic Network Topology

Code for benchmarking UniGraph against UnitedMet, Simplified GNN, XGBoost, Lasso, Ridge, MIRTH, and Kernel MKL for transcriptome-to-metabolome prediction.

## Structure

```
unigraph_code_release/
├── README.md                          ← this file
├── unigraph/                          ← Python package (4,822 LOC total)
│   ├── __init__.py
│   ├── data/
│   │   ├── load_camp.py               (201 lines) — CAMP pan-cancer metabolomics loader
│   │   ├── load_ccrcc.py              (119 lines) — ccRCC external validation loader
│   │   ├── preprocess.py              (204 lines) — TIC normalization, rank transform, RNA normalization
│   │   └── graph.py                   (338 lines) — Human-GEM metabolite graph construction + Morgan fingerprints
│   ├── models/
│   │   ├── unigraph.py                (385 lines) — UniGraph hybrid model (GNN + Bayesian MF + Plackett-Luce)
│   │   ├── baselines.py               (711 lines) — UnitedMet, SimplifiedGNN, MIRTH, KernelMKL baselines
│   │   ├── baselines_fast.py          (281 lines) — FastRidge, FastLasso, FastXGBoost baselines
│   │   └── ablations.py               (438 lines) — UniGraphAblation with configurable component removal
│   ├── evaluation/
│   │   ├── metrics.py                 (114 lines) — Spearman ρ, MAE, R² per metabolite
│   │   └── benchmark.py               (367 lines) — Benchmark framework (standalone scripts preferred)
│   └── utils/
│       └── __init__.py
└── scripts/                           ← Standalone benchmark & analysis scripts
    ├── run_benchmark.py               (478 lines) — Main benchmark: indist CV, LOMO, cross-dataset (fast models)
    ├── run_slow_benchmark.py          (323 lines) — Slow models benchmark with gene subsampling
    ├── run_ablations.py               (209 lines) — Ablation study runner (6 variants)
    ├── run_crossds_slow.py            (200 lines) — Cross-dataset for slow models with gene alignment
    ├── run_analysis.py                (381 lines) — Statistical analysis + figure generation
    └── test_fast_models.py            (73 lines)  — Quick timing test for fast baselines
```

## Requirements

```
Python >= 3.11
torch >= 2.7
torch-geometric >= 2.8
pyro-ppl >= 1.9
numpy, pandas, scipy, scikit-learn
rdkit >= 2025.3
cobra       (for Human-GEM model loading)
xgboost >= 3.0
matplotlib  (for figures)
```

Install with:
```bash
uv pip install torch torch-geometric pyro-ppl numpy pandas scipy scikit-learn rdkit cobra xgboost matplotlib
```

## Data Requirements

1. **CAMP** pan-cancer metabolomics: Download from Zenodo (doi:10.5281/zenodo.7150252)
   - Place in: `data/pancancer_metabolomics/`
   - Key file: `MasterMapping_MetImmune_03_16_2022_release.csv`

2. **ccRCC** clear cell renal cell carcinoma: Download from Zenodo (doi:10.5281/zenodo.11286535)
   - Place in: `data/ccrcc/`
   - Only CPTAC and CPTAC_val datasets have matched metabolomics+transcriptomics samples

3. **Human-GEM** v1.19.0: Clone from GitHub
   ```bash
   git clone https://github.com/SysBioChalmers/Human-GEM.git
   ```
   - Model files in: `Human-GEM/model/`

4. **UnitedMet** (for imports): Clone the repository
   ```bash
   git clone https://github.com/compbiolab/UnitedMet.git
   ```

## Usage

### 1. Fast model benchmark (Ridge, Lasso, MIRTH, Kernel MKL)
```bash
# In-distribution CV (3 seeds, 5 folds)
python scripts/run_benchmark.py --protocol indist --models fast --n_seeds 3 --n_folds 5 --device cpu

# Zero-shot LOMO (3 seeds)
python scripts/run_benchmark.py --protocol lomo --models fast --n_seeds 3 --device cpu

# Cross-dataset (3 seeds)
python scripts/run_benchmark.py --protocol crossds --models fast --n_seeds 3 --device cpu
```

### 2. Slow model benchmark (UnitedMet, UniGraph, GNN, XGBoost)
```bash
# In-distribution CV + LOMO (1 seed, 2 folds, 100 SVI steps, top 5000 genes)
python scripts/run_slow_benchmark.py --protocol all --models all --n_steps 100 --n_folds 2 --n_seeds 1 --n_top_genes 5000 --device cpu
```

### 3. Cross-dataset for slow models
```bash
python scripts/run_crossds_slow.py --models all --n_steps 100 --n_seeds 3 --n_top_genes 5000 --device cpu
```

### 4. Ablation study
```bash
python scripts/run_ablations.py --n_steps 50 --n_folds 2 --n_seeds 1 --n_top_genes 5000 --device cpu
```

### 5. Statistical analysis and figures
```bash
python scripts/run_analysis.py
```

## Key Design Decisions

- **Gene subsampling**: Matrix factorization models (UnitedMet, UniGraph, GNN) use top 5,000 genes by variance to make Plackett-Luce SVI feasible on CPU (~26s/step vs ~50s/step with full 16,927 genes).
- **Test prediction**: MF models learn sample embeddings (W) during training. For test samples, a post-hoc RidgeCV maps gene expression → W, then X_met = W_pred @ H_met, rank-transformed per batch.
- **Per-batch ranking**: Ranks are computed within each batch (dataset) to handle batch effects. Uses boolean mask indexing to handle non-contiguous samples after CV splitting.
- **LOMO protocol**: Zero-shot methods (UniGraph, GNN) mask 50 metabolites during training; non-zero-shot methods train on all metabolites and are evaluated on 10 held-out.
- **Cross-dataset alignment**: Genes aligned by name (16,053 common); metabolites aligned by name with asterisk suffix cleaning for ccRCC (1,003 common).

## Output Files

Results are written to `benchmark/`:
- `indist_cv.csv` — In-distribution CV (fast models)
- `indist_cv_slow.csv` — In-distribution CV (slow models)
- `lomo.csv` — Zero-shot LOMO (fast models)
- `lomo_slow.csv` — Zero-shot LOMO (slow models)
- `crossds.csv` — Cross-dataset (fast models)
- `crossds_slow.csv` — Cross-dataset (slow models)
- `ablation_indist.csv` — Ablation study
- `summary_table.csv` — Merged summary
- `wilcoxon_indist.csv` — Pairwise statistical tests

Figures are written to `figures/` (SVG + PNG):
- `indist_spearman_boxplot` — Box plot of Spearman ρ by method (in-distribution CV)
- `lomo_spearman_boxplot` — Box plot of Spearman ρ by method (LOMO)
- `crossds_spearman_boxplot` — Box plot of Spearman ρ by method (cross-dataset)
- `ablation_boxplot` — Box plot of Spearman ρ by ablation variant
- `multi_protocol_heatmap` — Heatmap of mean ρ across protocols and methods

## Citation

If you use this code, please cite:
- UnitedMet (rank-based Bayesian covariation)
- GAZE (metabolic network GNN)
- CAMP dataset (Zenodo doi:10.5281/zenodo.7150252)
- ccRCC dataset (Zenodo doi:10.5281/zenodo.11286535)
- Human-GEM v1.19.0 (SysBioChalmers)
