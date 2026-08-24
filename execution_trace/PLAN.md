# UniGraph: A Hybrid Graph-Regularized Bayesian Rank Model for Metabolite Imputation

## 1. Overview

**Goal**: Design, implement, and benchmark a hybrid model ("UniGraph") that combines UnitedMet's rank-based Bayesian covariation (Plackett-Luce + stochastic variational inference) with GAZE's metabolic network topology (GNN + chemical structure embeddings). The model should excel at both in-distribution imputation (UnitedMet's strength) and zero-shot prediction of unseen metabolites (GAZE's strength), while providing Bayesian uncertainty quantification.

**Core hypothesis**: Incorporating metabolic network topology and chemical structure as inductive biases into UnitedMet's rank-based Bayesian framework will (a) improve in-distribution imputation accuracy by constraining the latent space with biological structure, and (b) enable zero-shot generalization to unseen metabolites via chemical embeddings — without sacrificing the rank-based robustness and uncertainty quantification that make UnitedMet suitable for clinical translation.

**Scope**: Comprehensive benchmark — 7 methods, 3 evaluation protocols, 3 random seeds, 6 ablation variants, statistical significance testing. Deliverable: reusable Python package + full report with figures.

---

## 2. Hybrid Model Architecture (UniGraph)

### 2.1 Design Philosophy

UniGraph replaces UnitedMet's free metabolite embedding matrix `H_met` with a **GNN-encoded embedding** that is informed by:
1. Metabolic network topology (Human-GEM bipartite reaction-metabolite graph)
2. Chemical structure (RDKit Morgan fingerprints from SMILES strings)
3. Enzyme gene expression (sample-specific TPM mapped to reaction nodes)

The sample embedding `W` remains **Bayesian** (variational posterior via Pyro SVI), and the observation model remains **Plackett-Luce** on rank-transformed data. This preserves UnitedMet's strengths (rank robustness, uncertainty quantification, handling of left-censored data) while adding GAZE's strengths (biological grounding, zero-shot generalization).

### 2.2 Architecture Details

**Input preprocessing** (identical to UnitedMet):
- RNA-seq: TPM normalization → rank-transform within each dataset
- Metabolomics: total ion count normalization → rank-transform within each dataset
- Left-censored metabolite values → tied at lowest rank
- Aggregate into matrix `R` (samples × features), features = genes ∪ metabolites

**Metabolic graph construction** (adapted from GAZE):
- Source: Human-GEM v1.19.0 from SysBioChalmers/Human-GEM GitHub
- Bipartite graph: reaction nodes + metabolite nodes, edges = stoichiometric relationships
- Node features:
  - Metabolite nodes: RDKit Morgan fingerprints (2048-bit) → PCA → 128-dim
  - Reaction nodes: one-hot EC class (7 classes) + sample-specific gene expression (TPM, mapped via gene-reaction associations from Human-GEM)
- Edge attributes: stoichiometric coefficients (negative for reactants, positive for products)
- Metabolites in CAMP not found in Human-GEM: added as isolated nodes with chemical features only (no graph edges)

**GNN encoder** (simplified from GAZE):
- 3 layers of Graph Attention Network v2 (GATv2) convolutions from PyTorch Geometric
- Bipartite message passing: metabolite ↔ reaction through stoichiometric edges
- Hidden dimension: 256; output dimension: λ (latent dimension, selected by grid search)
- No cross-modal attention, no Metabolite-Conditioned Reader, no physics-informed losses (these are GAZE-specific components omitted in the simplified version; the core graph + chemical + expression signal is preserved)
- Output: metabolite embeddings `H_met ∈ R^{M × λ}` (deterministic function of GNN parameters θ)
- Gene embeddings `H_gene ∈ R^{G × λ}`: free parameters (nn.Parameter), initialized randomly

**Bayesian latent factorization** (from UnitedMet):
- Sample embeddings: `W ~ Normal(0, 1)`, shape `(S × λ)`
- Latent matrix: `Z = W · H`, where `H = [H_gene; H_met]` (row-concatenated)
- Observation model: `R ~ Plackett-Luce(exp(Z))`
  - For each feature j, the likelihood of observing ranking `R_j` is:
    `P(R_j | Z) = Π_{i=1}^{K} exp(Z_{σ_i,j}) / Σ_{r=i}^{S} exp(Z_{σ_r,j})`
  - Left-censored values handled as tied rankings (same as UnitedMet)
- Modality balancing: weighted log-likelihood (metabolites weighted 1, genes weighted `n_met/n_gene`)

**Joint training**:
- GNN parameters θ and gene embeddings H_gene: point estimates (PyTorch nn.Parameter)
- Sample embeddings W: variational posterior via Pyro SVI with AutoNormal guide
- Loss: `-ELBO = -(log p(R | Z) - KL(q_φ(W) || Normal(0,1)))`
- Gradients flow through both: SVI handles φ (variational params for W), autograd handles θ and H_gene through the Plackett-Luce log-likelihood
- Optimizer: Adam, learning rate 0.001
- Convergence: ELBO relative change < 0.01 (same as UnitedMet)
- Latent dimension λ: grid search over [10, 20, 30, 50, 100, 150, 200], selected by 10-fold CV MAE

**Zero-shot prediction**:
- For an unseen metabolite m* (not in training data):
  1. Obtain SMILES string for m* (from HMDB/PubChem)
  2. Compute Morgan fingerprint → PCA → 128-dim chemical embedding
  3. Add m* to metabolic graph (connect to reactions it participates in, if known; otherwise isolated node)
  4. Run GNN message passing → `H_met*` (embedding for unseen metabolite)
  5. Compute `Z* = W · H_met*` → Plackett-Luce → predicted rankings
- No per-metabolite parameters needed — the GNN generates embeddings from chemical structure + network position

**Uncertainty quantification**:
- Draw 1000 samples from posterior of W (variational distribution)
- For each sample, compute Z = W · H → predicted rankings via Gumbel-Max trick
- Posterior mean = point estimate; posterior SD = uncertainty
- For zero-shot: same procedure, but H_met* is deterministic (GNN output); uncertainty comes from W only

**Posterior prediction** (same as UnitedMet):
- Gumbel-Max trick for efficient ranking sampling: `U_{i,j} = Z_{i,j} + G_{i,j}` where `G ~ Gumbel(0)`
- Sort perturbed log-probabilities → sampled rankings
- 1000 posterior draws → mean and SD

### 2.3 Key Implementation Files

```
unigraph/                          # Reusable Python package
├── __init__.py
├── data/
│   ├── preprocessing.py           # Rank transform, normalization, data loading
│   ├── metabolic_graph.py         # Load Human-GEM, build bipartite graph, map metabolites
│   └── chemical_embeddings.py     # SMILES → Morgan fingerprints → PCA
├── models/
│   ├── gnn_encoder.py             # GATv2-based GNN for metabolite embeddings (PyG)
│   ├── bayesian_factorization.py  # Bayesian W + Plackett-Luce (Pyro SVI)
│   ├── unigraph.py                # Full hybrid: GNN + Bayesian + PL, joint training
│   └── baselines/
│       ├── unitedmet_adapter.py   # Wrapper around UnitedMet's public code
│       ├── simplified_gnn.py      # GAZE-like: GNN + continuous MSE loss, no rank transform
│       ├── xgboost_baseline.py    # Per-metabolite XGBoost regression
│       ├── lasso_baseline.py      # Per-metabolite Lasso regression
│       └── mirth_adapter.py       # Wrapper around MIRTH's public code
├── evaluation/
│   ├── protocols.py               # 5-fold CV, LOMO, cross-dataset
│   ├── metrics.py                 # Spearman ρ, MAE, R², fraction positive
│   └── statistical_tests.py       # Wilcoxon signed-rank, bootstrap CI
└── benchmarks/
    ├── run_benchmark.py           # Main benchmark orchestrator
    └ run_ablations.py             # Ablation study runner

r_baselines/
├── mofa2_baseline.R               # MOFA2 imputation
└── mixkernel_baseline.R           # mixKernel + kernel ridge regression
```

---

## 3. Data Acquisition & Preprocessing

### 3.1 Primary Dataset: CAMP

- **Source**: Zenodo doi:10.5281/zenodo.7150252 (Cancer Atlas of Metabolic Profiles)
- **Content**: 988 paired metabolomics + transcriptomics samples across 11 cancer types, 15 studies
- **Format**: Preprocessed and harmonized; inspect on download for exact file structure
- **If CAMP contains cell line subset matching GAZE's setup (867 cell lines, 180 metabolites)**: use that subset for direct comparison with GAZE's reported numbers
- **If CAMP is tissue-only**: use full 988 samples; note that GAZE's cell-line numbers are not directly comparable

### 3.2 External Validation: ccRCC

- **Source**: Zenodo doi:10.5281/zenodo.11286535 (UnitedMet reference data)
- **Content**: 4 ccRCC datasets, 341 samples, 1,148 metabolites, 20,171 genes
- **Use**: Cross-dataset validation (train on CAMP, test on ccRCC)

### 3.3 Metabolic Network: Human-GEM

- **Source**: GitHub SysBioChalmers/Human-GEM (v1.19.0)
- **Format**: YAML or SBML, loaded via COBRApy
- **Content**: ~3,927 reactions, ~1,487 metabolites (in GAZE's graph), gene-reaction associations
- **Annotations**: metabolites.tsv with HMDB ID, KEGG ID, PubChem ID, ChEBI ID for metabolite mapping
- **Graph construction**: bipartite (reactions + metabolites), edges from stoichiometric matrix

### 3.4 Chemical Structures: SMILES

- **Source**: HMDB (via HMDB ID from Human-GEM annotations) or PubChem (via PubChem ID or name search)
- **Fallback**: RDKit name-to-SMILES via PubChem queries for metabolites not in Human-GEM
- **Embedding**: RDKit Morgan fingerprints (radius 2, 2048-bit) → PCA → 128-dim
- **Expected coverage**: >80% of CAMP metabolites should have SMILES (common metabolites)

### 3.5 Preprocessing Pipeline

1. Load CAMP RNA-seq (TPM) and metabolomics (ion counts)
2. Harmonize gene names across datasets (use HUGO gene symbols)
3. Harmonize metabolite names (map to HMDB IDs where possible)
4. Total ion count normalization for metabolomics
5. TPM normalization for RNA-seq (if not already TPM)
6. Rank-transform within each dataset (UnitedMet's formula: `rank_ij = Σ P[f_ij > f_kj] / S`)
7. Handle left-censored values as tied at lowest rank
8. Aggregate into matrix R (samples × features)
9. Map metabolites to Human-GEM via HMDB/KEGG/PubChem IDs
10. Get SMILES for mapped metabolites → Morgan fingerprints → PCA → 128-dim
11. Construct bipartite metabolic graph with node features
12. Map gene expression to reaction nodes via gene-reaction associations

---

## 4. Baseline Methods

### 4.1 UnitedMet (original)
- **Source**: GitHub reznik-lab/UnitedMet (clone and adapt)
- **Architecture**: Rank-based Bayesian matrix factorization (Z=WH) + Plackett-Luce + SVI
- **Adaptation**: Adapt input format to CAMP data; use their SVI training loop
- **Hyperparameters**: λ (latent dim) via grid search [10, 20, 30, 50, 100, 150, 200]; 4000 SVI steps; lr=0.001
- **Zero-shot**: Not supported by design (will show poor performance, as GAZE demonstrated)

### 4.2 Simplified GNN (GAZE-like)
- **Architecture**: 3-layer GATv2 on metabolic graph + Morgan fingerprint metabolite features + gene expression on reaction nodes + MLP readout
- **Key difference from UniGraph**: Continuous predictions (MSE loss), no rank transform, no Plackett-Luce, no Bayesian inference (point estimates)
- **Key difference from full GAZE**: No Metabolite-Conditioned Reader, no cross-modal attention, no physics-informed losses
- **Zero-shot**: Supported via chemical embeddings (metabolite features are not per-metabolite parameters)
- **Hyperparameters**: Hidden dim 256, output dim 128, Adam lr=0.001, 200 epochs

### 4.3 MOFA2
- **Source**: R package MOFA2 (Bioconductor)
- **Architecture**: Bayesian group-wise matrix factorization with Automatic Relevance Determination
- **Adaptation**: Train on paired RNA + metabolomics; use trained factors to impute metabolites for held-out samples
- **Hyperparameters**: Number of factors selected by MOFA2's automatic variance decomposition; default training settings
- **Zero-shot**: Not supported (requires metabolite to be observed during training)

### 4.4 XGBoost (per-metabolite)
- **Architecture**: One gradient boosting model per metabolite, using all genes as features
- **Hyperparameters**: LassoCV for feature selection (top 100 genes per metabolite), then XGBoost with early stopping
- **Zero-shot**: Not supported (per-metabolite models)

### 4.5 Lasso (per-metabolite)
- **Architecture**: One Lasso regression per metabolite, using all genes as features
- **Hyperparameters**: LassoCV with 5-fold CV for α selection; max 1000 iterations
- **Zero-shot**: Not supported

### 4.6 MIRTH
- **Source**: GitHub reznik-lab/MIRTH (clone and adapt)
- **Architecture**: Matrix factorization on metabolite-metabolite covariation (extended for cross-modality)
- **Adaptation**: Follow UnitedMet paper's cross-modality extension
- **Zero-shot**: Not supported

### 4.7 Kernel MKL (mixKernel)
- **Source**: R package mixKernel (Bioconductor)
- **Architecture**: Multiple kernel learning combining RBF kernels for RNA and metabolomics; kernel ridge regression for prediction
- **Adaptation**: Compute separate kernels for RNA and metabolomics modalities; combine via MKL; use kernel ridge regression to predict metabolites from RNA kernel
- **Hyperparameters**: RBF kernel bandwidth via median heuristic; MKL weights learned by optimization
- **Zero-shot**: Not supported

---

## 5. Evaluation Protocol

### 5.1 Protocol 1: In-Distribution (Sample Holdout)

- **Setup**: 5-fold stratified cross-validation on samples
- All metabolites seen during training; held-out samples have RNA-seq only
- For each fold: train on 80% of samples (paired RNA + metabolomics), predict metabolite rankings for held-out 20%
- **Seeds**: 3 random seeds (for variance estimation)
- **Metrics** (computed per metabolite, then aggregated):
  - Spearman ρ between predicted and true metabolite rankings (primary metric)
  - MAE on normalized ranks [0, 1)
  - Fraction of metabolites with positive Spearman ρ
- **Reporting**: Distribution of per-metabolite Spearman ρ (boxplot), mean ± std across seeds

### 5.2 Protocol 2: Zero-Shot (Metabolite Holdout / LOMO)

- **Setup**: Leave-one-metabolite-out across 50 randomly selected metabolites
- For zero-shot methods (UniGraph, simplified GNN): train once on all metabolites, predict held-out using chemical embeddings
- For non-zero-shot methods (UnitedMet, MOFA, XGBoost, Lasso, MIRTH, Kernel MKL): retrain per held-out metabolite
  - Note: This is outside UnitedMet/MOFA's design scope (as GAZE acknowledged). Results will show the limitation.
  - For efficiency: use 10 metabolites (randomly selected from the 50) for non-zero-shot methods, 3 seeds
- **Metrics**:
  - Spearman ρ (primary)
  - R² (where continuous predictions are available)
  - Fraction of metabolites with positive Spearman ρ / positive R²
- **Reporting**: Per-metabolite scatter plot (predicted vs true), boxplot of Spearman ρ

### 5.3 Protocol 3: Cross-Dataset External Validation

- **Setup**: Train on CAMP, test on ccRCC (4 datasets, 341 samples)
- All metabolites seen during training (intersection of CAMP and ccRCC metabolites)
- Tests generalization across platforms, cancer types, and batch effects
- **Seeds**: 3 random seeds
- **Metrics**: Spearman ρ per metabolite, aggregated
- **Reporting**: Boxplot comparison across methods

### 5.4 Fairness Guarantees

- All methods use identical train/test splits (same random seeds)
- All methods evaluated on the same set of metabolites (intersection across methods)
- Hyperparameter tuning performed within cross-validation folds (no test leakage)
- For rank-based methods (UnitedMet, UniGraph): evaluate on rank scale
- For continuous methods (simplified GNN, XGBoost, MOFA, Lasso, Kernel MKL): evaluate on both rank scale (Spearman ρ) and continuous scale (R²)
- 5-minute timeout per method per fold (methods that don't converge are flagged)

---

## 6. Ablation Studies

Six ablation variants of UniGraph, each removing one component to isolate its contribution:

| Variant | GNN encoder | Rank transform | Bayesian W | Chemical embeddings | Purpose |
|---|---|---|---|---|---|
| **UniGraph (full)** | Yes | Yes | Yes | Yes | Full model |
| **No graph** | No (free H_met) | Yes | Yes | No | Tests graph topology contribution → should degrade to UnitedMet |
| **No rank** | Yes | No (continuous) | Yes | Yes | Tests rank transform contribution |
| **No Bayesian** | Yes | Yes | No (point est.) | Yes | Tests uncertainty quantification contribution |
| **No chemical** | Yes | Yes | Yes | No (random) | Tests chemical structure contribution |
| **No graph + no chemical** | No | Yes | Yes | No | = UnitedMet (sanity check) |

All ablations run on Protocol 1 (in-distribution) and Protocol 2 (zero-shot) with 3 seeds.

---

## 7. Statistical Analysis

- **Primary comparison**: Wilcoxon signed-rank test on per-metabolite Spearman ρ (paired by metabolite), UniGraph vs each baseline
- **Multiple testing**: Benjamini-Hochberg FDR correction across method comparisons
- **Effect size**: Report median difference in Spearman ρ and 95% bootstrap CI
- **Variance across seeds**: Report mean ± std across 3 seeds
- **Uncertainty calibration** (Bayesian methods only): Coverage of 80% and 95% credible intervals (fraction of true values falling within predicted interval)
- **Per-metabolite analysis**: Identify metabolites where UniGraph significantly outperforms or underperforms UnitedMet; check if these correlate with graph connectivity (metabolites with more graph neighbors → bigger improvement?)

---

## 8. Compute & Resource Estimate

### 8.1 Data Size
- CAMP: ~988 samples × (~18,000 genes + ~180 metabolites) → rank matrix ~143 MB
- ccRCC: ~341 samples × (~20,000 genes + ~1,148 metabolites) → ~60 MB
- Human-GEM: YAML/SBML ~10-50 MB; graph ~5K nodes, ~16K edges → negligible
- Morgan fingerprints: ~180 metabolites × 2048 bits → negligible
- Total data: <1 GB

### 8.2 Memory
- Rank matrix: ~143 MB (fits in 32 GB easily)
- GNN: ~5K nodes × 256-dim → ~5 MB
- Variational parameters: ~988 × λ (λ ~50) → negligible
- Per-method training: <4 GB peak
- Total: well within 32 GB per machine

### 8.3 Runtime Per Method

| Method | Per-fold training | 5-fold × 3 seeds | LOMO (50 met) | Total |
|---|---|---|---|---|
| UniGraph | ~15-20 min | ~5-10 hrs | ~20 min (train once) | ~6-11 hrs |
| UnitedMet | ~10-15 min | ~3-7 hrs | ~5 hrs (retrain × 10) | ~8-12 hrs |
| Simplified GNN | ~5-10 min | ~2-4 hrs | ~10 min (train once) | ~2-4 hrs |
| MOFA2 | ~3-5 min | ~1-2 hrs | ~1 hr (retrain × 10) | ~2-3 hrs |
| XGBoost | ~1-2 min | ~0.5-1 hr | ~10 min (retrain × 50) | ~1 hr |
| Lasso | ~30 sec | ~15 min | ~5 min (retrain × 50) | ~20 min |
| MIRTH | ~5-10 min | ~2-4 hrs | ~3 hrs (retrain × 10) | ~5-7 hrs |
| Kernel MKL | ~3-5 min | ~1-2 hrs | ~1 hr (retrain × 10) | ~2-3 hrs |

### 8.4 Parallelization Strategy

- **2 machines** (as allowed by plan):
  - Machine 1 (worker-0): UniGraph + UnitedMet + MIRTH + ablations
  - Machine 2 (worker-1): Simplified GNN + MOFA2 + XGBoost + Lasso + Kernel MKL
- **Within-machine parallelism**: 5 CV folds run in parallel (8 CPUs available), using Python multiprocessing or joblib
- **Estimated wall time**: ~6-10 hours with 2 machines and fold-level parallelism

### 8.5 Execution Target

- **No HPC needed**: All workloads are CPU-bound, fit in 32 GB RAM, and complete within hours
- **Machine 1 (worker-0)**: Default machine, upgraded to 8 CPU / 32 GB
- **Machine 2 (worker-1)**: Created via ManageMachine, 8 CPU / 32 GB
- **Checkpointing**: Save intermediate results to `/mnt/shared-workspace/` after each method completes, so partial results survive machine failures

---

## 9. Implementation Timeline (Chunked)

### Chunk 1: Environment Setup & Data Acquisition (~1-2 hrs)
- Install: torch-geometric, xgboost, cobra, MOFA2 (R), mixKernel (R)
- Clone: UnitedMet, MIRTH, Human-GEM repos
- Download: CAMP (Zenodo), ccRCC data (Zenodo)
- Inspect CAMP data structure, verify paired RNA + metabolomics

### Chunk 2: Data Preprocessing & Graph Construction (~1-2 hrs)
- Rank-transform CAMP and ccRCC data
- Load Human-GEM via COBRApy, extract stoichiometric matrix
- Map CAMP metabolites to Human-GEM metabolites (via HMDB/KEGG IDs)
- Get SMILES for metabolites → Morgan fingerprints → PCA
- Construct bipartite metabolic graph with node features
- Map gene expression to reaction nodes
- Save preprocessed data to `/mnt/shared-workspace/`

### Chunk 3: Implement UniGraph Hybrid Model (~2-3 hrs)
- Implement GNN encoder (gnn_encoder.py) using PyTorch Geometric
- Implement Bayesian factorization with Plackett-Luce (bayesian_factorization.py) using Pyro
- Implement joint training loop (unigraph.py)
- Implement zero-shot prediction
- Test on small subset (10 samples, 20 metabolites) to verify convergence

### Chunk 4: Implement Baselines (~2-3 hrs)
- UnitedMet adapter (adapt their code for CAMP data format)
- Simplified GNN (GAZE-like, continuous predictions)
- XGBoost and Lasso per-metabolite baselines
- MIRTH adapter
- MOFA2 R script
- mixKernel R script
- Test each baseline on small subset

### Chunk 5: Run In-Distribution Benchmark (~2-4 hrs, parallelized)
- 5-fold CV × 3 seeds × 7 methods
- Machine 1: UniGraph + UnitedMet + MIRTH
- Machine 2: Simplified GNN + MOFA2 + XGBoost + Lasso + Kernel MKL
- Save results to `/mnt/shared-workspace/`

### Chunk 6: Run Zero-Shot Benchmark (~1-2 hrs)
- LOMO × 50 metabolites (zero-shot methods) + 10 metabolites (non-zero-shot)
- 3 seeds
- Save results

### Chunk 7: Run Cross-Dataset Validation (~1 hr)
- Train on CAMP, test on ccRCC
- 3 seeds × 7 methods

### Chunk 8: Run Ablation Studies (~2-3 hrs)
- 6 ablation variants × Protocol 1 + Protocol 2 × 3 seeds
- Run on Machine 1

### Chunk 9: Analysis, Figures & Report (~2-3 hrs)
- Statistical tests (Wilcoxon, FDR correction)
- Generate figures (SVG + PNG):
  - Boxplot: per-metabolite Spearman ρ across methods (Protocol 1)
  - Boxplot: per-metabolite Spearman ρ across methods (Protocol 2, zero-shot)
  - Heatmap: method × metabolite Spearman ρ (top 30 metabolites)
  - Ablation bar plot: mean Spearman ρ by ablation variant
  - Uncertainty calibration plot: coverage vs nominal level
  - Cross-dataset validation boxplot
  - Per-metabolite improvement plot: UniGraph vs UnitedMet, colored by graph connectivity
- Write full report to `/mnt/results/report_unigraph_benchmark.md`

**Total estimated: ~14-22 hrs** (may span 2 sessions if exceeding sandbox limits; checkpoint after each chunk)

---

## 10. Deliverables

### 10.1 Reusable Python Package
- `unigraph/` package with all model code, baselines, evaluation, and benchmark scripts
- Saved to `/mnt/results/unigraph_package/`
- Can be imported and used for new datasets

### 10.2 Full Report
- `/mnt/results/report_unigraph_benchmark.md`
- Sections: Abstract, Introduction, Methods (UniGraph architecture, baselines, evaluation), Results (in-distribution, zero-shot, cross-dataset, ablations), Discussion, References
- Includes embedded figure references and result tables

### 10.3 Figures (SVG + PNG)
- `/mnt/results/figures/`
  - `fig1_indistribution_boxplot.svg` — Spearman ρ across methods (Protocol 1)
  - `fig2_zeroshot_boxplot.svg` — Spearman ρ across methods (Protocol 2)
  - `fig3_crossdataset_boxplot.svg` — Cross-dataset validation
  - `fig4_ablation_barplot.svg` — Ablation study results
  - `fig5_uncertainty_calibration.svg` — Coverage of credible intervals
  - `fig6_permetabolite_heatmap.svg` — Method × metabolite heatmap
  - `fig7_graph_connectivity.svg` — Improvement vs graph connectivity

### 10.4 Result Tables
- `/mnt/results/tables/`
  - `table1_indistribution_results.csv` — Mean ± std Spearman ρ, MAE, fraction positive
  - `table2_zeroshot_results.csv` — Per-metabolite Spearman ρ, R²
  - `table3_crossdataset_results.csv` — Cross-dataset Spearman ρ
  - `table4_ablation_results.csv` — Ablation variant × metric
  - `table5_statistical_tests.csv` — Wilcoxon p-values, FDR-adjusted

### 10.5 Execution Trace
- Notebook with all code, outputs, and visualizations for reproducibility

---

## 11. Assumptions & Risk Mitigation

### Assumptions
1. CAMP data on Zenodo contains paired RNA-seq (TPM) + metabolomics (ion counts) in accessible format (CSV/TSV)
2. >80% of CAMP metabolites can be mapped to HMDB/PubChem IDs for SMILES retrieval
3. >50% of CAMP metabolites can be mapped to Human-GEM metabolites for graph connectivity
4. PyTorch Geometric can be installed via pip in the sandbox environment
5. MOFA2 and mixKernel can be installed via Bioconductor in R
6. UnitedMet's code runs with Pyro 1.9.1 (already installed)
7. 3 random seeds provide sufficient variance estimation (5 would be better but 3 is feasible within time budget)

### Risk Mitigation
- **CAMP format unknown**: Inspect immediately on download; if format is unexpected, adapt preprocessing
- **Low metabolite-SMILES mapping rate**: Fall back to zero vectors for unmapped metabolites; report mapping rate; this becomes an ablation ("no chemical embeddings")
- **Low metabolite-Human-GEM mapping rate**: Unmapped metabolites become isolated graph nodes with chemical features only; report mapping rate
- **PyTorch Geometric installation fails**: Fall back to manual GCN implementation using PyTorch + NetworkX (sparse matrix operations)
- **MOFA2/mixKernel installation fails**: Skip these baselines and note in report; focus on Python-based methods
- **UnitedMet code incompatibility**: Reimplement UnitedMet's core (Plackett-Luce + SVI) from the paper, using Pyro directly
- **Joint training instability**: Fall back to two-stage training (train GNN first with self-supervised objective, then train Bayesian W with fixed H_met)
- **Compute exceeds time budget**: Reduce to 2 seeds, reduce LOMO to 30 metabolites, reduce ablations to 4 variants
- **Sandbox 24h limit**: Checkpoint after each chunk to `/mnt/shared-workspace/`; resume in fresh sandbox if needed

### Explicit Methodological Choices (Agent Decides)
- **Chemical embeddings**: RDKit Morgan fingerprints (2048-bit → PCA → 128-dim) instead of ChemBERTa. Rationale: RDKit is already installed, Morgan fingerprints are well-established, and ChemBERTa requires downloading a 77M parameter model from HuggingFace which adds complexity without clear benefit for this benchmark.
- **Enzyme features**: One-hot EC class + gene expression instead of EC2Vec. Rationale: EC2Vec requires a separate download; one-hot EC class captures the categorical enzyme function, and gene expression provides the sample-specific signal.
- **GNN architecture**: GATv2 (3 layers, hidden 256) instead of full GAZE architecture. Rationale: User selected "simplified GNN"; GATv2 captures the essential graph attention mechanism without the full cross-modal attention and MCR complexity.
- **Latent dimension search**: [10, 20, 30, 50, 100, 150, 200] instead of UnitedMet's [1, 351] step 10. Rationale: Reduced range for computational efficiency; covers the relevant range based on UnitedMet's findings (optimal ~30 for ccRCC).
- **3 seeds** instead of 5: Rationale: Feasibility within time budget; 3 seeds still provide meaningful variance estimates.
- **LOMO**: 50 metabolites for zero-shot methods, 10 for non-zero-shot methods. Rationale: Non-zero-shot methods require retraining per metabolite (expensive); 10 is sufficient to demonstrate the limitation. Zero-shot methods train once and predict all 50.
- **Statistical test**: Wilcoxon signed-rank test (non-parametric, paired). Rationale: Per-metabolite Spearman ρ values are not normally distributed; Wilcoxon is the standard choice for paired non-parametric comparisons.
- **Multiple testing**: Benjamini-Hochberg FDR at α=0.05. Rationale: Standard for method comparison benchmarks; controls false discovery rate while maintaining power.
