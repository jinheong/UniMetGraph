#!/usr/bin/env python3
"""
Statistical analysis and figure generation for UniGraph benchmark.
Merges fast and slow model results, runs Wilcoxon tests, generates figures.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = "/mnt/results/benchmark"
FIGURES_DIR = "/mnt/results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Model display names and order
MODEL_NAMES = {
    'unigraph': 'UniGraph',
    'unitedmet': 'UnitedMet',
    'gnn': 'Simplified GNN',
    'xgboost': 'XGBoost',
    'lasso': 'Lasso',
    'ridge': 'Ridge',
    'mirth': 'MIRTH',
    'kernel_mkl': 'Kernel MKL',
}

# Preferred display order
MODEL_ORDER = ['unigraph', 'unitedmet', 'gnn', 'xgboost', 'lasso', 'ridge', 'mirth', 'kernel_mkl']

# Ablation display names
ABLATION_NAMES = {
    'full': 'Full UniGraph',
    'no_graph': 'No Graph',
    'no_rank': 'No Rank (MSE)',
    'no_bayesian': 'No Bayesian',
    'no_chemical': 'No Chemical',
    'unitedmet': 'No Graph+No Chemical (=UnitedMet)',
}

ABLATION_ORDER = ['full', 'no_graph', 'no_rank', 'no_bayesian', 'no_chemical', 'unitedmet']


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    result = np.empty(n)
    result[order] = bh
    return result


def pairwise_wilcoxon(df, metric_col='spearman_mean', model_col='model',
                      seed_col='seed', fold_col='fold'):
    """Pairwise Wilcoxon signed-rank test between all models."""
    models = sorted(df[model_col].unique())
    pivot = df.pivot_table(index=[seed_col, fold_col], columns=model_col,
                           values=metric_col, aggfunc='first')

    results = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            common = pivot[[m1, m2]].dropna()
            if len(common) < 5:
                continue
            try:
                stat, pval = wilcoxon(common[m1], common[m2])
                results.append({
                    'model1': m1, 'model2': m2,
                    'mean_diff': common[m1].mean() - common[m2].mean(),
                    'wilcoxon_stat': stat, 'pvalue': pval,
                    'n_pairs': len(common),
                })
            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    res_df['pvalue_bh'] = benjamini_hochberg(res_df['pvalue'].values)
    res_df['significant'] = res_df['pvalue_bh'] < 0.05
    return res_df


def load_and_merge(base_name, slow_suffix='_slow'):
    """Load fast and slow result files and merge them."""
    fast_path = f'{RESULTS_DIR}/{base_name}.csv'
    slow_path = f'{RESULTS_DIR}/{base_name}{slow_suffix}.csv'

    dfs = []
    if os.path.exists(fast_path):
        df = pd.read_csv(fast_path)
        df['speed'] = 'fast'
        dfs.append(df)
        print(f"  Loaded {base_name}.csv: {len(df)} rows")
    if os.path.exists(slow_path):
        df = pd.read_csv(slow_path)
        df['speed'] = 'slow'
        dfs.append(df)
        print(f"  Loaded {base_name}{slow_suffix}.csv: {len(df)} rows")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def filter_valid(df):
    """Filter out rows with NaN spearman_mean or n_valid=0."""
    if 'spearman_mean' not in df.columns:
        return df
    df = df.dropna(subset=['spearman_mean'])
    if 'n_valid' in df.columns:
        df = df[df['n_valid'] > 0]
    return df


def create_boxplot(df, metric_col, title, filename, figsize=(10, 6), order=None):
    """Create box plot of metric across models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    matplotlib.rcParams['svg.fonttype'] = 'none'

    if order is None:
        medians = df.groupby('model')[metric_col].median().sort_values(ascending=False)
        order = medians.index.tolist()

    order = [m for m in order if m in df['model'].unique()]
    extra = [m for m in df['model'].unique() if m not in order]
    order = order + sorted(extra)

    fig, ax = plt.subplots(figsize=figsize)
    data = [df[df['model'] == m][metric_col].dropna().values for m in order]
    labels = [MODEL_NAMES.get(m, m) for m in order]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                               'markeredgecolor': 'black', 'markersize': 5})

    colors = ['#0279EE', '#FF9400', '#75A025', '#FD9BED', '#E9ED4C',
              '#000000', '#ECE9E2', '#FAF9F3']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Spearman ρ' if 'spearman' in metric_col else metric_col)
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    fig.savefig(f'{FIGURES_DIR}/{filename}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/{filename}.png', format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}.svg, {filename}.png")


def create_ablation_boxplot(df, metric_col='spearman_mean', filename='ablation_boxplot'):
    """Create box plot for ablation study."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    matplotlib.rcParams['svg.fonttype'] = 'none'

    order = [m for m in ABLATION_ORDER if m in df['model'].unique()]
    extra = [m for m in df['model'].unique() if m not in order]
    order = order + sorted(extra)

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df[df['model'] == m][metric_col].dropna().values for m in order]
    labels = [ABLATION_NAMES.get(m, m) for m in order]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                               'markeredgecolor': 'black', 'markersize': 5})

    colors = ['#0279EE', '#FF9400', '#75A025', '#FD9BED', '#E9ED4C', '#ECE9E2']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Spearman ρ')
    ax.set_title('Ablation Study: Component Contributions')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    fig.savefig(f'{FIGURES_DIR}/{filename}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/{filename}.png', format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}.svg, {filename}.png")


def create_multi_protocol_heatmap(all_results, filename='multi_protocol_heatmap'):
    """Create heatmap showing all methods across all protocols."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
    matplotlib.rcParams['svg.fonttype'] = 'none'

    protocols = ['In-distribution CV', 'Zero-shot LOMO', 'Cross-dataset']
    protocol_keys = ['indist', 'lomo', 'crossds']

    models = [m for m in MODEL_ORDER if any(m in all_results.get(k, pd.DataFrame())['model'].unique()
                                            for k in protocol_keys if not all_results.get(k, pd.DataFrame()).empty)]

    matrix = np.full((len(models), len(protocols)), np.nan)
    for j, key in enumerate(protocol_keys):
        df = all_results.get(key, pd.DataFrame())
        if df.empty:
            continue
        for i, m in enumerate(models):
            vals = df[df['model'] == m]['spearman_mean'].dropna()
            if len(vals) > 0:
                matrix[i, j] = vals.mean()

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.6)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in models])
    ax.set_xticks(range(len(protocols)))
    ax.set_xticklabels(protocols, rotation=20, ha='right')

    for i in range(len(models)):
        for j in range(len(protocols)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = 'white' if val > np.nanmean(matrix) else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label='Spearman ρ')
    plt.tight_layout()

    fig.savefig(f'{FIGURES_DIR}/{filename}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{FIGURES_DIR}/{filename}.png', format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}.svg, {filename}.png")


def create_summary_table(all_results):
    """Create a summary table of all results."""
    rows = []
    for protocol_name, key in [('In-distribution CV', 'indist'),
                                ('Zero-shot LOMO', 'lomo'),
                                ('Cross-dataset', 'crossds')]:
        df = all_results.get(key, pd.DataFrame())
        if df.empty:
            continue
        for model in MODEL_ORDER:
            vals = df[df['model'] == model]
            if len(vals) == 0:
                continue
            spearman = vals['spearman_mean'].dropna()
            if len(spearman) == 0:
                continue
            row = {
                'Protocol': protocol_name,
                'Method': MODEL_NAMES.get(model, model),
                'n_runs': len(spearman),
                'Spearman_mean': spearman.mean(),
                'Spearman_std': spearman.std() if len(spearman) > 1 else 0,
                'Spearman_median': spearman.median(),
            }
            if 'mae_mean' in vals.columns:
                mae = vals['mae_mean'].dropna()
                if len(mae) > 0:
                    row['MAE_mean'] = mae.mean()
            if 'r2_mean' in vals.columns:
                r2 = vals['r2_mean'].dropna()
                if len(r2) > 0:
                    row['R2_mean'] = r2.mean()
            rows.append(row)

    return pd.DataFrame(rows)


def run_analysis():
    """Run full statistical analysis and generate figures."""
    print("=== Statistical Analysis ===\n")

    all_results = {}

    # 1. In-distribution CV (merge fast + slow)
    print("--- In-distribution CV ---")
    indist = load_and_merge('indist_cv')
    indist = filter_valid(indist)
    all_results['indist'] = indist
    if len(indist) > 0:
        print(f"  Total: {len(indist)} valid results")
        summary = indist.groupby('model')['spearman_mean'].agg(['mean', 'std', 'median', 'count'])
        print(summary.to_string())

        print("\n  Pairwise Wilcoxon signed-rank tests:")
        wilcox = pairwise_wilcoxon(indist)
        if len(wilcox) > 0:
            print(wilcox.to_string(index=False))
            wilcox.to_csv(f'{RESULTS_DIR}/wilcoxon_indist.csv', index=False)
        else:
            print("  Not enough paired data for Wilcoxon tests")

        create_boxplot(indist, 'spearman_mean',
                       'In-distribution CV: Spearman ρ by Method',
                       'indist_spearman_boxplot', order=MODEL_ORDER)
    print()

    # 2. LOMO (merge fast + slow)
    print("--- Zero-shot LOMO ---")
    lomo = load_and_merge('lomo')
    lomo = filter_valid(lomo)
    all_results['lomo'] = lomo
    if len(lomo) > 0:
        print(f"  Total: {len(lomo)} valid results")
        summary = lomo.groupby('model')['spearman_mean'].agg(['mean', 'std', 'median', 'count'])
        print(summary.to_string())
        create_boxplot(lomo, 'spearman_mean',
                       'Zero-shot LOMO: Spearman ρ by Method',
                       'lomo_spearman_boxplot', order=MODEL_ORDER)
    print()

    # 3. Cross-dataset
    print("--- Cross-dataset ---")
    crossds_path = f'{RESULTS_DIR}/crossds.csv'
    if os.path.exists(crossds_path):
        crossds = pd.read_csv(crossds_path)
        crossds = filter_valid(crossds)
        all_results['crossds'] = crossds
        print(f"  Total: {len(crossds)} valid results")
        summary = crossds.groupby('model')['spearman_mean'].agg(['mean', 'std', 'median', 'count'])
        print(summary.to_string())
        create_boxplot(crossds, 'spearman_mean',
                       'Cross-dataset (CAMP→ccRCC): Spearman ρ by Method',
                       'crossds_spearman_boxplot', order=MODEL_ORDER)
    print()

    # 4. Ablation
    print("--- Ablation Study ---")
    ablation_path = f'{RESULTS_DIR}/ablation_indist.csv'
    if os.path.exists(ablation_path):
        ablation = pd.read_csv(ablation_path)
        ablation = filter_valid(ablation)
        if len(ablation) > 0:
            print(f"  Total: {len(ablation)} valid results")
            summary = ablation.groupby('model')['spearman_mean'].agg(['mean', 'std', 'median', 'count'])
            print(summary.to_string())
            create_ablation_boxplot(ablation)
    print()

    # 5. Multi-protocol heatmap
    print("--- Multi-protocol comparison ---")
    if any(len(v) > 0 for v in all_results.values()):
        create_multi_protocol_heatmap(all_results)

    # 6. Summary table
    print("\n--- Summary Table ---")
    summary_table = create_summary_table(all_results)
    if len(summary_table) > 0:
        print(summary_table.to_string(index=False))
        summary_table.to_csv(f'{RESULTS_DIR}/summary_table.csv', index=False)
        print(f"\n  Saved: summary_table.csv")

    print(f"\n=== Analysis complete ===")
    print(f"Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    run_analysis()
