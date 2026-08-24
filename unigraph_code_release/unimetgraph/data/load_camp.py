"""
Load CAMP pan-cancer metabolomics data using the MasterMapping file.
Creates matched metabolomics + transcriptomics matrices.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict


CAMP_DATA_DIR = "/workspace/data/pancancer_metabolomics/data"
MET_DIR = f"{CAMP_DATA_DIR}/metabolomics_processed"
RNA_DIR = f"{CAMP_DATA_DIR}/transcriptomics_processed"
MAPPING_FILE = f"{CAMP_DATA_DIR}/MasterMapping_MetImmune_03_16_2022_release.csv"


def load_master_mapping() -> pd.DataFrame:
    """Load the MasterMapping file that links metabolomics and transcriptomics samples."""
    mapping = pd.read_csv(MAPPING_FILE)
    # Filter to Tumor samples only for primary analysis
    # (Normal samples can be used for additional validation if needed)
    return mapping


def load_camp_data(tumor_only: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load CAMP data and create matched metabolomics + transcriptomics matrices.

    Returns:
        met_data: (N_samples, N_metabolites) metabolomics abundance matrix
        rna_data: (N_samples, N_genes) gene expression matrix
        met_anno: metabolite annotations (BIOCHEMICAL, HMDB, KEGG, etc.)
        sample_info: sample metadata (dataset, histology, TN)
        batch_info: dict with batch_index_vector, start_row, stop_row, batch_names
    """
    mapping = load_master_mapping()
    if tumor_only:
        mapping = mapping[mapping['TN'] == 'Tumor'].reset_index(drop=True)

    # Group by dataset to maintain batch structure
    datasets = mapping['Dataset'].unique()
    print(f"Loading {len(mapping)} samples across {len(datasets)} datasets...")

    # --- Load metabolomics data ---
    # Collect all metabolite names across datasets
    met_dfs = {}
    met_anno_dfs = {}
    all_met_names = set()

    for ds in datasets:
        ds_mapping = mapping[mapping['Dataset'] == ds]
        metab_file = ds_mapping['MetabFile'].iloc[0]
        metab_sheet = ds_mapping['MetabFile_sheet'].iloc[0]

        fpath = f"{MET_DIR}/{metab_file}"
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping {ds}")
            continue

        # Load metabolomics data (metabolites × samples)
        met_df = pd.read_excel(fpath, sheet_name=metab_sheet, index_col=0)
        met_df = met_df.T  # transpose to samples × metabolites

        # Load metabolite annotations
        anno = pd.read_excel(fpath, sheet_name='metanno')

        # Filter to mapped samples
        metab_ids = ds_mapping['MetabID'].tolist()
        available = [m for m in metab_ids if m in met_df.index]
        if len(available) < len(metab_ids):
            print(f"  {ds}: {len(available)}/{len(metab_ids)} metabolomics samples found")

        met_df = met_df.loc[available]
        met_dfs[ds] = met_df
        met_anno_dfs[ds] = anno
        all_met_names.update(met_df.columns)

    all_met_names = sorted(all_met_names)
    print(f"Total unique metabolites: {len(all_met_names)}")

    # --- Load transcriptomics data ---
    rna_dfs = {}
    all_gene_sets = []

    for ds in datasets:
        ds_mapping = mapping[mapping['Dataset'] == ds]
        rna_file = ds_mapping['RNAFile'].iloc[0]

        fpath = f"{RNA_DIR}/{rna_file}"
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping {ds}")
            continue

        # Load RNA-seq data (genes × samples)
        rna_df = pd.read_csv(fpath, index_col=0)
        rna_df = rna_df.T  # transpose to samples × genes

        # Filter to mapped samples
        rna_ids = ds_mapping['RNAID'].tolist()
        available = [r for r in rna_ids if r in rna_df.index]
        if len(available) < len(rna_ids):
            print(f"  {ds}: {len(available)}/{len(rna_ids)} RNA samples found")

        rna_df = rna_df.loc[available]
        rna_dfs[ds] = rna_df
        all_gene_sets.append(set(rna_df.columns))

    # Compute gene intersection
    common_genes = all_gene_sets[0]
    for gs in all_gene_sets[1:]:
        common_genes = common_genes.intersection(gs)
    common_genes = sorted(common_genes)
    print(f"Gene intersection across datasets: {len(common_genes)}")

    # --- Build unified matrices ---
    # Order datasets consistently
    valid_datasets = [ds for ds in datasets if ds in met_dfs and ds in rna_dfs]

    # Build sample ordering: for each dataset, match met and RNA samples
    sample_list = []
    batch_index_vector = []
    batch_names = []
    start_row = []
    stop_row = []
    met_rows = []  # list of pd.Series indexed by metabolite name
    rna_rows = []

    sidx = 0
    for bidx, ds in enumerate(valid_datasets):
        ds_mapping = mapping[mapping['Dataset'] == ds]
        met_df = met_dfs[ds]
        rna_df = rna_dfs[ds]

        # Create sample mapping: MetabID -> RNAID
        id_map = dict(zip(ds_mapping['MetabID'], ds_mapping['RNAID']))

        # Find matched samples (both met and RNA available)
        matched_met_ids = [m for m in met_df.index if m in id_map and id_map[m] in rna_df.index]

        if len(matched_met_ids) == 0:
            print(f"  WARNING: No matched samples for {ds}, skipping")
            continue

        for met_id in matched_met_ids:
            rna_id = id_map[met_id]
            met_rows.append(met_df.loc[met_id])  # keep as Series for alignment
            rna_rows.append(rna_df.loc[rna_id, common_genes].values)
            sample_list.append({
                'sample_id': met_id,
                'rna_id': rna_id,
                'dataset': ds,
                'histology': ds_mapping[ds_mapping['MetabID'] == met_id]['Histology'].iloc[0],
                'TN': ds_mapping[ds_mapping['MetabID'] == met_id]['TN'].iloc[0],
            })

        batch_names.append(ds)
        start_row.append(sidx)
        n_samples = len(matched_met_ids)
        stop_row.append(sidx + n_samples)
        batch_index_vector.extend([bidx] * n_samples)
        sidx += n_samples
        print(f"  {ds}: {n_samples} matched samples")

    # Build unified metabolite matrix (reindex each row to all_met_names, NaN for missing)
    met_df_all = pd.DataFrame(met_rows)
    met_df_all = met_df_all.reindex(columns=all_met_names)
    met_data = met_df_all.values.astype(float)
    rna_data = np.array(rna_rows, dtype=float)
    sample_info = pd.DataFrame(sample_list)

    # Build metabolite annotation dataframe
    # Merge annotations from all datasets (deduplicate by BIOCHEMICAL)
    all_anno = pd.concat(met_anno_dfs.values()).drop_duplicates(subset='BIOCHEMICAL')
    # Reindex to match all_met_names
    met_anno = all_anno.set_index('BIOCHEMICAL').reindex(all_met_names).reset_index()
    met_names = all_met_names

    batch_info = {
        'batch_index_vector': np.array(batch_index_vector),
        'start_row': np.array(start_row),
        'stop_row': np.array(stop_row),
        'batch_names': batch_names,
        'n_batch': len(batch_names),
        'gene_names': common_genes,
    }

    print(f"\nFinal data shapes:")
    print(f"  Metabolomics: {met_data.shape} (samples × metabolites)")
    print(f"  Transcriptomics: {rna_data.shape} (samples × genes)")
    print(f"  Batches: {batch_info['n_batch']}")

    return met_data, rna_data, met_anno, met_names, sample_info, batch_info


if __name__ == "__main__":
    met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)
    print(f"\nSample info:")
    print(sample_info['dataset'].value_counts())
    print(f"\nMetabolite annotation coverage:")
    print(f"  HMDB: {met_anno['H_HMDB'].notna().sum()}/{len(met_anno)}")
    print(f"  KEGG: {met_anno['H_KEGG'].notna().sum()}/{len(met_anno)}")
