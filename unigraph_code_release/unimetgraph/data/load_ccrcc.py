"""
Load ccRCC matched metabolomics + transcriptomics data from Zenodo files.
Used as cross-dataset external validation.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict


CCRCC_DIR = "/workspace/data/ccrcc"

# Dataset name -> (metabolomics file, RNA-seq file)
CCRCC_FILES = {
    'CPTAC': ('matched_Harmonized_Met_CPTAC.csv', 'matched_tpm_CPTAC.csv'),
    'CPTAC_val': ('matched_Harmonized_Met_CPTAC_val.csv', 'matched_tpm_CPTAC_val.csv'),
    'RC18': ('matched_Harmonized_RawData_RC18.csv', 'matched_tpm_RC18.csv'),
    'RC20': ('matched_Harmonized_RawData_RC20.csv', 'matched_tpm_RC20.csv'),
}


def load_ccrcc_data() -> Tuple[np.ndarray, np.ndarray, list, list, pd.DataFrame, Dict]:
    """
    Load all 4 ccRCC datasets and create unified matrices.

    Returns:
        met_data: (N_samples, N_metabolites) metabolomics abundance matrix
        rna_data: (N_samples, N_genes) gene expression matrix
        met_names: list of metabolite names
        gene_names: list of gene names
        sample_info: sample metadata (dataset, sample_id)
        batch_info: dict with batch_index_vector, start_row, stop_row, batch_names
    """
    met_dfs = {}
    rna_dfs = {}

    for ds_name, (met_file, rna_file) in CCRCC_FILES.items():
        met_path = os.path.join(CCRCC_DIR, met_file)
        rna_path = os.path.join(CCRCC_DIR, rna_file)

        met_df = pd.read_csv(met_path, index_col=0)
        rna_df = pd.read_csv(rna_path, index_col=0)

        # Ensure sample alignment
        common_samples = sorted(set(met_df.index) & set(rna_df.index))
        met_df = met_df.loc[common_samples]
        rna_df = rna_df.loc[common_samples]

        met_dfs[ds_name] = met_df
        rna_dfs[ds_name] = rna_df
        print(f"  {ds_name}: {len(common_samples)} samples, "
              f"{met_df.shape[1]} metabolites, {rna_df.shape[1]} genes")

    # Collect all metabolite names (union)
    all_met_names = sorted(set().union(*[set(df.columns) for df in met_dfs.values()]))
    print(f"Total unique metabolites: {len(all_met_names)}")

    # Gene intersection
    gene_sets = [set(df.columns) for df in rna_dfs.values()]
    common_genes = sorted(gene_sets[0].intersection(*gene_sets[1:]))
    print(f"Gene intersection: {len(common_genes)}")

    # Build unified matrices
    sample_list = []
    batch_index_vector = []
    batch_names = []
    start_row = []
    stop_row = []
    met_rows = []
    rna_rows = []

    sidx = 0
    for bidx, ds_name in enumerate(CCRCC_FILES.keys()):
        met_df = met_dfs[ds_name]
        rna_df = rna_dfs[ds_name]

        for sample_id in met_df.index:
            met_rows.append(met_df.loc[sample_id])
            rna_rows.append(rna_df.loc[sample_id, common_genes].values)
            sample_list.append({
                'sample_id': sample_id,
                'dataset': ds_name,
            })

        batch_names.append(ds_name)
        start_row.append(sidx)
        n = len(met_df)
        stop_row.append(sidx + n)
        batch_index_vector.extend([bidx] * n)
        sidx += n

    # Build unified metabolite matrix
    met_df_all = pd.DataFrame(met_rows)
    met_df_all = met_df_all.reindex(columns=all_met_names)
    met_data = met_df_all.values.astype(float)
    rna_data = np.array(rna_rows, dtype=float)
    sample_info = pd.DataFrame(sample_list)

    batch_info = {
        'batch_index_vector': np.array(batch_index_vector),
        'start_row': np.array(start_row),
        'stop_row': np.array(stop_row),
        'batch_names': batch_names,
        'n_batch': len(batch_names),
        'gene_names': common_genes,
    }

    print(f"\nFinal data shapes:")
    print(f"  Metabolomics: {met_data.shape}")
    print(f"  Transcriptomics: {rna_data.shape}")
    print(f"  Batches: {batch_info['n_batch']}")

    return met_data, rna_data, all_met_names, common_genes, sample_info, batch_info


if __name__ == "__main__":
    met_data, rna_data, met_names, gene_names, sample_info, batch_info = load_ccrcc_data()
    print(f"\nSample info:")
    print(sample_info['dataset'].value_counts())
