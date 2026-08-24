"""
Construct the Human-GEM metabolic graph and map CAMP/ccRCC metabolites to it.
Builds a bipartite metabolite-reaction graph with chemical and gene expression features.
"""
import os
import re
import numpy as np
import pandas as pd
import cobra
import warnings
import urllib.request
import json
from typing import Tuple, Dict, Optional
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

warnings.filterwarnings('ignore')

HUMAN_GEM_DIR = "/workspace/Human-GEM"
HUMAN_GEM_MODEL = f"{HUMAN_GEM_DIR}/model/Human-GEM.yml"
HUMAN_GEM_MET_TSV = f"{HUMAN_GEM_DIR}/model/metabolites.tsv"
HUMAN_GEM_GENE_TSV = f"{HUMAN_GEM_DIR}/model/genes.tsv"


def normalize_hmdb(h):
    """Normalize HMDB ID to consistent format HMDBXXXXxxx (7 digits)."""
    h = str(h).strip()
    if ',' in h:
        h = h.split(',')[0].strip()
    m = re.match(r'HMDB0*(\d+)', h)
    if m:
        return f'HMDB{int(m.group(1)):07d}'
    return h


def load_human_gem():
    """Load Human-GEM model and annotation tables."""
    model = cobra.io.load_yaml_model(HUMAN_GEM_MODEL)
    met_anno = pd.read_csv(HUMAN_GEM_MET_TSV, sep='\t')
    gene_anno = pd.read_csv(HUMAN_GEM_GENE_TSV, sep='\t')
    return model, met_anno, gene_anno


def build_bipartite_graph(model) -> Dict:
    """Build bipartite metabolite-reaction graph from stoichiometric matrix."""
    met_ids = [m.id for m in model.metabolites]
    rxn_ids = [r.id for r in model.reactions]
    met_idx = {m: i for i, m in enumerate(met_ids)}
    rxn_idx = {r: i for i, r in enumerate(rxn_ids)}

    edges = []
    stoich = []
    for rxn in model.reactions:
        r_i = rxn_idx[rxn.id]
        for met, coeff in rxn.metabolites.items():
            m_i = met_idx[met.id]
            edges.append([m_i, r_i])
            stoich.append(coeff)

    edge_index = np.array(edges, dtype=np.int64).T
    stoichiometry = np.array(stoich, dtype=np.float32)

    print(f"Bipartite graph: {len(met_ids)} metabolites, {len(rxn_ids)} reactions, {len(edges)} edges")
    return {
        'met_ids': met_ids,
        'rxn_ids': rxn_ids,
        'edge_index': edge_index,
        'stoichiometry': stoichiometry,
    }


def map_camp_to_human_gem(met_anno_camp: pd.DataFrame, met_anno_hgem: pd.DataFrame) -> Dict:
    """
    Map CAMP metabolites to Human-GEM metabolites using HMDB and KEGG IDs.
    Returns dict: {camp_met_idx: hgem_metsNoComp}
    """
    # Build lookup from Human-GEM annotations (using metsNoComp as canonical)
    hgem_by_hmdb = {}
    hgem_by_kegg = {}

    for _, row in met_anno_hgem.iterrows():
        met_nocomp = row['metsNoComp']
        if pd.notna(row['metHMDBID']):
            h = normalize_hmdb(row['metHMDBID'])
            hgem_by_hmdb[h] = met_nocomp
        if pd.notna(row['metKEGGID']):
            k = str(row['metKEGGID']).strip()
            hgem_by_kegg[k] = met_nocomp

    # Map CAMP metabolites
    mapping = {}
    n_hmdb = 0
    n_kegg = 0

    for i, row in met_anno_camp.iterrows():
        mapped = None

        # Try HMDB first
        if pd.notna(row.get('H_HMDB')):
            h = normalize_hmdb(row['H_HMDB'])
            if h in hgem_by_hmdb:
                mapped = hgem_by_hmdb[h]
                n_hmdb += 1

        # Try KEGG
        if mapped is None and pd.notna(row.get('H_KEGG')):
            k = str(row['H_KEGG']).strip()
            if k in hgem_by_kegg:
                mapped = hgem_by_kegg[k]
                n_kegg += 1

        if mapped is not None:
            mapping[i] = mapped

    print(f"CAMP -> Human-GEM mapping: {len(mapping)}/{len(met_anno_camp)} "
          f"(HMDB: {n_hmdb}, KEGG: {n_kegg})")
    return mapping


def get_smiles_from_pubchem(pubchem_ids: list, batch_size: int = 50) -> Dict:
    """
    Get SMILES strings from PubChem CIDs using the REST API.
    Returns dict: {pubchem_cid_str: smiles}
    """
    smiles_map = {}
    cids = [str(int(cid)) for cid in pubchem_ids if pd.notna(cid)]

    for i in range(0, len(cids), batch_size):
        batch = cids[i:i+batch_size]
        cid_str = ','.join(batch)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES,ConnectivitySMILES/JSON"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())
                props = data.get('PropertyTable', {}).get('Properties', [])
                for prop in props:
                    cid = str(prop['CID'])
                    smiles = prop.get('CanonicalSMILES') or prop.get('ConnectivitySMILES', '')
                    if smiles:
                        smiles_map[cid] = smiles
        except Exception as e:
            print(f"  Batch {i//batch_size}: error - {e}")
            # Try smaller batches
            for cid in batch:
                try:
                    url_single = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,ConnectivitySMILES/JSON"
                    req = urllib.request.Request(url_single)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read())
                        props = data.get('PropertyTable', {}).get('Properties', [])
                        if props:
                            smiles = props[0].get('CanonicalSMILES') or props[0].get('ConnectivitySMILES', '')
                            if smiles:
                                smiles_map[cid] = smiles
                except:
                    pass

        if i % 200 == 0:
            print(f"  Processed {i}/{len(cids)} PubChem CIDs, got {len(smiles_map)} SMILES")

    return smiles_map


def compute_morgan_fingerprints(smiles_map: Dict, met_ids: list, n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Morgan fingerprints for metabolites."""
    n_mets = len(met_ids)
    fingerprints = np.zeros((n_mets, n_bits), dtype=np.float32)
    valid_mask = np.zeros(n_mets, dtype=bool)

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)

    for i, met_id in enumerate(met_ids):
        smiles = smiles_map.get(met_id)
        if smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = mfpgen.GetFingerprintAsNumPy(mol)
                fingerprints[i] = fp
                valid_mask[i] = True

    print(f"Morgan fingerprints: {valid_mask.sum()}/{n_mets} valid")
    return fingerprints, valid_mask


def build_reaction_gene_features(model, gene_anno: pd.DataFrame, gene_names: list) -> Dict:
    """Build reaction-gene associations and map to gene expression indices."""
    ensg_to_symbol = dict(zip(gene_anno['genes'], gene_anno['geneSymbols']))
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    rxn_gene_indices = []
    for rxn in model.reactions:
        gpr = rxn.gene_reaction_rule
        if not gpr or gpr == '':
            rxn_gene_indices.append([])
            continue

        gene_ids = re.findall(r'ENSG\d+', gpr)
        gene_idxs = []
        for gid in gene_ids:
            symbol = ensg_to_symbol.get(gid)
            if symbol and symbol in gene_to_idx:
                gene_idxs.append(gene_to_idx[symbol])
        rxn_gene_indices.append(gene_idxs)

    n_with_genes = sum(1 for g in rxn_gene_indices if len(g) > 0)
    print(f"Reactions with mapped genes: {n_with_genes}/{len(rxn_gene_indices)}")

    return {'rxn_gene_indices': rxn_gene_indices}


def construct_graph(met_anno_camp: pd.DataFrame, gene_names: list,
                    rna_data: np.ndarray = None, cache_dir: str = "/workspace/data/graph_cache") -> Dict:
    """
    Full graph construction pipeline with caching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/graph_data.npz"
    smiles_cache = f"{cache_dir}/smiles_map.json"

    # Try loading from cache
    if os.path.exists(cache_file) and os.path.exists(smiles_cache):
        print("Loading graph from cache...")
        data = np.load(cache_file, allow_pickle=True)
        with open(smiles_cache) as f:
            smiles_map = json.load(f)
        return {
            'hgem_met_ids': data['hgem_met_ids'].tolist(),
            'camp_to_hgem': dict(zip(data['camp_indices'].tolist(), data['hgem_met_mapped'].tolist())),
            'fingerprints': data['fingerprints'],
            'fp_valid': data['fp_valid'],
            'met_met_edges': data['met_met_edges'],
            'smiles_map': smiles_map,
            'rxn_gene_indices': data['rxn_gene_indices'].tolist(),
            'gene_names': gene_names,
        }

    print("Loading Human-GEM model...")
    model, met_anno_hgem, gene_anno = load_human_gem()

    print("\nBuilding bipartite graph...")
    graph = build_bipartite_graph(model)

    print("\nMapping CAMP metabolites to Human-GEM...")
    camp_to_hgem = map_camp_to_human_gem(met_anno_camp, met_anno_hgem)

    # Get unique Human-GEM metabolites (without compartment) that CAMP maps to
    hgem_met_nocomp = sorted(set(camp_to_hgem.values()))

    # Get PubChem IDs for these metabolites
    hgem_pubchem = {}
    for _, row in met_anno_hgem.iterrows():
        if row['metsNoComp'] in hgem_met_nocomp and pd.notna(row['metPubChemID']):
            hgem_pubchem[row['metsNoComp']] = row['metPubChemID']

    print(f"\nPubChem IDs for mapped metabolites: {len(hgem_pubchem)}/{len(hgem_met_nocomp)}")

    # Get SMILES from PubChem
    print("\nFetching SMILES from PubChem...")
    pubchem_ids = list(hgem_pubchem.values())
    smiles_by_cid = get_smiles_from_pubchem(pubchem_ids)

    # Map metabolite -> SMILES
    smiles_map = {}
    for met_nocomp, pubchem_id in hgem_pubchem.items():
        cid = str(int(pubchem_id))
        if cid in smiles_by_cid:
            smiles_map[met_nocomp] = smiles_by_cid[cid]

    print(f"SMILES obtained: {len(smiles_map)}/{len(hgem_met_nocomp)}")

    # Save SMILES cache
    with open(smiles_cache, 'w') as f:
        json.dump(smiles_map, f)

    # Compute Morgan fingerprints
    print("\nComputing Morgan fingerprints...")
    fingerprints, fp_valid = compute_morgan_fingerprints(smiles_map, hgem_met_nocomp, n_bits=2048)

    # Build reaction-gene associations
    print("\nBuilding reaction-gene associations...")
    rxn_gene = build_reaction_gene_features(model, gene_anno, gene_names)

    # Collapse bipartite graph to metabolite-metabolite graph
    print("\nCollapsing to metabolite-metabolite graph...")
    met_nocomp_to_idx = {m: i for i, m in enumerate(hgem_met_nocomp)}

    # Map compartment-specific met IDs to metsNoComp
    met_id_to_nocomp = dict(zip(met_anno_hgem['mets'], met_anno_hgem['metsNoComp']))

    # Build met-met edges
    met_met_edges = set()
    for rxn in model.reactions:
        rxn_mets = [met_id_to_nocomp.get(m.id) for m in rxn.metabolites]
        rxn_mets = [m for m in rxn_mets if m is not None and m in met_nocomp_to_idx]
        for i in range(len(rxn_mets)):
            for j in range(i+1, len(rxn_mets)):
                idx_i = met_nocomp_to_idx[rxn_mets[i]]
                idx_j = met_nocomp_to_idx[rxn_mets[j]]
                if idx_i != idx_j:
                    met_met_edges.add((min(idx_i, idx_j), max(idx_i, idx_j)))

    met_met_edges = np.array(list(met_met_edges), dtype=np.int64).T
    print(f"Metabolite-metabolite graph: {len(hgem_met_nocomp)} nodes, {met_met_edges.shape[1]} edges")

    # Save to cache
    camp_indices = np.array(list(camp_to_hgem.keys()))
    hgem_met_mapped = np.array([camp_to_hgem[i] for i in camp_indices])
    np.savez(cache_file,
             hgem_met_ids=np.array(hgem_met_nocomp),
             camp_indices=camp_indices,
             hgem_met_mapped=hgem_met_mapped,
             fingerprints=fingerprints,
             fp_valid=fp_valid,
             met_met_edges=met_met_edges,
             rxn_gene_indices=np.array(rxn_gene['rxn_gene_indices'], dtype=object))

    return {
        'hgem_met_ids': hgem_met_nocomp,
        'camp_to_hgem': camp_to_hgem,
        'fingerprints': fingerprints,
        'fp_valid': fp_valid,
        'met_met_edges': met_met_edges,
        'smiles_map': smiles_map,
        'rxn_gene_indices': rxn_gene['rxn_gene_indices'],
        'gene_names': gene_names,
    }


if __name__ == "__main__":
    from unigraph.data.load_camp import load_camp_data
    met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)
    # Get gene names from one of the RNA files
    import pandas as pd
    rna_df = pd.read_csv('data/pancancer_metabolomics/data/transcriptomics_processed/Cornell_DLBCL.tpm.gene_symbol.csv', index_col=0)
    gene_names = list(rna_df.index)
    graph_data = construct_graph(met_anno, gene_names, rna_data)
