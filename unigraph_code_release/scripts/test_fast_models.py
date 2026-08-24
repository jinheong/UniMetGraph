#!/usr/bin/env python3
"""Quick test of fast baseline models to find timing bottlenecks."""
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/UnitedMet')

import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from unigraph.data.load_camp import load_camp_data
from unigraph.data.preprocess import tic_normalization_across, log_transform_rna, normalize_rna
from unigraph.models.baselines_fast import FastRidgeBaseline, FastLassoBaseline

# Load data
print("Loading CAMP data...")
t0 = time.time()
met_data, rna_data, met_anno, met_names, sample_info, batch_info = load_camp_data(tumor_only=True)
print(f"  Data loaded in {time.time()-t0:.1f}s")
print(f"  met: {met_data.shape}, rna: {rna_data.shape}")

# Use small subset
np.random.seed(42)
test_idx = np.random.choice(764, 100, replace=False)
test_idx = np.sort(test_idx)
test_met = met_data[test_idx]
test_rna = rna_data[test_idx]
test_batch = {
    'batch_index_vector': np.array([0]*50 + [1]*50),
    'start_row': np.array([0, 50]),
    'stop_row': np.array([50, 100]),
    'batch_names': ['batch0', 'batch1'],
    'n_batch': 2,
}

# Test Ridge (should be very fast)
print("\n--- Ridge ---")
t0 = time.time()
model = FastRidgeBaseline(n_top_genes=500, seed=42)
model.fit(test_met, test_rna, test_batch, verbose=True)
print(f"  Fit time: {time.time()-t0:.1f}s")

t0 = time.time()
preds = model.predict_met_ranks(test_rna, test_batch)
print(f"  Predict time: {time.time()-t0:.1f}s")
print(f"  Pred shape: {preds['rank_hat_mean'].shape}")

# Test Lasso
print("\n--- Lasso ---")
t0 = time.time()
model = FastLassoBaseline(n_top_genes=200, alpha=0.01, seed=42)
model.fit(test_met, test_rna, test_batch, verbose=True)
print(f"  Fit time: {time.time()-t0:.1f}s")

t0 = time.time()
preds = model.predict_met_ranks(test_rna, test_batch)
print(f"  Predict time: {time.time()-t0:.1f}s")

# Test TIC normalization timing
print("\n--- TIC normalization ---")
t0 = time.time()
met_tic = tic_normalization_across(test_met, test_batch['batch_index_vector'])
print(f"  TIC time: {time.time()-t0:.1f}s")

# Test RNA normalization
print("\n--- RNA normalization ---")
t0 = time.time()
rna_log = log_transform_rna(test_rna)
rna_norm = normalize_rna(rna_log)
print(f"  RNA norm time: {time.time()-t0:.1f}s")

print("\nAll tests passed!")
