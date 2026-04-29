#!/usr/bin/env python
# coding: utf-8
"""
Bio-SPC test script: Verify that the reconstructed library runs correctly.

"""

import os
import scanpy as sc
import pandas as pd
from biospc import bio_spc_pipeline

os.chdir('./Bio-SPC')

# Loading data
print("Loading data...")
adata = sc.read("./data/scRNAepithgc.h5ad")
metadata = pd.read_csv("./data/gc_epith_monocle_pseudo_meta.csv", index_col=0)
adata.obs['Pseudotime'] = metadata['Pseudotime']
print(f"Data loading complete: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Running Bio-SPC Pipeline
print("\n" + "=" * 60)
print("Running Bio-SPC Pipeline")
print("=" * 60)

peak_report = bio_spc_pipeline(
    adata=adata,
    time_col='Pseudotime',
    out_dim=30,
    hidden_dim=128,
    epochs=1000,
    lr=1e-4,
    step_ratio=0.02,
    safe_margin_ratio=0.02,
    min_cells_ratio=0.025,
    smooth_window=7,
    polyorder=2,
    prominence=0.1,
    distance=5,
    extract_ratio=0.04,
    save_dir='./bio_spc_output',
    random_seed=42,
    do_plot=True,
    colors=['#3498db', '#9b59b6', '#e67e22'],
    alpha=0.2,
    figsize=(12, 6),
)

if peak_report is not None:
    print("\nPipeline run successful!")
    print(f"Detected {len(peak_report)} cliff points")
    print(peak_report[['Rank', 't', 'Confidence']].to_string(index=False))
else:
    print("\nPipeline completed, but no valid cliff points detected.")