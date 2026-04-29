#!/usr/bin/env python
# coding: utf-8
"""
Bio-SPC: A Python library for identifying cliff points along time/pseudotime trajectories from single-cell data.

Exposed core function: bio_spc_pipeline
"""

import os
import numpy as np
import pandas as pd

from .utils import set_seed
from .data import load_and_validate
from .features import extract_resnet_features
from .analysis import sliding_window_scan, compute_combined_score, find_tipping_peaks
from .extract_cell import extract_tipping_cells
from .visual import plot_cpi_curve


def bio_spc_pipeline(
    adata,
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
    save_dir='./',
    random_seed=42,
    do_plot=True,
    colors=None,
    alpha=0.2,
    figsize=(12, 6),
):
    """
    Bio-SPC developmental/aging/evolutionary cliff point detection pipeline.

    Parameters
    ----------
    adata : AnnData
        Single-cell data object.
    time_col : str
        Column name for pseudotime or continuous trajectory in `adata.obs`. Default is 'Pseudotime'.
    out_dim : int
        Dimensionality of features after ResNet reduction. Default is 30.
    hidden_dim : int
        Hidden layer dimension of the ResNet. Default is 128.
    epochs : int
        Number of training epochs for the ResNet. Default is 1000.
    lr : float
        Learning rate for ResNet training. Default is 1e-4.
    step_ratio : float
        Sliding window step size as a fraction of the total pseudotime span. Default is 0.02.
    safe_margin_ratio : float
        Safety margin at the trajectory boundaries, as a fraction of the total pseudotime span. Default is 0.02.
    min_cells_ratio : float
        Minimum fraction of cells required within a window relative to the total number of cells (lower bound 10). Default is 0.025.
    smooth_window : int
        Window length for Savitzky–Golay filter (must be odd). Default is 7.
    polyorder : int
        Polynomial order for Savitzky–Golay filter. Default is 2.
    prominence : float
        Minimum prominence of peaks for peak detection. Default is 0.1.
    distance : int
        Minimum distance between detected peaks (in number of points). Default is 5.
    extract_ratio : float
        Extraction window radius as a fraction of the total pseudotime span. Default is 0.04.
    save_dir : str
        Directory to save output results. Default is './'.
    random_seed : int
        Global random seed. Default is 42.
    do_plot : bool
        Whether to generate visualization plots. Default is True.
    colors : list
        Colors for the original metric curves. Default is ['#3498db', '#9b59b6', '#e67e22'].
    alpha : float
        Transparency of the original metric curves. Default is 0.2.
    figsize : tuple
        Figure size. Default is (12, 6).

    Returns
    -------
    peak_report : pd.DataFrame or None
        DataFrame reporting detected peaks with columns `Rank`, `t`, and `Confidence`. Returns None if no valid cliff points are detected.
    """
    # 0. Fix random seed
    set_seed(random_seed)

    # 1. Data validation
    time_vec = load_and_validate(adata, time_col=time_col)
    total_cells = len(time_vec)
    total_span = time_vec.max() - time_vec.min()

    # 2. Feature extraction
    print("ResNet1D feature extraction is in progress...")
    X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_pca = extract_resnet_features(
        X_data,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
    )
    print("ResNet feature extraction completed")

    # 3. Sliding Window Scan
    print("Scanning trajectory and calculating metrics...")
    df_res = sliding_window_scan(
        X_pca,
        time_vec,
        step_ratio=step_ratio,
        safe_margin_ratio=safe_margin_ratio,
        min_cells_ratio=min_cells_ratio,
        random_state=random_seed,
    )

    if len(df_res) <= 5:
        print("The number of valid scanning points is insufficient to determine cliff points.")
        return None

    # 4. Compute combined scores and apply smoothing
    df_res, norm_scores, weights = compute_combined_score(
        df_res,
        smooth_window=smooth_window,
        polyorder=polyorder,
    )

    # 5. Perform automatic peak detection
    peak_report = find_tipping_peaks(
        df_res,
        prominence=prominence,
        distance=distance,
    )

    # Print the report
    print("\n" + "=" * 45)
    print("Bio-SPC Cliff Point Detection Report")
    print("=" * 45)
    print(f"Current adaptive weights: \nBhatt: {weights[0]:.2f} | Wass: {weights[1]:.2f} | IsoForest: {weights[2]:.2f}")
    print("-" * 45)
    if not peak_report.empty:
        print(peak_report[['Rank', 't', 'Confidence']].to_string(index=False))
    else:
        print("No significant cliff points detected.")
    print("=" * 45)

    # 6. Visualization
    if do_plot and not peak_report.empty:
        plot_colors = colors if colors is not None else ['#3498db', '#9b59b6', '#e67e22']
        plot_cpi_curve(
            df_res,
            norm_scores,
            peak_report,
            metrics=['bhatt', 'wass', 'isoforest'],
            colors=plot_colors,
            alpha=alpha,
            figsize=figsize,
            time_col=time_col,
        )

    # 7. Cell Extraction and Export
    if not peak_report.empty:
        extract_tipping_cells(
            adata,
            peak_report,
            time_col=time_col,
            total_span=total_span,
            extract_ratio=extract_ratio,
            save_dir=save_dir,
        )

    return peak_report
