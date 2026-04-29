#!/usr/bin/env python
# coding: utf-8
"""Sliding Window and Peak Detection Analysis"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

from .metrics import (
    calc_bhattacharyya_distance,
    calc_wasserstein,
    calc_isoforest_anomaly,
    get_entropy_weights,
)


def sliding_window_scan(
    resnet_features,
    time_vec,
    step_ratio=0.02,
    safe_margin_ratio=0.02,
    min_cells_ratio=0.025,
    random_state=42,
):
    """
    Slide a window along pseudotime to compute Bhattacharyya, Wasserstein, and IsolationForest scores for each window.
    
    Parameters
    ----------
    resnet_features : np.ndarray
        Deep feature matrix of shape (n_cells, out_dim).
    time_vec : np.ndarray
        Time/Pseudotime vector.
    step_ratio : float
        Step size as a fraction of the total time/pseudotime span.
    safe_margin_ratio : float
        Safety margin at the boundaries, expressed as a fraction of the total time/pseudotime span.
    min_cells_ratio : float
        Minimum fraction of cells required within a window relative to the total number of cells (lower bound 10).
    random_state : int
        Random seed for IsolationForest.
    
    Returns
    -------
    df_res : pd.DataFrame
        DataFrame containing each scanning point t along time/pseudotime and the corresponding computed metrics.
     """
    total_cells = len(time_vec)
    total_span = time_vec.max() - time_vec.min()

    step_size = total_span * step_ratio
    SAFE_MARGIN = total_span * safe_margin_ratio
    MIN_CELLS_FLOOR = max(10, int(total_cells * min_cells_ratio))

    results = []
    for t in np.arange(time_vec.min() + SAFE_MARGIN, time_vec.max() - SAFE_MARGIN, step_size):
        mask_a = time_vec < t
        mask_b = time_vec >= t
        if mask_a.sum() < MIN_CELLS_FLOOR or mask_b.sum() < MIN_CELLS_FLOOR:
            continue
        idx_a = np.where(mask_a)[0][np.argsort(time_vec[mask_a])][-MIN_CELLS_FLOOR:]
        idx_b = np.where(mask_b)[0][np.argsort(time_vec[mask_b])][:MIN_CELLS_FLOOR]
        X_a, X_b = resnet_features[idx_a], resnet_features[idx_b]

        try:
            d_b = calc_bhattacharyya_distance(X_a, X_b)
            d_w = calc_wasserstein(X_a, X_b)
            d_if = calc_isoforest_anomaly(X_a, X_b, random_state=random_state)
            if not np.isnan(d_b):
                results.append({'t': t, 'bhatt': d_b, 'wass': d_w, 'isoforest': d_if})
        except:
            continue

    return pd.DataFrame(results)


def compute_combined_score(df_res, smooth_window=7, polyorder=2):
    """
    Compute a combined score and apply smoothing.
    
    Parameters
    ----------
    df_res : pd.DataFrame
        DataFrame containing the scanning results.
    smooth_window : int
        Window length for Savitzky–Golay filter (must be odd).
    polyorder : int
        Polynomial order for Savitzky–Golay filter.
    
    Returns
    -------
    df_res : pd.DataFrame
        DataFrame with added columns `combined_score` and `combined_score_smooth`.
    norm_scores : np.ndarray
        Normalized scores for each metric.
    weights : np.ndarray
        Weights for each metric computed using the entropy weight method.
     """
    metrics = ['bhatt', 'wass', 'isoforest']
    norm_scores, weights = get_entropy_weights(df_res[metrics].values)
    df_res['combined_score'] = np.dot(norm_scores, weights)

    # Ensure that `smooth_window` is odd and does not exceed the length of the data
    wl = smooth_window
    n = len(df_res)
    if wl >= n:
        wl = n - 1 if n % 2 == 0 else n
    if wl % 2 == 0:
        wl -= 1
    wl = max(3, wl)

    df_res['combined_score_smooth'] = savgol_filter(
        df_res['combined_score'], window_length=wl, polyorder=polyorder
    )
    return df_res, norm_scores, weights


def find_tipping_peaks(df_res, prominence=0.1, distance=5):
    """
    Automatically detect peaks and generate a rank report.
    
    Parameters
    ----------
    df_res : pd.DataFrame
        DataFrame containing the `combined_score_smooth` column.
    prominence : float
        Minimum prominence of peaks.
    distance : int
        Minimum distance between peaks (in number of points).
    
    Returns
    -------
    peak_report : pd.DataFrame
        DataFrame reporting detected peaks, including columns `t`, `Confidence`, and `Rank`.
     """
    peaks, _ = find_peaks(df_res['combined_score_smooth'], prominence=prominence, distance=distance)

    peak_report = pd.DataFrame({
        't': df_res.loc[peaks, 't'].values,
        'Confidence': df_res.loc[peaks, 'combined_score_smooth'].values
    }).sort_values('Confidence', ascending=False)
    peak_report['Rank'] = range(1, len(peak_report) + 1)
    return peak_report
