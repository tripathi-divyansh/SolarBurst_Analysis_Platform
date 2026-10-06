"""
Noise estimation, MAD spike screening, and standardized z/asinh transformations.
"""

from typing import Tuple, Dict, Any
import numpy as np
from ..core.quality import QualityFlag


def estimate_robust_noise(y: np.ndarray, mask_valid: np.ndarray) -> float:
    """
    Estimate local high-frequency noise level using median absolute deviation (MAD)
    of first differences on valid consecutive points.
    
    sigma = 1.4826 * MAD(dy) / sqrt(2)
    """
    valid_y = y[mask_valid]
    if len(valid_y) < 5:
        return float(np.nanstd(valid_y)) if len(valid_y) > 1 else 1.0

    diffs = np.diff(valid_y)
    med_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - med_diff))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return float(max(sigma, 1e-8))


def hampel_spike_screen(
    y: np.ndarray,
    mask_valid: np.ndarray,
    window_bins: int = 7,
    n_sigmas: float = 4.5
) -> np.ndarray:
    """
    Hampel filter to flag single-bin spikes without erasing them.
    Returns boolean array of suspect spike positions.
    """
    n = len(y)
    suspect_spikes = np.zeros(n, dtype=bool)
    half = window_bins // 2

    valid_indices = np.where(mask_valid)[0]
    for i_idx, i in enumerate(valid_indices):
        start = max(0, i_idx - half)
        end = min(len(valid_indices), i_idx + half + 1)
        sub_idx = valid_indices[start:end]
        
        # Exclude point itself for window median
        neighbors = sub_idx[sub_idx != i]
        if len(neighbors) < 3:
            continue

        local_med = np.median(y[neighbors])
        local_mad = 1.4826 * np.median(np.abs(y[neighbors] - local_med))
        thresh = max(n_sigmas * local_mad, 1e-6)

        # Check if single bin spike returns immediately to baseline
        if np.abs(y[i] - local_med) > thresh:
            # Check neighbors
            left_ok = (i_idx > 0) and (np.abs(y[valid_indices[i_idx - 1]] - local_med) < thresh)
            right_ok = (i_idx < len(valid_indices) - 1) and (np.abs(y[valid_indices[i_idx + 1]] - local_med) < thresh)
            if left_ok and right_ok:
                suspect_spikes[i] = True

    return suspect_spikes


def compute_standardized_series(
    y: np.ndarray,
    baseline: np.ndarray,
    noise_sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute standardized residuals and asinh transformation:
      z_i = (y_i - b_i) / sigma_i
      x_i = asinh(z_i)
    
    asinh behaves logarithmically for large signals but is smooth and defined for zero/negative values.
    """
    sigma = max(noise_sigma, 1e-8)
    z = (y - baseline) / sigma
    x = np.arcsinh(z)
    return z, x
