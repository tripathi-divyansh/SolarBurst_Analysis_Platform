"""
Matched filter candidate generator using Gaussian-rise / exponential-decay template banks.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.signal import fftconvolve


def generate_template(rise_s: float, decay_s: float, dt_s: float) -> np.ndarray:
    """
    Generate discrete normalized Gaussian-rise / exponential-decay template kernel.
    """
    dt_s = max(dt_s, 0.1)
    # Span: 3 sigmas on rise, 4 e-folds on decay
    n_rise = max(3, int(np.ceil(3.0 * rise_s / dt_s)))
    n_decay = max(4, int(np.ceil(4.0 * decay_s / dt_s)))

    t_rise = -np.arange(n_rise, 0, -1) * dt_s
    t_decay = np.arange(n_decay + 1) * dt_s

    k_rise = np.exp(-0.5 * (t_rise / max(rise_s, 1e-4))**2)
    k_decay = np.exp(-t_decay / max(decay_s, 1e-4))

    kernel = np.concatenate([k_rise, k_decay])
    # Zero-mean template for robustness against local offset
    kernel = kernel - np.mean(kernel)
    norm = np.sqrt(np.sum(kernel**2))
    if norm > 1e-9:
        kernel /= norm
    return kernel


def run_matched_filter_bank(
    z_series: np.ndarray,
    dt_s: float,
    threshold: float = 3.5
) -> Dict[str, Any]:
    """
    Run bank of matched filters across standardized z_series.
    
    Timescales:
    Rise widths: 4, 16, 64, 256 s.
    Decays: 16, 64, 256, 1024, 4096 s.
    
    Returns:
    --------
    dict with:
      - 'max_score': 1D array of maximum template SNR across the bank
      - 'best_scale_rise': 1D array of best matching rise width
      - 'best_scale_decay': 1D array of best matching decay time
      - 'candidate_indices': peak indices exceeding threshold
    """
    n = len(z_series)
    max_scores = np.zeros(n, dtype=np.float64)
    best_rise = np.zeros(n, dtype=np.float64)
    best_decay = np.zeros(n, dtype=np.float64)

    rise_scales = [4.0, 16.0, 64.0, 256.0]
    decay_scales = [16.0, 64.0, 256.0, 1024.0, 4096.0]

    # Clean NaNs in z_series for convolution
    z_clean = np.nan_to_num(z_series, nan=0.0, posinf=0.0, neginf=0.0)

    for r in rise_scales:
        for d in decay_scales:
            kernel = generate_template(r, d, dt_s)
            if len(kernel) >= n:
                continue
            # Convolve: kernel reversed for matched filter correlation
            corr = fftconvolve(z_clean, kernel[::-1], mode="same")
            higher = corr > max_scores
            max_scores[higher] = corr[higher]
            best_rise[higher] = r
            best_decay[higher] = d

    # Find local peaks above threshold
    candidate_indices = []
    for i in range(1, n - 1):
        if max_scores[i] >= threshold and max_scores[i] >= max_scores[i - 1] and max_scores[i] >= max_scores[i + 1]:
            candidate_indices.append(i)

    return {
        "max_score": max_scores,
        "best_scale_rise": best_rise,
        "best_scale_decay": best_decay,
        "candidate_indices": candidate_indices
    }
