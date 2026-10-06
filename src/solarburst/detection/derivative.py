"""
Smoothed derivative and sustained excess candidate generator for fast onset detection.
"""

from typing import Dict, Any, List
import numpy as np
from scipy.signal import savgol_filter


def run_derivative_excess_detector(
    z_series: np.ndarray,
    dt_s: float,
    derivative_threshold: float = 2.5,
    excess_threshold: float = 2.0,
    window_bins: int = 9
) -> Dict[str, Any]:
    """
    Detect impulsive onset bursts via smoothed positive derivative followed by sustained excess.
    """
    n = len(z_series)
    z_clean = np.nan_to_num(z_series, nan=0.0)

    # Ensure window_bins is odd and <= n
    win = min(window_bins if window_bins % 2 == 1 else window_bins + 1, n if n % 2 == 1 else n - 1)
    if win < 5:
        return {"onset_scores": np.zeros(n), "candidate_indices": []}

    # Savitzky-Golay first derivative
    deriv = savgol_filter(z_clean, window_length=win, polyorder=2, deriv=1, delta=max(dt_s, 0.1))
    
    # Positive derivative (rising edge)
    pos_deriv = np.maximum(deriv, 0.0)

    # Combined score: onset sharpness * sustained forward excess
    forward_span = min(15, n)
    forward_excess = np.zeros(n)
    for i in range(n - forward_span):
        forward_excess[i] = np.mean(z_clean[i:i + forward_span])

    onset_score = pos_deriv * np.maximum(forward_excess, 0.0)

    candidate_indices = []
    for i in range(1, n - 1):
        if (
            deriv[i] >= derivative_threshold
            and forward_excess[i] >= excess_threshold
            and onset_score[i] >= onset_score[i - 1]
            and onset_score[i] >= onset_score[i + 1]
        ):
            candidate_indices.append(i)

    return {
        "derivative": deriv,
        "onset_score": onset_score,
        "candidate_indices": candidate_indices
    }
