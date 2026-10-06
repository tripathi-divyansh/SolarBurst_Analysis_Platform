"""
Continuous Wavelet Transform (CWT) candidate generator and scalogram computation using PyWavelets / SciPy.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pywt


def ricker_wavelet(points: int, a: float) -> np.ndarray:
    """Analytical Mexican-hat (Ricker) wavelet kernel."""
    a = max(float(a), 0.1)
    vec = np.arange(0, points) - (points - 1.0) / 2.0
    xsq = (vec / a) ** 2
    mod = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    return mod * (1.0 - xsq) * np.exp(-xsq / 2.0)



def run_cwt_candidate_detector(
    z_series: np.ndarray,
    dt_s: float,
    threshold: float = 3.0,
    scales_s: List[float] = None
) -> Dict[str, Any]:
    """
    Multiscale Mexican-hat wavelet candidate detector.
    
    Parameters:
    -----------
    z_series: standardized residuals.
    dt_s: cadence in seconds.
    threshold: wavelet coefficient SNR threshold.
    scales_s: physical timescales in seconds.
    
    Returns:
    --------
    dict with:
      - 'max_cwt_score': maximum normalized CWT score across scales
      - 'best_scale_s': timescale yielding highest response
      - 'candidate_indices': detected peak positions
    """
    n = len(z_series)
    z_clean = np.nan_to_num(z_series, nan=0.0)

    if scales_s is None:
        scales_s = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0]

    # Convert physical seconds to wavelet scale parameter: scale = tau_s / dt_s
    scales = [max(1.0, s / dt_s) for s in scales_s]

    max_cwt_score = np.zeros(n, dtype=np.float64)
    best_scale_s = np.zeros(n, dtype=np.float64)

    # Compute CWT coefficients using Ricker / Mexican hat wavelet
    try:
        coefs, freqs = pywt.cwt(z_clean, scales, "mexh")
    except Exception:
        # Fallback to scipy ricker convolution
        coefs = np.zeros((len(scales), n))
        for idx, sc in enumerate(scales):
            points = min(int(10 * sc), n)
            if points % 2 == 0:
                points += 1
            kernel = ricker_wavelet(points, sc)
            from scipy.signal import fftconvolve
            coefs[idx] = fftconvolve(z_clean, kernel, mode="same")

    for idx, sc_s in enumerate(scales_s):
        # Normalize coefficient by sqrt(scale)
        c = np.abs(coefs[idx]) / np.sqrt(scales[idx])
        higher = c > max_cwt_score
        max_cwt_score[higher] = c[higher]
        best_scale_s[higher] = sc_s

    # Find peaks exceeding threshold
    candidate_indices = []
    for i in range(1, n - 1):
        if max_cwt_score[i] >= threshold and max_cwt_score[i] >= max_cwt_score[i - 1] and max_cwt_score[i] >= max_cwt_score[i + 1]:
            candidate_indices.append(i)

    return {
        "max_cwt_score": max_cwt_score,
        "best_scale_s": best_scale_s,
        "candidate_indices": candidate_indices
    }


def compute_scalogram(
    signal: np.ndarray,
    time_sec: np.ndarray,
    wavelet: str = "morlet",
    num_scales: int = 40,
    min_scale_s: float = 4.0,
    max_scale_s: float = 2048.0
) -> Dict[str, Any]:
    """
    Compute time-frequency / time-scale scalogram for interactive GUI visualization.
    Includes cone of influence / edge influence boundary.
    """
    n = len(signal)
    if n == 0:
        return {"scales": [], "power": [], "times": []}

    dt = np.median(np.diff(time_sec)) if n > 1 else 1.0
    dt = max(dt, 0.1)

    sig_clean = np.nan_to_num(signal - np.nanmedian(signal), nan=0.0)

    # Logarithmically spaced scales
    scales_s = np.geomspace(max(min_scale_s, 2 * dt), min(max_scale_s, (time_sec[-1] - time_sec[0]) / 2), num_scales)
    scales = scales_s / dt

    wav_name = "morl" if "morlet" in wavelet.lower() else "mexh"
    coefs, freqs = pywt.cwt(sig_clean, scales, wav_name)
    power = (np.abs(coefs))**2

    # Cone of influence (COI) in seconds from each edge: e-folding distance ~ sqrt(2) * scale
    coi = np.minimum(time_sec - time_sec[0], time_sec[-1] - time_sec) / np.sqrt(2.0)

    return {
        "times": time_sec.tolist(),
        "scales_s": scales_s.tolist(),
        "power": power.tolist(),
        "coi_s": coi.tolist(),
        "wavelet": wavelet
    }
