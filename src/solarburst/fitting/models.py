"""
Physical flare emission models: Piecewise Gaussian-rise/exponential-decay and EMG convolution.
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy.stats import exponnorm


def piecewise_gauss_exp(
    t: np.ndarray,
    amplitude: float,
    t_peak: float,
    sigma_rise: float,
    tau_decay: float,
    b0: float,
    b1: float,
    t_ref: float = 0.0
) -> np.ndarray:
    """
    Model A: Continuous piecewise Gaussian rise and exponential decay plus linear background.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    sig = np.zeros_like(t_arr)

    # Rise phase (t <= t_peak)
    rise_idx = t_arr <= t_peak
    sig[rise_idx] = amplitude * np.exp(-0.5 * ((t_arr[rise_idx] - t_peak) / max(sigma_rise, 1e-4))**2)

    # Decay phase (t > t_peak)
    decay_idx = t_arr > t_peak
    sig[decay_idx] = amplitude * np.exp(-(t_arr[decay_idx] - t_peak) / max(tau_decay, 1e-4))

    # Baseline
    baseline = b0 + b1 * (t_arr - t_ref)
    return baseline + sig


def emg_convolution_bin_averaged(
    lo: np.ndarray,
    hi: np.ndarray,
    fluence: float,
    mu: float,
    sigma_h: float,
    tau_c: float,
    b0: float,
    b1: float,
    t_ref: float = 0.0
) -> np.ndarray:
    """
    Model B: Bin-averaged exponentially modified Gaussian (EMG) convolution.
    Heating: Gaussian centered at mu with width sigma_h.
    Cooling: Exponential decay response with timescale tau_c.
    """
    mid = 0.5 * (lo + hi)
    dt = hi - lo
    dt = np.maximum(dt, 1e-6)

    # Scipy exponnorm parameterization:
    # scale = sigma_h, K = tau_c / sigma_h, loc = mu
    K = max(tau_c / max(sigma_h, 1e-6), 1e-4)
    dist = exponnorm(K=K, loc=mu, scale=max(sigma_h, 1e-6))

    # Bin-averaged signal using CDF difference
    cdf_hi = dist.cdf(hi)
    cdf_lo = dist.cdf(lo)
    signal = fluence * (cdf_hi - cdf_lo) / dt

    baseline = b0 + b1 * (mid - t_ref)
    return baseline + signal
