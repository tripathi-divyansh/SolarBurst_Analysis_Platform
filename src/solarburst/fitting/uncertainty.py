"""
Uncertainty estimation: Jacobian covariance propagation, bootstrap resampling, and defined boundary crossings.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


def compute_profile_crossings(
    t_dense: np.ndarray,
    signal_dense: np.ndarray,
    t_peak: float,
    net_peak: float
) -> Dict[str, float]:
    """
    Compute defined boundary crossings on dense fitted profile:
      - 5% crossings (official start and end)
      - 10% crossings (for morphology asymmetry ratio rho)
      - 50% crossings (for FWHM)
    """
    t_arr = np.asarray(t_dense)
    sig_arr = np.asarray(signal_dense)

    # 1. 5% threshold
    thresh_5 = 0.05 * net_peak
    above_5 = sig_arr >= thresh_5
    
    # Pre-peak 5% crossing
    pre_peak = t_arr <= t_peak
    pre_5_idx = np.where(pre_peak & above_5)[0]
    t_start_5 = float(t_arr[pre_5_idx[0]]) if len(pre_5_idx) > 0 else float(t_arr[0])

    # Post-peak 5% crossing
    post_peak = t_arr >= t_peak
    post_5_idx = np.where(post_peak & above_5)[0]
    t_end_5 = float(t_arr[post_5_idx[-1]]) if len(post_5_idx) > 0 else float(t_arr[-1])

    # 2. 10% threshold for asymmetry ratio rho
    thresh_10 = 0.10 * net_peak
    above_10 = sig_arr >= thresh_10
    pre_10_idx = np.where(pre_peak & above_10)[0]
    t_rise_10 = float(t_arr[pre_10_idx[0]]) if len(pre_10_idx) > 0 else t_start_5

    post_10_idx = np.where(post_peak & above_10)[0]
    t_decay_10 = float(t_arr[post_10_idx[-1]]) if len(post_10_idx) > 0 else t_end_5

    denom = max(t_decay_10 - t_peak, 0.1)
    asymmetry_rho = float((t_peak - t_rise_10) / denom)

    # 3. 50% threshold for FWHM
    thresh_50 = 0.50 * net_peak
    above_50 = sig_arr >= thresh_50
    pre_50_idx = np.where(pre_peak & above_50)[0]
    t_half_1 = float(t_arr[pre_50_idx[0]]) if len(pre_50_idx) > 0 else t_start_5

    post_50_idx = np.where(post_peak & above_50)[0]
    t_half_2 = float(t_arr[post_50_idx[-1]]) if len(post_50_idx) > 0 else t_end_5
    fwhm_s = float(max(t_half_2 - t_half_1, 0.1))

    rise_time_s = float(max(t_peak - t_start_5, 0.1))
    decay_duration_s = float(max(t_end_5 - t_peak, 0.1))
    duration_s = float(max(t_end_5 - t_start_5, 0.2))

    return {
        "start_time_met": t_start_5,
        "end_time_met": t_end_5,
        "rise_time_s": rise_time_s,
        "decay_duration_s": decay_duration_s,
        "duration_s": duration_s,
        "fwhm_s": fwhm_s,
        "asymmetry_rho": asymmetry_rho,
        "t_rise_10_met": t_rise_10,
        "t_decay_10_met": t_decay_10
    }


def run_bootstrap_uncertainties(
    lo: np.ndarray,
    hi: np.ndarray,
    fit_result: Dict[str, Any],
    n_bootstrap: int = 50,
    rng_seed: int = 42
) -> Dict[str, Any]:
    """
    Run parametric/residual bootstrap realizations to construct 68% and 95% confidence intervals.
    """
    from .emg_fit import fit_emg_profile
    fitted = fit_result["fitted_series"]
    residuals = fit_result["residuals"]
    n_pts = len(fitted)
    
    if n_pts < 10 or n_bootstrap < 5:
        return {}

    rng = np.random.default_rng(rng_seed)
    bootstrap_peaks = []
    bootstrap_fluences = []
    bootstrap_taus = []

    err_pseudo = np.full(n_pts, max(float(np.std(residuals)), 1e-4))

    for b in range(n_bootstrap):
        # Block or iid residual resampling
        resamp_resid = rng.choice(residuals, size=n_pts, replace=True)
        y_pseudo = fitted + resamp_resid
        try:
            res_b = fit_emg_profile(lo, hi, y_pseudo, err_pseudo)
            bootstrap_peaks.append(res_b["net_peak"])
            bootstrap_fluences.append(res_b["fluence"])
            bootstrap_taus.append(res_b["decay_tau_s"])
        except Exception:
            continue

    if len(bootstrap_peaks) < 5:
        return {}

    p_arr = np.asarray(bootstrap_peaks)
    f_arr = np.asarray(bootstrap_fluences)
    t_arr = np.asarray(bootstrap_taus)

    return {
        "n_successful_realizations": len(p_arr),
        "net_peak_ci_68": [float(np.percentile(p_arr, 16)), float(np.percentile(p_arr, 84))],
        "net_peak_ci_95": [float(np.percentile(p_arr, 2.5)), float(np.percentile(p_arr, 97.5))],
        "fluence_ci_68": [float(np.percentile(f_arr, 16)), float(np.percentile(f_arr, 84))],
        "fluence_ci_95": [float(np.percentile(f_arr, 2.5)), float(np.percentile(f_arr, 97.5))],
        "decay_tau_ci_68": [float(np.percentile(t_arr, 16)), float(np.percentile(t_arr, 84))],
        "decay_tau_ci_95": [float(np.percentile(t_arr, 2.5)), float(np.percentile(t_arr, 97.5))],
    }
