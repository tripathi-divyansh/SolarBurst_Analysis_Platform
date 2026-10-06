"""
Bounded least squares fitting of bin-averaged EMG heating-cooling convolution profile.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import exponnorm


def fit_emg_profile(
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    err: np.ndarray
) -> Dict[str, Any]:
    """
    Fit bin-averaged EMG profile jointly with linear background.
    lo, hi: bin edges in seconds relative to observation start/epoch.
    y, err: valid observations in physical units.
    """
    lo, hi, y, err = map(np.asarray, (lo, hi, y, err))
    good = (
        np.isfinite(lo) & np.isfinite(hi) &
        np.isfinite(y) & np.isfinite(err) &
        (hi > lo) & (err > 0)
    )
    lo, hi, y, err = [a[good] for a in (lo, hi, y, err)]
    if len(y) < 10:
        raise ValueError("Insufficient valid bins for a six-parameter profile fit")

    mid = (lo + hi) / 2.0
    tref = float(mid.mean())
    lo_rel = lo - tref
    hi_rel = hi - tref
    mid_rel = mid - tref
    dt = float(np.median(hi - lo))
    span = float(hi_rel.max() - lo_rel.min())

    scale = max(float(np.ptp(y)), float(np.median(err)), 1e-6)
    ys = y / scale
    es = err / scale
    edge = max(3, len(y) // 10)
    b0 = float(np.median(np.r_[ys[:edge], ys[-edge:]]))
    noise = float(np.median(es))
    peak_idx = int(np.argmax(ys))
    peak = float(mid_rel[peak_idx])
    area = max(float(np.sum(np.maximum(ys - b0, 0.0) * (hi_rel - lo_rel))), noise * dt)

    # Parameter vector q:
    # q[0]: log(fluence)
    # q[1]: mu (heating center relative to tref)
    # q[2]: log(sigma_h) (heating duration)
    # q[3]: log(tau_c) (effective cooling timescale)
    # q[4]: b0 (baseline at tref)
    # q[5]: b1 (baseline slope)

    def prediction(q):
        fluence = np.exp(q[0])
        mu = q[1]
        sigma = np.exp(q[2])
        tau = np.exp(q[3])
        K = max(tau / max(sigma, 1e-6), 1e-4)
        dist = exponnorm(K=K, loc=mu, scale=max(sigma, 1e-6))
        signal = fluence * (dist.cdf(hi_rel) - dist.cdf(lo_rel)) / (hi_rel - lo_rel)
        return q[4] + q[5] * mid_rel + signal

    lower = [
        np.log(area * 1e-6), float(lo_rel.min()), np.log(max(dt / 4.0, 1e-3)),
        np.log(max(dt / 4.0, 1e-3)), b0 - 8.0 * noise, -np.inf
    ]
    upper = [
        np.log(area * 1e4), float(hi_rel.max()), np.log(span * 2.0),
        np.log(span * 20.0), b0 + 8.0 * noise, np.inf
    ]

    candidates = []
    # Multi-start initial guesses across tau timescales
    for tau0 in (span / 20.0, span / 5.0, span * 0.8):
        sigma0 = max(dt, span / 30.0)
        q0 = [
            np.log(area),
            np.clip(peak - sigma0, float(lo_rel.min()), float(hi_rel.max())),
            np.log(sigma0),
            np.log(max(dt, tau0)),
            b0,
            0.0
        ]
        try:
            res = least_squares(
                lambda q: (prediction(q) - ys) / es,
                q0,
                bounds=(lower, upper),
                method="trf",
                loss="linear",
                x_scale="jac",
                max_nfev=2500
            )
            if res.success and np.all(np.isfinite(res.fun)):
                candidates.append(res)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("Profile fit failed to converge")

    best = min(candidates, key=lambda r: np.sum(r.fun**2))
    q = best.x

    # Unscale fitted arrays and parameters
    fit_ys = prediction(q) * scale
    residuals = y - fit_ys
    n_params = len(q)
    dof = max(len(y) - n_params, 1)
    chi2_val = float(np.sum(best.fun**2))
    chi2_reduced = float(chi2_val / dof)

    # Baseline component
    baseline_fit = (q[4] + q[5] * mid_rel) * scale
    signal_fit = fit_ys - baseline_fit

    # AICc and BIC
    n_pts = len(y)
    rss = max(float(np.sum(residuals**2)), 1e-12)
    aic = n_pts * np.log(rss / n_pts) + 2 * n_params
    aicc = aic + (2 * n_params * (n_params + 1)) / max(n_pts - n_params - 1, 1)
    bic = n_pts * np.log(rss / n_pts) + n_params * np.log(n_pts)

    # R-squared
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r_squared = float(1.0 - (rss / max(ss_tot, 1e-12)))

    # Parameter uncertainties via Jacobian covariance
    # Cov = inv(J^T J) * chi2_reduced
    uncertainties = {}
    try:
        J = best.jac
        JtJ = J.T @ J
        cov = np.linalg.pinv(JtJ) * max(chi2_reduced, 1.0)
        param_stderrs = np.sqrt(np.maximum(np.diag(cov), 0.0))
        # Log parameter delta method: sigma_param = param * sigma_log_param
        fluence_val = float(np.exp(q[0]) * scale)
        uncertainties["fluence_err"] = float(fluence_val * param_stderrs[0])
        uncertainties["mu_err"] = float(param_stderrs[1])
        uncertainties["sigma_h_err"] = float(np.exp(q[2]) * param_stderrs[2])
        uncertainties["tau_c_err"] = float(np.exp(q[3]) * param_stderrs[3])
        uncertainties["b0_err"] = float(param_stderrs[4] * scale)
        uncertainties["b1_err"] = float(param_stderrs[5] * scale)
    except Exception:
        uncertainties = {}

    # Exact peak time and net peak of fitted signal (EMG mode)
    # The peak of EMG occurs slightly after mu due to convolution
    grid_dense = np.linspace(float(lo_rel.min()), float(hi_rel.max()), 1000)
    K = max(np.exp(q[3]) / max(np.exp(q[2]), 1e-6), 1e-4)
    dist_dense = exponnorm(K=K, loc=q[1], scale=np.exp(q[2]))
    sig_dense = (np.exp(q[0]) * scale) * dist_dense.pdf(grid_dense)
    peak_dense_idx = int(np.argmax(sig_dense))
    
    t_peak_fitted = float(grid_dense[peak_dense_idx] + tref)
    net_peak_fitted = float(sig_dense[peak_dense_idx])
    b_at_peak = float(q[4] * scale + q[5] * scale * (t_peak_fitted - tref))
    total_peak_fitted = net_peak_fitted + b_at_peak

    return {
        "fluence": float(np.exp(q[0]) * scale),
        "heating_center_met": float(q[1] + tref),
        "heating_sigma_s": float(np.exp(q[2])),
        "decay_tau_s": float(np.exp(q[3])),
        "background_at_tref": float(q[4] * scale),
        "background_slope": float(q[5] * scale),
        "tref_met": tref,
        "fitted_peak_time_met": t_peak_fitted,
        "net_peak": net_peak_fitted,
        "total_peak": total_peak_fitted,
        "fitted_series": fit_ys,
        "baseline_series": baseline_fit,
        "burst_component_series": signal_fit,
        "residuals": residuals,
        "chi2_reduced": chi2_reduced,
        "aicc": float(aicc),
        "bic": float(bic),
        "r_squared": float(r_squared),
        "uncertainties": uncertainties,
        "optimizer_success": best.success,
    }
