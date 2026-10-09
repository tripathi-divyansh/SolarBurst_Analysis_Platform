import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

FEATURE_VERSION = "fred-features-2"
FEATURES = ["peak_snr", "net_peak_observed", "background_at_peak", "peak_background_ratio",
            "rise_observed_s", "decay_observed_s", "duration_observed_s", "fwhm_observed_s",
            "fluence_observed", "max_rise_rate", "max_decay_rate", "skewness", "kurtosis",
            "wavelet_energy", "wavelet_max", "dominant_scale_s", "n_peaks", "cadence_s",
            "sigma_evidence", "wavelet_evidence", "bb_evidence", "edge_candidate",
            "fit_success", "fit_r2", "fit_rmse_over_noise", "fit_reduced_statistic",
            "fit_rise_constant_s", "fit_decay_constant_s", "fit_duration_s", "fit_net_peak",
            "fit_residual_autocorrelation", "fit_censored"]
# Optional regression explicitly excludes fitted target quantities and diagnostics.
# It remains a correlated cross-check because observations/features are shared.
REGRESSION_FEATURES = FEATURES[:22]
REGRESSION_TARGETS = ["target_rise_s", "target_decay_constant_s", "target_duration_s", "target_net_peak"]


def candidate_features(t, width, y, base, noise, candidate, evidence, fit):
    lo, hi, peak = (candidate[x] for x in ("lo", "hi", "peak"))
    ids = np.arange(lo, hi+1)
    signal = y-base
    local = signal[ids]
    dt = float(np.median(np.diff(t)))
    derivative = np.gradient(local, t[ids]) if len(ids) > 1 else np.array([np.nan])
    finite_der = derivative[np.isfinite(derivative)]
    half = local >= max(signal[peak], 0)/2
    half_ids = ids[half]
    npk = len(find_peaks(local, prominence=max(float(np.median(noise[ids])*2), 1e-12))[0])
    nonconstant = len(local) >= 4 and np.std(local) > 1e-9*max(1, float(np.max(np.abs(local))))
    wave = evidence["wavelet"][ids]
    vals = {
        "peak_snr": evidence["z"][peak], "net_peak_observed": signal[peak],
        "background_at_peak": base[peak],
        "peak_background_ratio": y[peak]/base[peak] if base[peak] > 3*noise[peak] else np.nan,
        "rise_observed_s": t[peak]-(t[lo]-width[lo]/2),
        "decay_observed_s": t[hi]+width[hi]/2-t[peak],
        "duration_observed_s": t[hi]+width[hi]/2-t[lo]+width[lo]/2,
        "fwhm_observed_s": t[half_ids[-1]]+width[half_ids[-1]]/2-t[half_ids[0]]+width[half_ids[0]]/2 if len(half_ids) else np.nan,
        "fluence_observed": np.sum(local*width[ids]),
        "max_rise_rate": np.max(finite_der) if len(finite_der) else np.nan,
        "max_decay_rate": -np.min(finite_der) if len(finite_der) else np.nan,
        "skewness": skew(local, bias=False) if nonconstant else np.nan,
        "kurtosis": kurtosis(local, bias=False) if nonconstant else np.nan,
        "wavelet_energy": np.mean(wave**2), "wavelet_max": np.max(wave),
        "dominant_scale_s": evidence["dominant_scale"][ids[np.argmax(wave)]],
        "n_peaks": max(1, npk), "cadence_s": dt,
        "edge_candidate": float(lo == 0 or hi == len(t)-1),
    }
    vals.update({key: float(candidate[key]) for key in ("sigma_evidence", "wavelet_evidence", "bb_evidence")})
    vals["fit_success"] = float(fit["success"])
    mapping = {"fit_r2": "r2", "fit_reduced_statistic": "reduced_statistic",
               "fit_rise_constant_s": "rise_constant_s", "fit_decay_constant_s": "decay_constant_s",
               "fit_duration_s": "duration_s", "fit_net_peak": "net_peak",
               "fit_residual_autocorrelation": "residual_autocorrelation"}
    physical = {"fit_rise_constant_s", "fit_decay_constant_s", "fit_duration_s", "fit_net_peak"}
    vals.update({key: (np.nan if key in physical and not fit.get("parameters_reliable", False)
                       else fit.get(value, np.nan)) for key, value in mapping.items()})
    vals["fit_rmse_over_noise"] = fit.get("rmse", np.nan)/np.median(noise[ids])
    vals["fit_censored"] = float(any("censored" in f for f in fit["flags"])) if fit["success"] else np.nan
    return {key: float(vals[key]) if np.isfinite(vals[key]) else np.nan for key in FEATURES}


def operational_morphology(fit, n_peaks):
    """Transparent fallback labels, not learned class probabilities."""
    if n_peaks > 1:
        return "complex_or_multi_peaked"
    if not fit["success"] or not fit.get("parameters_reliable", False) or any(f in fit["flags"] for f in ("left_censored", "right_censored", "poor_likelihood_fit")):
        return "unresolved"
    ratio = fit["rise_duration_s"]/fit["decay_duration_s"]
    return "fast_rise_slow_decay" if ratio < 0.5 else "approximately_symmetric" if ratio <= 2 else "slow_rise_fast_decay"
