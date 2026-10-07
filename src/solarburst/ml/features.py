"""
Standardized feature extraction and schema for candidate burst classification.
Shared identically across training and inference.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from ..core.schema import Candidate, LightCurve
from ..detection.matched_filter import run_matched_filter_bank
from ..detection.wavelet_cwt import run_cwt_candidate_detector


# Exact ordered feature names (24 features)
FEATURE_NAMES = [
    # Signal strength
    "peak_snr",
    "net_peak",
    "integrated_excess",
    # Duration and shape
    "prelim_duration_s",
    "prelim_width_s",
    "rise_slope",
    "decay_slope",
    "rise_decay_ratio",
    # Multiscale evidence
    "mf_max_score",
    "mf_best_rise_s",
    "mf_best_decay_s",
    "cwt_max_score",
    "cwt_best_scale_s",
    # Background and noise
    "local_noise",
    "local_baseline_level",
    "local_baseline_slope",
    "pre_post_level_diff",
    # Complexity
    "num_peaks",
    "peak_prominence_ratio",
    # Quality and sampling
    "exposure_fraction",
    "gap_fraction",
    "edge_distance_s",
    "cadence_s",
    "valid_sample_count"
]


def extract_candidate_features(
    cand: Candidate,
    time_met: np.ndarray,
    rate: np.ndarray,
    baseline: np.ndarray,
    noise_sigma: float,
    dt_s: float,
    mf_global: Any = None,
    cwt_global: Any = None
) -> Dict[str, float]:
    """
    Extract a single row of numeric features for a candidate within its context window.
    """
    # Context slice
    mask_ctx = (time_met >= cand.context_start_met) & (time_met <= cand.context_end_met)
    t_ctx = time_met[mask_ctx]
    r_ctx = rate[mask_ctx]
    b_ctx = baseline[mask_ctx]
    
    if len(t_ctx) < 5:
        # Unsupported, return NaNs
        return {k: np.nan for k in FEATURE_NAMES}

    # Burst core slice
    mask_burst = (time_met >= cand.start_time_met) & (time_met <= cand.end_time_met)
    t_b = time_met[mask_burst]
    r_b = rate[mask_burst]
    b_b = baseline[mask_burst]
    
    if len(r_b) == 0:
        r_b = r_ctx
        b_b = b_ctx
        t_b = t_ctx

    # 1. Signal strength
    peak_idx = np.argmax(r_b)
    peak_val = float(r_b[peak_idx])
    b_at_peak = float(b_b[peak_idx])
    net_peak = float(max(0.0, peak_val - b_at_peak))
    peak_snr = float(net_peak / max(noise_sigma, 1e-6))
    integrated_excess = float(np.sum(np.maximum(r_b - b_b, 0.0) * dt_s))

    # 2. Duration and shape
    dur_s = float(cand.end_time_met - cand.start_time_met)
    peak_t = float(t_b[peak_idx])
    rise_dur_s = max(peak_t - cand.start_time_met, dt_s)
    decay_dur_s = max(cand.end_time_met - peak_t, dt_s)
    
    rise_slope = float(net_peak / rise_dur_s)
    decay_slope = float(net_peak / decay_dur_s)
    rise_decay_ratio = float(rise_dur_s / decay_dur_s)

    # 3. Multiscale evidence in context (O(1) slice from global precomputation)
    if mf_global is not None and "max_score" in mf_global and len(mf_global["max_score"]) == len(time_met):
        mf_max = float(np.nanmax(mf_global["max_score"][mask_ctx])) if np.any(mask_ctx) else 0.0
        mf_best_r = float(np.median(mf_global["best_scale_rise"][mask_ctx])) if np.any(mask_ctx) else 16.0
        mf_best_d = float(np.median(mf_global["best_scale_decay"][mask_ctx])) if np.any(mask_ctx) else 64.0
    else:
        z_ctx = (r_ctx - b_ctx) / max(noise_sigma, 1e-6)
        mf = run_matched_filter_bank(z_ctx, dt_s, threshold=1.0)
        mf_max = float(np.nanmax(mf["max_score"])) if len(mf["max_score"]) > 0 else 0.0
        mf_best_r = float(np.median(mf["best_scale_rise"])) if len(mf["best_scale_rise"]) > 0 else 16.0
        mf_best_d = float(np.median(mf["best_scale_decay"])) if len(mf["best_scale_decay"]) > 0 else 64.0

    if cwt_global is not None and "max_cwt_score" in cwt_global and len(cwt_global["max_cwt_score"]) == len(time_met):
        cwt_max = float(np.nanmax(cwt_global["max_cwt_score"][mask_ctx])) if np.any(mask_ctx) else 0.0
        cwt_best_scale = float(np.median(cwt_global["best_scale_s"][mask_ctx])) if np.any(mask_ctx) else 32.0
    else:
        z_ctx = (r_ctx - b_ctx) / max(noise_sigma, 1e-6)
        cwt = run_cwt_candidate_detector(z_ctx, dt_s, threshold=1.0)
        cwt_max = float(np.nanmax(cwt["max_cwt_score"])) if len(cwt["max_cwt_score"]) > 0 else 0.0
        cwt_best_scale = float(np.median(cwt["best_scale_s"])) if len(cwt["best_scale_s"]) > 0 else 32.0
    
    # 4. Background and local slopes
    # Fit simple line to baseline in context
    p_b = np.polyfit(t_ctx - t_ctx[0], b_ctx, 1)
    b_slope = float(p_b[0])
    
    # Pre- and post-burst levels
    pre_mask = t_ctx < cand.start_time_met
    post_mask = t_ctx > cand.end_time_met
    pre_lvl = float(np.mean(r_ctx[pre_mask])) if np.any(pre_mask) else b_at_peak
    post_lvl = float(np.mean(r_ctx[post_mask])) if np.any(post_mask) else b_at_peak
    pre_post_diff = float(np.abs(post_lvl - pre_lvl) / max(noise_sigma, 1e-6))

    # 5. Complexity: number of peaks in burst core
    peaks_in_core = 0
    for i in range(1, len(r_b) - 1):
        if r_b[i] > r_b[i - 1] and r_b[i] > r_b[i + 1] and (r_b[i] - b_b[i]) > 2 * noise_sigma:
            peaks_in_core += 1
    num_peaks = max(1, peaks_in_core)

    # 6. Quality and sampling
    n_expected = int(np.round((cand.end_time_met - cand.start_time_met) / max(dt_s, 0.1))) + 1
    valid_sample_count = len(r_b)
    exp_frac = float(np.clip(valid_sample_count / max(n_expected, 1), 0.0, 1.0))
    gap_frac = float(max(0.0, 1.0 - exp_frac))
    edge_dist = float(min(cand.start_time_met - time_met[0], time_met[-1] - cand.end_time_met))

    features = {
        "peak_snr": peak_snr,
        "net_peak": net_peak,
        "integrated_excess": integrated_excess,
        "prelim_duration_s": dur_s,
        "prelim_width_s": float(cand.preliminary_width_s),
        "rise_slope": rise_slope,
        "decay_slope": decay_slope,
        "rise_decay_ratio": rise_decay_ratio,
        "mf_max_score": mf_max,
        "mf_best_rise_s": mf_best_r,
        "mf_best_decay_s": mf_best_d,
        "cwt_max_score": cwt_max,
        "cwt_best_scale_s": cwt_best_scale,
        "local_noise": float(noise_sigma),
        "local_baseline_level": b_at_peak,
        "local_baseline_slope": b_slope,
        "pre_post_level_diff": pre_post_diff,
        "num_peaks": float(num_peaks),
        "peak_prominence_ratio": float(net_peak / max(np.ptp(r_b), 1e-6)),
        "exposure_fraction": exp_frac,
        "gap_fraction": gap_frac,
        "edge_distance_s": edge_dist,
        "cadence_s": float(dt_s),
        "valid_sample_count": float(valid_sample_count)
    }

    # Ensure all declared features exist
    for k in FEATURE_NAMES:
        if k not in features or not np.isfinite(features[k]):
            features[k] = np.nan

    return features


def extract_features_table(
    candidates: List[Candidate],
    time_met: np.ndarray,
    rate: np.ndarray,
    baseline: np.ndarray,
    noise_sigma: float,
    dt_s: float
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract 2D feature matrix (n_candidates x n_features) and update candidate.features in-place.
    Precomputes multiscale evidence globally once across the series for instant evaluation.
    """
    z_global = (rate - baseline) / max(noise_sigma, 1e-6)
    mf_global = run_matched_filter_bank(z_global, dt_s, threshold=1.0)
    cwt_global = run_cwt_candidate_detector(z_global, dt_s, threshold=1.0)

    rows = []
    for cand in candidates:
        feat_dict = extract_candidate_features(
            cand, time_met, rate, baseline, noise_sigma, dt_s,
            mf_global=mf_global, cwt_global=cwt_global
        )
        cand.features = feat_dict
        row = [feat_dict[k] for k in FEATURE_NAMES]
        rows.append(row)

    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), FEATURE_NAMES

    return np.asarray(rows, dtype=np.float64), FEATURE_NAMES
