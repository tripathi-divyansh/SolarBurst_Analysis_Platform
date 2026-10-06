"""
Evaluation metrics: completeness, false-alarm rate (FAR) per 24h, PR trade-offs, and parameter recovery.
"""

from typing import Dict, Any, List
import numpy as np
from ..core.schema import BurstResult


def compute_detection_metrics(
    matched_results: Dict[str, Any],
    valid_exposure_s: float
) -> Dict[str, Any]:
    """
    Compute scientific detection metrics: Precision, Recall, F1, and FAR per 24h.
    """
    n_tp = len(matched_results["matched_pairs"])
    n_fp = len(matched_results["unmatched_predictions"])
    n_fn = len(matched_results["unmatched_references"])

    precision = float(n_tp / max(n_tp + n_fp, 1))
    recall = float(n_tp / max(n_tp + n_fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-9))

    # FAR per 24h (86400 s) of valid observing time
    valid_days = max(valid_exposure_s / 86400.0, 1e-5)
    far_per_24h = float(n_fp / valid_days)

    return {
        "true_positives": n_tp,
        "false_positives": n_fp,
        "false_negatives": n_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "valid_exposure_s": float(valid_exposure_s),
        "valid_days": float(valid_days),
        "far_per_24h": far_per_24h
    }


def compute_parameter_recovery_errors(matched_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute parameter recovery bias and median absolute error (MAE).
    """
    if not matched_pairs:
        return {
            "n_matched": 0,
            "peak_time_bias_s": 0.0,
            "peak_time_mae_s": 0.0,
            "amplitude_bias_pct": 0.0,
            "amplitude_mae_pct": 0.0,
        }

    dt_peaks = []
    amp_errs_pct = []

    for pair in matched_pairs:
        p: BurstResult = pair["predicted"]
        r = pair["reference"]
        
        # Timing error
        dt = p.peak_time_met - r["peak_time_met"]
        dt_peaks.append(dt)

        # Amplitude relative error
        if "amplitude" in r and r["amplitude"] > 0:
            rel_amp = 100.0 * (p.net_peak - r["amplitude"]) / r["amplitude"]
            amp_errs_pct.append(rel_amp)

    dt_arr = np.asarray(dt_peaks)
    amp_arr = np.asarray(amp_errs_pct) if amp_errs_pct else np.array([0.0])

    return {
        "n_matched": len(matched_pairs),
        "peak_time_bias_s": float(np.mean(dt_arr)),
        "peak_time_mae_s": float(np.median(np.abs(dt_arr))),
        "peak_time_std_s": float(np.std(dt_arr)),
        "amplitude_bias_pct": float(np.mean(amp_arr)),
        "amplitude_mae_pct": float(np.median(np.abs(amp_arr))),
    }
