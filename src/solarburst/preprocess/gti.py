"""
Good Time Interval (GTI) processing, gap splitting, and contiguous segment identification.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from ..core.schema import LightCurve
from ..core.quality import QualityFlag


def find_contiguous_segments(
    time_sec: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_s: float = 30.0
) -> List[Tuple[int, int]]:
    """
    Split time series into continuous valid observation segments.
    A segment breaks when time difference between consecutive points exceeds max_gap_s
    or when valid_mask is False across multiple bins.
    
    Returns:
    --------
    List of (start_idx, end_idx) tuples (inclusive slice range).
    """
    n = len(time_sec)
    if n == 0:
        return []

    segments = []
    seg_start = None

    for i in range(n):
        if valid_mask[i]:
            if seg_start is None:
                seg_start = i
            else:
                # Check for gap between current and previous point
                dt = time_sec[i] - time_sec[i - 1]
                if dt > max_gap_s:
                    # End previous segment
                    segments.append((seg_start, i - 1))
                    seg_start = i
        else:
            if seg_start is not None:
                segments.append((seg_start, i - 1))
                seg_start = None

    if seg_start is not None:
        segments.append((seg_start, n - 1))

    # Filter out segments with too few points (< 5 points)
    return [s for s in segments if (s[1] - s[0] + 1) >= 5]


def compute_coverage_summary(lc: LightCurve) -> Dict[str, Any]:
    """
    Compute total observing duration, valid exposure, gap fraction, and segment counts.
    """
    met = lc.met_seconds
    valid = lc.valid_mask
    n_total = len(met)
    n_valid = int(np.sum(valid))
    
    if n_total == 0:
        return {"total_duration_s": 0, "valid_exposure_s": 0, "gap_fraction": 1.0}

    span_s = float(met[-1] - met[0])
    valid_exp_s = float(np.sum(lc.exposure_s[valid]))
    gap_fraction = float(max(0.0, 1.0 - (valid_exp_s / max(span_s, 1.0))))
    segments = find_contiguous_segments(met, valid)

    return {
        "start_iso": lc.time[0].isot,
        "end_iso": lc.time[-1].isot,
        "span_s": span_s,
        "valid_points": n_valid,
        "total_points": n_total,
        "valid_exposure_s": valid_exp_s,
        "gap_fraction": gap_fraction,
        "contiguous_segments": len(segments),
        "segments": [{"start_idx": s[0], "end_idx": s[1], "duration_s": float(met[s[1]] - met[s[0]])} for s in segments]
    }
