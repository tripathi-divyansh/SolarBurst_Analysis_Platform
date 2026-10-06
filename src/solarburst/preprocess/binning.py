"""
Cadence regularization and detection series creation.
Maintains original scientific series and exposure-aware detection series.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from astropy.time import Time, TimeDelta
from ..core.schema import LightCurve
from ..core.quality import QualityFlag


def regularize_detection_series(
    lc: LightCurve,
    target_dt_s: Optional[float] = None
) -> Dict[str, Any]:
    """
    Construct a detection series sampled on a regular grid without fabricating measurements in gaps.
    
    Returns:
    --------
    dict with:
      - 'time_grid_met': regular 1D MET array
      - 'value_grid': binned count rate (NaN where unobserved)
      - 'error_grid': propagated uncertainty
      - 'exposure_grid': total valid exposure in each regular bin
      - 'mask_valid': boolean mask of bins with valid observation
      - 'target_dt': bin size in seconds
    """
    met = lc.met_seconds
    valid = lc.valid_mask
    val = lc.value
    err = lc.error
    exp = lc.exposure_s

    # Determine target cadence: median of consecutive valid intervals
    if target_dt_s is None:
        if len(met) > 1:
            diffs = np.diff(met[valid]) if np.sum(valid) > 1 else np.diff(met)
            diffs = diffs[diffs > 0]
            target_dt_s = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        else:
            target_dt_s = 1.0
    target_dt_s = max(target_dt_s, 0.1)

    t_start = met[0]
    t_end = met[-1]
    n_bins = int(np.ceil((t_end - t_start) / target_dt_s)) + 1
    
    grid_edges = t_start + np.arange(n_bins + 1) * target_dt_s
    grid_centers = 0.5 * (grid_edges[:-1] + grid_edges[1:])

    # Assign points to regular grid bins
    bin_indices = np.digitize(met, grid_edges) - 1
    in_range = (bin_indices >= 0) & (bin_indices < n_bins)

    grid_val = np.full(n_bins, np.nan, dtype=np.float64)
    grid_err = np.full(n_bins, np.nan, dtype=np.float64)
    grid_exp = np.zeros(n_bins, dtype=np.float64)
    grid_valid = np.zeros(n_bins, dtype=bool)

    # Accumulate valid points
    for i in np.where(in_range & valid)[0]:
        b = bin_indices[i]
        w = max(exp[i], 1e-6)
        if np.isnan(grid_val[b]):
            grid_val[b] = val[i] * w
            grid_err[b] = (err[i] * w)**2 if np.isfinite(err[i]) else (val[i] * w)
            grid_exp[b] = w
            grid_valid[b] = True
        else:
            grid_val[b] += val[i] * w
            if np.isfinite(err[i]):
                grid_err[b] += (err[i] * w)**2
            else:
                grid_err[b] += (val[i] * w)
            grid_exp[b] += w

    # Normalize by valid accumulated exposure
    has_exp = grid_exp > 0
    grid_val[has_exp] /= grid_exp[has_exp]
    grid_err[has_exp] = np.sqrt(grid_err[has_exp]) / grid_exp[has_exp]

    return {
        "time_grid_met": grid_centers,
        "value_grid": grid_val,
        "error_grid": grid_err,
        "exposure_grid": grid_exp,
        "mask_valid": grid_valid,
        "target_dt_s": target_dt_s
    }
