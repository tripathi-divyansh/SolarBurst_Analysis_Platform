"""
Time utilities and XSM MET (Mission Elapsed Time) conversions.
XSM MET: seconds elapsed since 2017-01-01 00:00:00 UTC.
"""

from typing import Union
import numpy as np
from astropy.time import Time

# XSM MET Reference Epoch: 2017-01-01 00:00:00 UTC
XSM_EPOCH_ISO = "2017-01-01T00:00:00"
XSM_EPOCH_TIME = Time(XSM_EPOCH_ISO, format="isot", scale="utc")


def met_to_astropy_time(met_seconds: Union[float, np.ndarray]) -> Time:
    """Convert XSM MET seconds to astropy.time.Time in UTC."""
    met_arr = np.asarray(met_seconds, dtype=np.float64)
    # Using TimeDelta in seconds
    from astropy.time import TimeDelta
    delta = TimeDelta(met_arr, format="sec")
    return XSM_EPOCH_TIME + delta


def astropy_time_to_met(t: Time) -> np.ndarray:
    """Convert astropy.time.Time to XSM MET seconds."""
    delta = t - XSM_EPOCH_TIME
    return delta.to_value("sec")


def parse_time_array(
    raw_time: Union[np.ndarray, list],
    format_hint: str = "auto",
    reference_epoch: str = None,
    scale: str = "utc"
) -> Time:
    """
    Parse an input time array into canonical astropy.time.Time.
    
    Parameters:
    -----------
    raw_time: array-like of numeric timestamps or ISO string dates.
    format_hint: 'xsm_met', 'unix', 'mjd', 'isot', or 'auto'.
    reference_epoch: custom epoch if format_hint is 'offset'.
    scale: 'utc', 'tt', etc.
    """
    arr = np.asarray(raw_time)
    
    # Check if string dates
    if arr.dtype.kind in ('U', 'S', 'O') and isinstance(arr[0], str):
        try:
            return Time(arr, format="isot", scale=scale)
        except Exception:
            return Time(arr, scale=scale)
            
    # Numeric arrays
    num_arr = arr.astype(np.float64)
    
    if format_hint == "xsm_met":
        return met_to_astropy_time(num_arr)
    elif format_hint == "unix":
        return Time(num_arr, format="unix", scale=scale)
    elif format_hint == "mjd":
        return Time(num_arr, format="mjd", scale=scale)
    elif format_hint == "offset" and reference_epoch:
        ref = Time(reference_epoch, scale=scale)
        from astropy.time import TimeDelta
        return ref + TimeDelta(num_arr, format="sec")
    else:
        # Auto-heuristic:
        # MET for Chandrayaan-2 is typically between 0 (2017) and 4e8 (~2030)
        # Unix timestamp is typically > 1.4e9
        # MJD is typically between 50000 and 65000
        min_v, max_v = np.nanmin(num_arr), np.nanmax(num_arr)
        if 50000 <= min_v and max_v <= 70000:
            return Time(num_arr, format="mjd", scale=scale)
        elif min_v > 1e9:
            return Time(num_arr, format="unix", scale=scale)
        elif 0 <= min_v and max_v < 6e8:
            # Likely XSM MET
            return met_to_astropy_time(num_arr)
        else:
            # Fallback assuming MET
            return met_to_astropy_time(num_arr)
