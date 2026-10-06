"""
NASA Common Data Format (CDF) reader using cdflib with proper epoch conversions.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import cdflib
from astropy.time import Time, TimeDelta
from ..core.schema import LightCurve
from .detector import compute_sha256


def inspect_cdf_file(filepath: str) -> Dict[str, Any]:
    """
    Inspect variables and global attributes in a NASA CDF file.
    """
    sha256 = compute_sha256(filepath)
    cdf = cdflib.CDF(filepath)
    info = cdf.cdf_info()

    variables_info = []
    for var in info.zVariables:
        var_atts = cdf.varattsget(var)
        var_info = {
            "name": var,
            "shape": cdf.varinq(var).Dim_Sizes,
            "data_type": cdf.varinq(var).Data_Type_Description,
            "units": var_atts.get("UNITS", ""),
            "description": var_atts.get("CATDESC", var_atts.get("FIELDNAM", ""))
        }
        variables_info.append(var_info)

    # Propose time and value variables
    p_time = None
    p_val = None
    for v in variables_info:
        vn = v["name"].lower()
        if p_time is None and ("epoch" in vn or "time" in vn):
            p_time = v["name"]
        if p_val is None and ("flux" in vn or "rate" in vn or "count" in vn or "data" in vn):
            p_val = v["name"]

    return {
        "filepath": filepath,
        "sha256": sha256,
        "variables": variables_info,
        "proposed_mapping": {
            "time_var": p_time or (variables_info[0]["name"] if variables_info else None),
            "value_var": p_val or (variables_info[1]["name"] if len(variables_info) > 1 else None)
        }
    }


def read_cdf_lightcurve(
    filepath: str,
    time_var: str = "Epoch",
    value_var: str = "Flux",
    error_var: Optional[str] = None
) -> LightCurve:
    """
    Read a NASA CDF file into a canonical LightCurve.
    """
    sha256 = compute_sha256(filepath)
    cdf = cdflib.CDF(filepath)
    
    raw_epoch = cdf.varget(time_var)
    raw_val = np.asarray(cdf.varget(value_var), dtype=np.float64).flatten()
    n = len(raw_val)

    # Convert CDF Epochs to ISO strings / Astropy Time
    try:
        iso_times = cdflib.cdfepoch.to_datetime(raw_epoch)
        t_obj = Time(iso_times, scale="utc")
    except Exception:
        # Fallback numeric conversion
        t_obj = Time(raw_epoch, format="unix", scale="utc")

    if error_var and error_var in cdf.cdf_info().zVariables:
        raw_err = np.asarray(cdf.varget(error_var), dtype=np.float64).flatten()
    else:
        raw_err = np.full(n, np.nan, dtype=np.float64)

    raw_qual = np.zeros(n, dtype=np.uint32)
    
    if n > 1:
        dt = np.median(np.diff((t_obj - t_obj[0]).to_value("sec")))
        if dt <= 0 or not np.isfinite(dt):
            dt = 1.0
    else:
        dt = 1.0

    exp_arr = np.full(n, dt, dtype=np.float64)
    half_bin = TimeDelta(dt / 2.0, format="sec")
    bin_start = t_obj - half_bin
    bin_end = t_obj + half_bin

    val_atts = cdf.varattsget(value_var)
    unit = str(val_atts.get("UNITS", "count / s"))

    metadata = {
        "instrument": "CDF-Instrument",
        "quantity": "count_rate" if "rate" in value_var.lower() else "energy_flux",
        "unit": unit,
        "energy_band_keV": None,
        "time_scale": "utc",
        "original_epoch": "cdf_epoch",
        "processing_version": "SolarBurst-CDF-1.0",
        "calibration_version": "unspecified",
        "error_kind": "supplied" if np.any(np.isfinite(raw_err)) else "estimated",
        "source_sha256": sha256,
        "source_filepath": filepath,
    }

    return LightCurve(
        time=t_obj,
        value=raw_val,
        error=raw_err,
        quality=raw_qual,
        bin_start=bin_start,
        bin_end=bin_end,
        exposure_s=exp_arr,
        metadata=metadata
    )
