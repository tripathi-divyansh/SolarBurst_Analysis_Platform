"""
FITS reader for astronomical X-ray light curves and GTIs using Astropy.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from ..core.schema import LightCurve
from ..core.time_utils import parse_time_array, met_to_astropy_time
from ..core.quality import create_quality_mask, QualityFlag
from .detector import compute_sha256


def inspect_fits_file(filepath: str) -> Dict[str, Any]:
    """
    Inspect HDU list and metadata of a FITS file.
    
    Returns:
    --------
    dict containing list of HDUs, column names, dimensions, and key header cards.
    """
    hdus_info = []
    with fits.open(filepath) as hdul:
        for idx, hdu in enumerate(hdul):
            info = {
                "index": idx,
                "name": hdu.name,
                "type": type(hdu).__name__,
                "header_cards": {k: str(v) for k, v in hdu.header.items() if k in [
                    "TELESCOP", "INSTRUME", "OBJECT", "DATE-OBS", "TIME-OBS",
                    "TIMESYS", "MJDREF", "TIMEZERO", "TSTART", "TSTOP", "TIMEDEL",
                    "CREATOR", "ORIGIN", "EQUINOX"
                ]}
            }
            if hasattr(hdu, "columns") and hdu.columns is not None:
                info["columns"] = hdu.columns.names
                info["nrows"] = len(hdu.data) if hdu.data is not None else 0
            hdus_info.append(info)
            
    return {
        "filepath": filepath,
        "sha256": compute_sha256(filepath),
        "hdus": hdus_info
    }


def read_fits_lightcurve(
    filepath: str,
    hdu_index: Optional[int] = None,
    time_col: str = "TIME",
    rate_col: str = "RATE",
    error_col: Optional[str] = "ERROR",
    quality_col: Optional[str] = "QUALITY",
    time_format: str = "xsm_met"
) -> LightCurve:
    """
    Read light curve and GTIs from a FITS file into a canonical LightCurve.
    """
    sha256 = compute_sha256(filepath)
    with fits.open(filepath) as hdul:
        # Auto-detect target table HDU if not provided
        target_hdu = None
        target_idx = 1
        if hdu_index is not None:
            target_hdu = hdul[hdu_index]
            target_idx = hdu_index
        else:
            for idx, h in enumerate(hdul):
                if hasattr(h, "columns") and h.columns is not None:
                    colnames = [c.upper() for c in h.columns.names]
                    if "TIME" in colnames and ("RATE" in colnames or "COUNTS" in colnames):
                        target_hdu = h
                        target_idx = idx
                        break
            if target_hdu is None:
                # Fallback to HDU 1
                target_hdu = hdul[1] if len(hdul) > 1 else hdul[0]

        header = target_hdu.header
        primary_header = hdul[0].header
        data = target_hdu.data
        if data is None or len(data) == 0:
            raise ValueError(f"No table rows found in HDU {target_idx}")

        colnames_map = {c.upper(): c for c in target_hdu.columns.names}
        
        # 1. Extract Time
        actual_time_col = colnames_map.get(time_col.upper(), time_col)
        raw_time = np.asarray(data[actual_time_col], dtype=np.float64)
        
        # 2. Extract Value (RATE or COUNTS)
        actual_rate_col = colnames_map.get(rate_col.upper(), None)
        if actual_rate_col is None:
            if "RATE" in colnames_map:
                actual_rate_col = colnames_map["RATE"]
            elif "COUNTS" in colnames_map:
                actual_rate_col = colnames_map["COUNTS"]
            else:
                raise ValueError(f"Could not find rate/count column in FITS table: {list(colnames_map.keys())}")
        raw_val = np.asarray(data[actual_rate_col], dtype=np.float64)

        # 3. Extract Error
        actual_err_col = colnames_map.get(error_col.upper() if error_col else "", None)
        if actual_err_col is not None and actual_err_col in colnames_map:
            raw_err = np.asarray(data[actual_err_col], dtype=np.float64)
        else:
            # Check for standard error column names
            for candidate in ["ERROR", "STAT_ERR", "ERR", "RATE_ERR"]:
                if candidate in colnames_map:
                    raw_err = np.asarray(data[colnames_map[candidate]], dtype=np.float64)
                    break
            else:
                raw_err = np.full_like(raw_val, np.nan)

        # 4. Extract Quality
        actual_qual_col = colnames_map.get(quality_col.upper() if quality_col else "", None)
        if actual_qual_col is not None and actual_qual_col in colnames_map:
            raw_qual = np.asarray(data[actual_qual_col], dtype=np.uint32)
        else:
            raw_qual = np.zeros(len(raw_val), dtype=np.uint32)

        # 5. Extract Timedel / Exposure
        timedel = float(header.get("TIMEDEL", 1.0))
        if "TIMEDEL" in colnames_map:
            exp_arr = np.asarray(data[colnames_map["TIMEDEL"]], dtype=np.float64)
        elif "EXPOSURE" in colnames_map:
            exp_arr = np.asarray(data[colnames_map["EXPOSURE"]], dtype=np.float64)
        elif "LIVETIME" in colnames_map:
            exp_arr = np.asarray(data[colnames_map["LIVETIME"]], dtype=np.float64)
        else:
            exp_arr = np.full_like(raw_val, timedel)

        # Check GTI extensions
        gtis = []
        for h in hdul:
            if h.name.upper() == "GTI" and hasattr(h, "data") and h.data is not None:
                gti_names = [c.upper() for c in h.columns.names]
                if "START" in gti_names and "STOP" in gti_names:
                    s_col = h.columns.names[gti_names.index("START")]
                    e_col = h.columns.names[gti_names.index("STOP")]
                    gtis.extend(list(zip(h.data[s_col], h.data[e_col])))

    # Time parsing
    t_obj = parse_time_array(raw_time, format_hint=time_format)
    from astropy.time import TimeDelta
    half_bin = TimeDelta(exp_arr / 2.0, format="sec")
    bin_start = t_obj - half_bin
    bin_end = t_obj + half_bin

    # Metadata assembly
    instrument = str(primary_header.get("INSTRUME", header.get("INSTRUME", "XSM"))).strip()
    telescope = str(primary_header.get("TELESCOP", header.get("TELESCOP", "CH2"))).strip()
    unit = str(header.get(f"TUNIT{list(colnames_map.values()).index(actual_rate_col)+1}", "count / s")).strip()

    metadata = {
        "instrument": instrument,
        "telescope": telescope,
        "quantity": "count_rate" if "rate" in actual_rate_col.lower() else "counts",
        "unit": unit,
        "energy_band_keV": [1.0, 15.0] if "XSM" in instrument else None,
        "time_scale": "utc",
        "original_epoch": "2017-01-01T00:00:00" if time_format == "xsm_met" else str(header.get("MJDREF", "unknown")),
        "processing_version": str(primary_header.get("CREATOR", "SolarBurst-1.0")),
        "calibration_version": str(primary_header.get("CALDB", "v1.5")),
        "error_kind": "supplied" if np.any(np.isfinite(raw_err)) else "poisson",
        "source_sha256": sha256,
        "source_filepath": filepath,
        "gti_count": len(gtis),
        "gtis": gtis[:50],  # store up to 50 for summary
        "raw_headers": {k: str(v) for k, v in header.items() if k in ["TELESCOP", "INSTRUME", "TIMEDEL", "TSTART", "TSTOP"]}
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
