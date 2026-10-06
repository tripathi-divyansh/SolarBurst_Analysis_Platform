"""
Chandrayaan-2 XSM (Solar X-ray Monitor) mission adapter and unified light curve loader.
"""

from typing import Dict, Any, Optional
import os
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from ..core.schema import LightCurve
from ..core.quality import QualityFlag, create_quality_mask
from ..core.time_utils import met_to_astropy_time
from .detector import detect_file_format, compute_sha256
from .fits_reader import read_fits_lightcurve, inspect_fits_file
from .text_reader import read_text_lightcurve, inspect_text_file
from .excel_reader import read_excel_lightcurve, inspect_excel_file
from .cdf_reader import read_cdf_lightcurve, inspect_cdf_file


def load_xsm_fits(filepath: str) -> LightCurve:
    """
    Dedicated Chandrayaan-2 XSM PRADAN FITS reader.
    Converts XSM MET to UTC, checks dead-time fractions, GTIs, and filter flags.
    """
    sha256 = compute_sha256(filepath)
    with fits.open(filepath) as hdul:
        # Locate RATE table
        rate_hdu = None
        for h in hdul:
            if hasattr(h, "columns") and h.columns is not None:
                cols = [c.upper() for c in h.columns.names]
                if "TIME" in cols and ("RATE" in cols or "COUNTS" in cols):
                    rate_hdu = h
                    break
        if rate_hdu is None:
            rate_hdu = hdul[1] if len(hdul) > 1 else hdul[0]

        data = rate_hdu.data
        hdr = rate_hdu.header
        pri_hdr = hdul[0].header

        cols_map = {c.upper(): c for c in rate_hdu.columns.names}
        time_met = np.asarray(data[cols_map["TIME"]], dtype=np.float64)
        n = len(time_met)

        # Rate
        rate_col_name = "RATE" if "RATE" in cols_map else ("COUNTS" if "COUNTS" in cols_map else list(cols_map.keys())[1])
        rate_val = np.asarray(data[cols_map[rate_col_name]], dtype=np.float64)

        # Error
        if "ERROR" in cols_map:
            rate_err = np.asarray(data[cols_map["ERROR"]], dtype=np.float64)
        elif "STAT_ERR" in cols_map:
            rate_err = np.asarray(data[cols_map["STAT_ERR"]], dtype=np.float64)
        else:
            # Poisson approximation for rate with exposure
            timedel_val = float(hdr.get("TIMEDEL", 1.0))
            expected_counts = np.maximum(rate_val * timedel_val, 0.0)
            rate_err = np.sqrt(np.maximum(expected_counts, 1.0)) / timedel_val

        # Quality & Fraction
        qual_mask = np.zeros(n, dtype=np.uint32)
        if "QUALITY" in cols_map:
            raw_q = np.asarray(data[cols_map["QUALITY"]], dtype=np.uint32)
            qual_mask |= raw_q

        # Dead-time fraction / LIVETIME
        timedel = float(hdr.get("TIMEDEL", 1.0))
        if "TIMEDEL" in cols_map:
            timedel_arr = np.asarray(data[cols_map["TIMEDEL"]], dtype=np.float64)
        else:
            timedel_arr = np.full(n, timedel, dtype=np.float64)

        if "FRACTION" in cols_map:
            fraction = np.asarray(data[cols_map["FRACTION"]], dtype=np.float64)
            # When fraction is unusually low (< 0.1), flag as potential saturation / loss
            low_frac = fraction < 0.1
            qual_mask[low_frac] |= int(QualityFlag.SATURATED)
            exp_arr = timedel_arr * np.clip(fraction, 0.0, 1.0)
        else:
            exp_arr = timedel_arr

        # Check for Be filter state keywords or column
        filter_state = "OPEN"
        for k in ["FILTER", "BE_FILT", "FILTER_STATUS"]:
            if k in hdr:
                filter_state = str(hdr[k])
            elif k in pri_hdr:
                filter_state = str(pri_hdr[k])

        # Flag filter transition discontinuities if present
        if "FILTER" in cols_map:
            filt_arr = data[cols_map["FILTER"]]
            # Any changes flag filter transition
            diffs = np.where(filt_arr[:-1] != filt_arr[1:])[0] + 1
            qual_mask[diffs] |= int(QualityFlag.FILTER_TRANSITION)

        # Flag NaNs or negative invalid rates
        bad_idx = ~np.isfinite(rate_val) | (rate_val < -1e-5)
        qual_mask[bad_idx] |= int(QualityFlag.INVALID)

        # Extract GTIs
        gtis = []
        for h in hdul:
            if h.name.upper() == "GTI" and hasattr(h, "data") and h.data is not None:
                g_cols = [c.upper() for c in h.columns.names]
                if "START" in g_cols and "STOP" in g_cols:
                    s_c = h.columns.names[g_cols.index("START")]
                    e_c = h.columns.names[g_cols.index("STOP")]
                    gtis = list(zip(h.data[s_c].astype(float), h.data[e_c].astype(float)))

    # Convert XSM MET to UTC
    t_obj = met_to_astropy_time(time_met)
    half_bin = TimeDelta(exp_arr / 2.0, format="sec")
    bin_start = t_obj - half_bin
    bin_end = t_obj + half_bin

    metadata = {
        "instrument": "XSM",
        "telescope": "Chandrayaan-2",
        "quantity": "count_rate",
        "unit": "count / s",
        "energy_band_keV": [1.0, 15.0],
        "time_scale": "utc",
        "original_epoch": "2017-01-01T00:00:00",
        "processing_version": str(pri_hdr.get("CREATOR", "PRADAN-XSM-Adapter-1.5")),
        "calibration_version": str(pri_hdr.get("CALDB", "v1.5")),
        "filter_state": filter_state,
        "error_kind": "supplied" if "ERROR" in cols_map else "poisson",
        "source_sha256": sha256,
        "source_filepath": filepath,
        "gti_count": len(gtis),
        "gtis": gtis[:50],
        "met_range": [float(np.nanmin(time_met)), float(np.nanmax(time_met))],
        "exposure_total_s": float(np.sum(exp_arr[qual_mask == 0])),
    }

    return LightCurve(
        time=t_obj,
        value=rate_val,
        error=rate_err,
        quality=qual_mask,
        bin_start=bin_start,
        bin_end=bin_end,
        exposure_s=exp_arr,
        metadata=metadata
    )


def inspect_file(filepath: str) -> Dict[str, Any]:
    """Inspect any supported file format and return structural metadata and preview."""
    detection = detect_file_format(filepath)
    fmt = detection["format"]

    if fmt in ["fits", "fits_gz"]:
        res = inspect_fits_file(filepath)
    elif fmt in ["csv", "tsv", "ascii"]:
        res = inspect_text_file(filepath)
    elif fmt in ["xlsx", "xls"]:
        res = inspect_excel_file(filepath)
    elif fmt == "cdf":
        res = inspect_cdf_file(filepath)
    else:
        res = {"filepath": filepath, "sha256": detection["sha256"], "error": f"Unsupported format: {fmt}"}

    res["detection"] = detection
    return res


def load_lightcurve(filepath: str, options: Optional[Dict[str, Any]] = None) -> LightCurve:
    """
    Unified loader for all supported formats.
    """
    options = options or {}
    detection = detect_file_format(filepath)
    fmt = detection["format"]

    # If XSM FITS or user requests XSM loader
    if fmt in ["fits", "fits_gz"]:
        # Check if XSM specific or generic FITS
        try:
            return load_xsm_fits(filepath)
        except Exception:
            return read_fits_lightcurve(
                filepath,
                time_col=options.get("time_col", "TIME"),
                rate_col=options.get("value_col", "RATE"),
                error_col=options.get("error_col", "ERROR"),
                time_format=options.get("time_format", "xsm_met")
            )
    elif fmt in ["csv", "tsv", "ascii"]:
        return read_text_lightcurve(
            filepath,
            time_col=options.get("time_col", "TIME"),
            value_col=options.get("value_col", "RATE"),
            error_col=options.get("error_col", None),
            time_format=options.get("time_format", "xsm_met"),
            delimiter=options.get("delimiter", None),
            unit=options.get("unit", "count / s")
        )
    elif fmt in ["xlsx", "xls"]:
        return read_excel_lightcurve(
            filepath,
            sheet_name=options.get("sheet_name", None),
            time_col=options.get("time_col", "TIME"),
            value_col=options.get("value_col", "RATE"),
            error_col=options.get("error_col", None),
            time_format=options.get("time_format", "xsm_met")
        )
    elif fmt == "cdf":
        return read_cdf_lightcurve(
            filepath,
            time_var=options.get("time_var", "Epoch"),
            value_var=options.get("value_var", "Flux"),
            error_var=options.get("error_var", None)
        )
    else:
        raise ValueError(f"Cannot load file with detected format '{fmt}': {filepath}")
