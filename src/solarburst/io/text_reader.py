"""
ASCII / CSV / TSV text reader with automatic delimiter detection and column mapping.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from ..core.schema import LightCurve
from ..core.time_utils import parse_time_array
from .detector import compute_sha256


def inspect_text_file(filepath: str, max_preview_lines: int = 15) -> Dict[str, Any]:
    """
    Inspect text file headers, delimiters, and columns.
    """
    sha256 = compute_sha256(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        sample_lines = [line.strip() for line in f.readlines()[:max_preview_lines]]

    # Detect delimiter
    non_comment_lines = [l for l in sample_lines if l and not l.startswith("#")]
    delimiter = ","
    if non_comment_lines:
        first = non_comment_lines[0]
        if "\t" in first:
            delimiter = "\t"
        elif ";" in first:
            delimiter = ";"
        elif "," in first:
            delimiter = ","
        elif " " in first:
            delimiter = r"\s+"

    # Read preview dataframe
    try:
        df_prev = pd.read_csv(
            filepath,
            sep=delimiter,
            comment="#",
            nrows=5,
            engine="python" if delimiter == r"\s+" else "c"
        )
        columns = list(df_prev.columns)
    except Exception:
        columns = []

    # Propose column mappings
    proposed_time = None
    proposed_rate = None
    proposed_err = None
    for col in columns:
        cl = str(col).lower()
        if proposed_time is None and any(k in cl for k in ["time", "date", "met", "seconds", "mjd"]):
            proposed_time = col
        if proposed_rate is None and any(k in cl for k in ["rate", "count", "flux", "value", "intensity"]):
            proposed_rate = col
        if proposed_err is None and any(k in cl for k in ["err", "sigma", "unc"]):
            proposed_err = col

    return {
        "filepath": filepath,
        "sha256": sha256,
        "delimiter": delimiter,
        "preview_lines": sample_lines,
        "columns": columns,
        "proposed_mapping": {
            "time_col": proposed_time or (columns[0] if columns else None),
            "value_col": proposed_rate or (columns[1] if len(columns) > 1 else None),
            "error_col": proposed_err,
        }
    }


def read_text_lightcurve(
    filepath: str,
    time_col: str,
    value_col: str,
    error_col: Optional[str] = None,
    quality_col: Optional[str] = None,
    time_format: str = "xsm_met",
    delimiter: Optional[str] = None,
    unit: str = "count / s",
    quantity: str = "count_rate"
) -> LightCurve:
    """
    Read text light curve into canonical LightCurve.
    """
    sha256 = compute_sha256(filepath)
    if delimiter is None:
        inspect = inspect_text_file(filepath)
        delimiter = inspect["delimiter"]

    df = pd.read_csv(
        filepath,
        sep=delimiter,
        comment="#",
        engine="python" if delimiter == r"\s+" else "c"
    )

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in file. Available: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in file. Available: {list(df.columns)}")

    raw_time = df[time_col].to_numpy()
    raw_val = df[value_col].to_numpy(dtype=np.float64)
    n = len(raw_val)

    if error_col and error_col in df.columns:
        raw_err = df[error_col].to_numpy(dtype=np.float64)
    else:
        raw_err = np.full(n, np.nan, dtype=np.float64)

    if quality_col and quality_col in df.columns:
        raw_qual = df[quality_col].to_numpy(dtype=np.uint32)
    else:
        raw_qual = np.zeros(n, dtype=np.uint32)

    t_obj = parse_time_array(raw_time, format_hint=time_format)

    # Estimate cadence / bin duration
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

    metadata = {
        "instrument": "XSM",
        "quantity": quantity,
        "unit": unit,
        "energy_band_keV": [1.0, 15.0],
        "time_scale": "utc",
        "original_epoch": "2017-01-01T00:00:00" if time_format == "xsm_met" else "custom",
        "processing_version": "SolarBurst-TextReader-1.0",
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
