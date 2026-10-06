"""
Excel workbook reader supporting both modern XLSX (via openpyxl) and legacy XLS (via xlrd).
"""

from typing import Dict, Any, Optional, List
import os
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from ..core.schema import LightCurve
from ..core.time_utils import parse_time_array
from .detector import compute_sha256


def inspect_excel_file(filepath: str) -> Dict[str, Any]:
    """
    Inspect sheets and columns of an Excel workbook.
    """
    sha256 = compute_sha256(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    
    xl = pd.ExcelFile(filepath, engine=engine)
    sheet_names = xl.sheet_names

    sheets_info = []
    for sheet in sheet_names:
        df_head = pd.read_excel(filepath, sheet_name=sheet, nrows=5, engine=engine)
        cols = list(df_head.columns)
        sheets_info.append({
            "sheet_name": sheet,
            "columns": [str(c) for c in cols],
            "sample_rows": df_head.head(3).to_dict(orient="records")
        })

    # Propose mappings for first sheet
    first_cols = sheets_info[0]["columns"] if sheets_info else []
    p_time = None
    p_rate = None
    for c in first_cols:
        cl = c.lower()
        if p_time is None and any(k in cl for k in ["time", "date", "met"]):
            p_time = c
        if p_rate is None and any(k in cl for k in ["rate", "count", "flux", "value"]):
            p_rate = c

    return {
        "filepath": filepath,
        "sha256": sha256,
        "format": ext.lstrip("."),
        "sheet_names": sheet_names,
        "sheets": sheets_info,
        "proposed_mapping": {
            "sheet_name": sheet_names[0] if sheet_names else None,
            "time_col": p_time or (first_cols[0] if first_cols else None),
            "value_col": p_rate or (first_cols[1] if len(first_cols) > 1 else None),
        }
    }


def read_excel_lightcurve(
    filepath: str,
    sheet_name: Optional[str] = None,
    time_col: str = "TIME",
    value_col: str = "RATE",
    error_col: Optional[str] = None,
    time_format: str = "xsm_met"
) -> LightCurve:
    """
    Read an Excel sheet into a canonical LightCurve.
    """
    sha256 = compute_sha256(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"

    df = pd.read_excel(filepath, sheet_name=sheet_name or 0, engine=engine)

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not in sheet columns: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not in sheet columns: {list(df.columns)}")

    raw_time = df[time_col].to_numpy()
    raw_val = df[value_col].to_numpy(dtype=np.float64)
    n = len(raw_val)

    if error_col and error_col in df.columns:
        raw_err = df[error_col].to_numpy(dtype=np.float64)
    else:
        raw_err = np.full(n, np.nan, dtype=np.float64)

    raw_qual = np.zeros(n, dtype=np.uint32)
    t_obj = parse_time_array(raw_time, format_hint=time_format)

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
        "quantity": "count_rate",
        "unit": "count / s",
        "energy_band_keV": [1.0, 15.0],
        "time_scale": "utc",
        "original_epoch": "2017-01-01T00:00:00" if time_format == "xsm_met" else "custom",
        "processing_version": f"SolarBurst-Excel-{engine}-1.0",
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
