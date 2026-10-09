"""Explicit mapping; no silent FITS epoch or quantity guessing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .schema import LightCurve


def load_lightcurve(path, mapping):
    """mapping is a dict or JSON path. All numeric input times are explicitly seconds.

    FITS: select the HDU and supply resolved epoch/time_scale, time_offset_s,
    time_unit_s, and bin-center convention. GTIs are applied conservatively.
    This generic adapter does not claim automatic mission calibration.
    """
    path = Path(path)
    m = json.loads(Path(mapping).read_text()) if isinstance(mapping, (str, Path)) else dict(mapping)
    required = {"time", "value", "quantity", "unit", "epoch", "time_scale", "time_unit_s", "time_position"}
    if required - m.keys():
        raise ValueError(f"Missing explicit mapping fields: {sorted(required-m.keys())}")
    if m["time_position"] not in {"center", "start", "end"}:
        raise ValueError("time_position must be center, start or end")
    factor = float(m["time_unit_s"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("time_unit_s must be positive")
    suffix = path.suffix.lower()
    gtis = []
    if suffix in {".fits", ".fit", ".lc", ".gz"}:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdus:
            hdu = hdus[m.get("hdu", 1)]
            if hdu.data is None or not hasattr(hdu.data, "names"):
                raise ValueError("Selected HDU is not a table")
            # Casting columns independently avoids big-endian pandas operations.
            df = pd.DataFrame({name: np.asarray(hdu.data[name], dtype=float)
                               for name in hdu.data.names if np.asarray(hdu.data[name]).ndim == 1})
            for gh in hdus:
                if gh.name.upper() in {"GTI", "STDGTI"} and gh.data is not None:
                    names = set(gh.data.names or [])
                    if {"START", "STOP"} <= names:
                        gtis.extend(zip(np.asarray(gh.data["START"], float), np.asarray(gh.data["STOP"], float)))
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path, sheet_name=m.get("sheet", 0))
    elif suffix in {".csv", ".tsv", ".txt", ".dat"}:
        df = pd.read_csv(path, sep=m.get("delimiter", "\t" if suffix == ".tsv" else r"\s+" if suffix in {".txt", ".dat"} else ","), comment="#")
    else:
        raise ValueError("Supported input: CSV/TSV/whitespace ASCII, XLS/XLSX, FITS/.lc/.fits.gz")

    def column(key, scalar_key=None, default=None):
        if key in m:
            if m[key] not in df:
                raise ValueError(f"Column {m[key]!r} not present")
            return pd.to_numeric(df[m[key]], errors="raise").to_numpy(float)
        if scalar_key and scalar_key in m:
            return np.full(len(df), float(m[scalar_key]))
        if default is not None:
            return np.full(len(df), default)
        raise ValueError(f"Specify {key} column or {scalar_key} scalar")

    t = column("time") * factor + float(m.get("time_offset_s", 0))
    width = column("bin_width", "bin_width_s")  # widths/exposure explicitly in seconds
    exp = column("exposure", "exposure_s")
    t = t + width/2 if m["time_position"] == "start" else t-width/2 if m["time_position"] == "end" else t
    quality = column("quality", default=0)
    # User must map instrument bit semantics first; any nonzero value is invalid.
    if gtis:
        contained = np.zeros(len(t), bool)
        offset = float(m.get("time_offset_s", 0))
        for lo, hi in gtis:
            contained |= (t-width/2 >= lo*factor+offset) & (t+width/2 <= hi*factor+offset)
        quality[~contained] = 1
    metadata = {k: m[k] for k in ("quantity", "unit", "epoch", "time_scale")}
    metadata.update({"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "input_mapping": m, "energy_band": m.get("energy_band"),
                     "instrument": m.get("instrument", "unspecified")})
    return LightCurve(t, column("value"), column("error") if "error" in m else None,
                      width, exp, quality, metadata)
