from pathlib import Path
import json
import numpy as np


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False))


def save_analysis(result, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    result["catalog"].to_csv(out/"candidates.csv", index=False)
    write_json(out/"fits.json", result["fits"])
    write_json(out/"manifest.json", result["manifest"])
    np.savez_compressed(out/"baseline.npz", time_s=result["lightcurve"].time, background=result["baseline"])
