"""
Catalog exporters supporting CSV, FITS binary tables, JSON provenance manifests, and HTML reports.
"""

from typing import List, Dict, Any, Optional
import os
import json
import pandas as pd
import numpy as np
from astropy.table import Table
from ..core.schema import BurstResult, LightCurve, AnalysisConfig


def export_catalog_csv(results: List[BurstResult], output_path: str) -> str:
    """
    Export catalog of bursts to CSV format with complete parameter definitions and uncertainties.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = []
    for r in results:
        d = r.to_dict()
        # Flatten uncertainties
        for k, v in r.uncertainties.items():
            if isinstance(v, (int, float)):
                d[f"unc_{k}"] = v
        d.pop("uncertainties", None)
        d.pop("quality_flags", None)
        rows.append(d)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def export_catalog_fits(results: List[BurstResult], output_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Export catalog of bursts to standard FITS binary table.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not results:
        t = Table()
        t.write(output_path, format="fits", overwrite=True)
        return output_path

    cols = {
        "BURST_ID": [r.burst_id for r in results],
        "PEAK_MET": [r.peak_time_met for r in results],
        "PEAK_ISO": [r.peak_time_iso for r in results],
        "START_MET": [r.start_time_met for r in results],
        "END_MET": [r.end_time_met for r in results],
        "DURATION_S": [r.duration_s for r in results],
        "RISE_TIME_S": [r.rise_time_s for r in results],
        "DECAY_TAU_S": [r.decay_tau_s for r in results],
        "NET_PEAK": [r.net_peak for r in results],
        "TOTAL_PEAK": [r.total_peak for r in results],
        "FLUENCE": [r.fluence for r in results],
        "PEAK_SNR": [r.peak_snr for r in results],
        "ASYMMETRY": [r.asymmetry_rho for r in results],
        "MORPHOLOGY": [r.morphology_class for r in results],
        "DURATION_CLASS": [r.duration_class for r in results],
        "INTENSITY": [r.intensity_class for r in results],
        "RELIABILITY": [r.reliability for r in results],
        "CHI2_RED": [r.reduced_chi2 for r in results],
        "ML_SCORE": [r.ml_score for r in results],
        "DECISION": [r.decision for r in results],
    }

    t = Table(cols)
    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)):
                t.meta[k[:8].upper()] = str(v)
    t.meta["CREATOR"] = "SolarBurst-XSM-Platform"
    t.write(output_path, format="fits", overwrite=True)
    return output_path


def export_provenance_json(
    results: List[BurstResult],
    lc: LightCurve,
    config: AnalysisConfig,
    output_path: str
) -> str:
    """
    Export comprehensive execution manifest and provenance log.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    manifest = {
        "software": "SolarBurst — XSM Burst Analysis Platform",
        "version": "1.0.0",
        "input_file": lc.metadata.get("source_filepath", "unspecified"),
        "input_sha256": lc.metadata.get("source_sha256", "unspecified"),
        "instrument": lc.metadata.get("instrument", "XSM"),
        "telescope": lc.metadata.get("telescope", "Chandrayaan-2"),
        "quantity": lc.metadata.get("quantity", "count_rate"),
        "unit": lc.metadata.get("unit", "count / s"),
        "time_span_s": float(lc.met_seconds[-1] - lc.met_seconds[0]) if len(lc.value) > 1 else 0,
        "total_points": len(lc.value),
        "config": {
            "name": config.name,
            "asls_lambda": config.asls_lambda,
            "asls_p": config.asls_p,
            "model_family": config.model_family,
            "fit_profile": config.fit_profile,
            "ml_accept_threshold": config.ml_accept_threshold,
            "ml_review_threshold": config.ml_review_threshold,
        },
        "catalog_summary": {
            "total_detected": len(results),
            "accepted": sum(1 for r in results if r.decision == "accepted"),
            "review": sum(1 for r in results if r.decision == "review"),
            "rejected": sum(1 for r in results if r.decision == "rejected"),
        },
        "bursts": [r.to_dict() for r in results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output_path
