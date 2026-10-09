from dataclasses import asdict
import hashlib
import numpy as np
import pandas as pd
from .schema import PipelineConfig, PIPELINE_VERSION
from .detection import estimate_background, proposals
from .physics import fit_fred
from .features import FEATURES, FEATURE_VERSION, candidate_features, operational_morphology


def analyze(lightcurve, config=None, group_id="unassigned"):
    cfg = config or PipelineConfig()
    lc = lightcurve.prepared(cfg.duplicate_policy)
    rate, err = lc.rates()
    rows, details, warnings = [], {}, list(lc.warnings)
    full_base = np.full(len(lc.time), np.nan)
    valid_exposure = float(lc.exposure[lc.valid].sum())
    processed_exposure = 0.0
    source = lc.metadata.get("source_sha256") or hashlib.sha256(np.column_stack([lc.time, lc.value]).tobytes()).hexdigest()
    signal_unit = "count / s" if lc.metadata["quantity"] == "counts" else lc.metadata["unit"]
    for segno, ids in enumerate(lc.segments()):
        if len(ids) < cfg.min_segment_bins:
            warnings.append(f"segment_{segno}: insufficient valid bins; excluded from processed exposure")
            continue
        processed_exposure += float(lc.exposure[ids].sum())
        t, y, e, width = lc.time[ids], rate[ids], err[ids], lc.bin_width[ids]
        base, noise, poor_baseline = estimate_background(t, y, e, cfg.baseline_window_s)
        full_base[ids] = base
        candidates, evidence = proposals(t, y, e, base, noise, cfg)
        for j, candidate in enumerate(candidates):
            lo, hi = candidate["lo"], candidate["hi"]
            context = max(cfg.fit_context_s, 0.5*(t[hi]-t[lo]))
            fit_ids = np.flatnonzero((t >= t[lo]-context) & (t <= t[hi]+context))
            # Prevent other separately proposed events from being fitted as background.
            contaminated = np.zeros(len(t), bool)
            for k, other in enumerate(candidates):
                if k != j:
                    contaminated[other["lo"]:other["hi"]+1] = True
            fit_ids = fit_ids[~contaminated[fit_ids]]
            raw = lc.value[ids[fit_ids]] if lc.metadata["quantity"] == "counts" else None
            if hi == lo:
                fit = {"success": False, "parameters_reliable": False,
                       "flags": ["insufficient_event_bins"]}
            else:
                fit = fit_fred(t[fit_ids]-width[fit_ids]/2, t[fit_ids]+width[fit_ids]/2,
                               y[fit_ids], noise[fit_ids], lc.exposure[ids[fit_ids]], counts=raw,
                               bootstrap=cfg.bootstrap_samples, seed=cfg.random_seed+j,
                               peak_window=(t[lo]-width[lo]/2, t[hi]+width[hi]/2))
            fit["signal_unit"] = signal_unit
            fit["fluence_unit"] = "count" if lc.metadata["quantity"] == "counts" else f"({signal_unit}) s"
            feat = candidate_features(t, width, y, base, noise, candidate, evidence, fit)
            quality = list(evidence["flags"])+list(fit["flags"])
            if poor_baseline: quality.append("insufficient_quiet_background")
            if feat["edge_candidate"]: quality.append("event_at_segment_boundary")
            if hi == lo: quality.append("single_bin_candidate")
            if feat["n_peaks"] > 1: quality.append("multi_peak_single_component_fit")
            key = f"{source[:10]}_{cfg.fingerprint()[:8]}_{segno}_{j}"
            row = {"candidate_id": key, "group_id": str(group_id), "source_hash": source,
                   "pipeline_hash": cfg.fingerprint(), "feature_version": FEATURE_VERSION,
                   "quantity": lc.metadata["quantity"], "unit": signal_unit,
                   "observed_start_s": float(t[lo]-width[lo]/2),
                   "observed_peak_s": float(t[candidate["peak"]]),
                   "observed_end_s": float(t[hi]+width[hi]/2),
                   "status": "review_unscored", "burst_score": np.nan,
                   "operational_morphology": operational_morphology(fit, feat["n_peaks"]),
                   "quality_flags": ";".join(sorted(set(quality))), **feat}
            rows.append(row)
            details[key] = fit
    columns = ["candidate_id", "group_id", "source_hash", "pipeline_hash", "feature_version", "quantity", "unit",
               "observed_start_s", "observed_peak_s", "observed_end_s", "status", "burst_score",
               "operational_morphology", "quality_flags"]+FEATURES
    return {"catalog": pd.DataFrame(rows, columns=columns), "fits": details,
            "baseline": full_base, "lightcurve": lc,
            "manifest": {"pipeline_version": PIPELINE_VERSION, "metadata": lc.metadata, "signal_unit": signal_unit, "config": asdict(cfg), "source_hash": source,
                         "pipeline_hash": cfg.fingerprint(), "feature_version": FEATURE_VERSION,
                         "valid_exposure_s": valid_exposure, "processed_exposure_s": processed_exposure,
                         "warnings": warnings, "n_candidates": len(rows),
                         "scientific_status": "unvalidated_on_real_XSM"}}
