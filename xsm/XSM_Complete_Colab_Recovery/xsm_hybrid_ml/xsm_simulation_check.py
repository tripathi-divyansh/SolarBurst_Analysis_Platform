"""Controlled simulation check of an EXISTING, frozen XSM anomaly pipeline.

No fitting, threshold tuning, or annotation changes occur here. Simulated
Gaussian measurement noise is not a validated model of XSM instrument noise.
Recovery on these examples is not real-XSM accuracy or an overfitting verdict.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from sklearn.utils.validation import check_is_fitted
from xsm_burst.schema import LightCurve, PipelineConfig
from xsm_burst.pipeline import analyze
from xsm_burst.physics import fred_bin_mean
from xsm_burst.evaluation import match_events

EVENT_COLUMNS = ["group_id", "observed_start_s", "observed_peak_s", "observed_end_s"]


def run_simulation_check(model, feature_columns, threshold, frame, output_dir):
    """24 independent 900-second controls, with fixed seeds and protocol.

16 burst cases: four strengths x four trials, two FRED and two Gaussian at
each strength, all also containing artificial instrument-like transients.
Eight zero-burst controls: four noise-only, four with artificial transients.
Baseline/noise scales come ONLY from the training candidate table.
"""
    check_is_fitted(model)
    if "split" not in frame:
        raise ValueError("Pass demo_data or review.frame with its saved day split.")
    train = frame.loc[frame.split.eq("train")].copy()
    if len(train) == 0 or not np.isfinite(threshold):
        raise ValueError("Training rows and a finite frozen threshold are required.")
    cfg = PipelineConfig()
    if set(train.pipeline_hash) != {cfg.fingerprint()}:
        raise ValueError("This test expects the default v0.2 extraction configuration.")
    if set(train.quantity) != {"count_rate"} or set(train.unit) != {"count / s"}:
        raise ValueError("This test expects XSM count-rate features in count / s.")
    if not np.allclose(train.cadence_s, 1.0, atol=1e-4):
        raise ValueError("This simulation protocol assumes one-second sampling.")
    if list(model.feature_names_in_) != list(feature_columns):
        raise ValueError("Feature ordering differs from the fitted model.")
    # Approximate effective noise inferred from candidate features, not a
    # measurement of the quiet-time instrument noise distribution.
    scales = pd.DataFrame({
        "background": train.background_at_peak,
        "noise": train.net_peak_observed / train.peak_snr.replace(0, np.nan),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    scales = scales[(scales.background > 0) & (scales.noise > 0)]
    if not len(scales):
        raise ValueError("No positive background/noise scale estimates.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "scope": "Controlled simulation only; not real-XSM accuracy or an overfitting diagnosis",
        "seed_start": 71000, "seconds_per_trial": 900, "trials_per_strength": 4,
        "strengths_peak_over_noise": [3, 5, 8, 12], "matching_peak_tolerance_s": 35,
        "threshold_frozen": float(threshold), "feature_columns": list(feature_columns),
        "pipeline_hash": cfg.fingerprint(), "train_group_ids": sorted(train.group_id.astype(str).unique().tolist()),
        "noise_assumption": "Independent Gaussian noise; scale approximated using training candidate net_peak_observed / peak_snr",
        "background_assumption": "Training candidate background level plus a small smooth drift",
        "artifacts": "One 8-sigma single-bin spike and one 6-sigma six-second rectangular transient",
        "truth": "Injected profiles above 5% peak; one-to-one interval overlap and peak separation <=35 seconds",
        "warning": "Only four trials per strength. Inspect counts, not just percentages. Do not tune on these fixed controls then call them independent tests.",
    }
    (out / "simulation_protocol.json").write_text(json.dumps(protocol, indent=2))

    trials = [("burst", strength, rep) for strength in [3, 5, 8, 12] for rep in range(4)]
    trials += [(kind, 0, rep) for kind in ["noise_only", "artifacts_only"] for rep in range(4)]
    records, predictions, truths = [], [], []
    for trial_id, (kind, strength, rep) in enumerate(trials):
        rng = np.random.default_rng(71000 + trial_id)
        chosen = scales.iloc[int(rng.integers(len(scales)))]
        sigma, level = float(chosen.noise), float(chosen.background)
        time = np.arange(900, dtype=float) + 0.5
        baseline = level + 0.5*sigma*np.sin(2*np.pi*time/900) + 0.5*sigma*(time/900 - 0.5)
        values = baseline + rng.normal(0, sigma, len(time))
        group = f"sim_{trial_id:03d}"
        truth = pd.DataFrame(columns=EVENT_COLUMNS)
        shape = "none"
        if kind == "burst":
            if rep % 2 == 0:
                shape = "FRED"
                profile = fred_bin_mean(time-0.5, time+0.5, 1, 180, rng.uniform(4,10), rng.uniform(18,35))
            else:
                shape = "Gaussian"
                width = rng.uniform(8,18)
                center = 220.0
                profile = width*np.sqrt(np.pi/2)*(erf((time+0.5-center)/(np.sqrt(2)*width))-erf((time-0.5-center)/(np.sqrt(2)*width)))
            profile *= strength*sigma/profile.max()
            values += profile
            event = np.flatnonzero(profile >= 0.05*profile.max())
            truth = pd.DataFrame([{
                "group_id": group,
                "observed_start_s": float(time[event[0]]-0.5),
                "observed_peak_s": float(time[np.argmax(profile)]),
                "observed_end_s": float(time[event[-1]]+0.5),
            }])
            truths.append(truth.assign(shape=shape, injected_peak_over_noise=strength))
        if kind in {"burst", "artifacts_only"}:
            values[500] += 8*sigma
            values[700:706] += 6*sigma
        lc = LightCurve(time, values, np.full(900, sigma), np.ones(900), np.ones(900), np.zeros(900, int),
                        {"quantity": "count_rate", "unit": "count / s", "epoch": "relative", "time_scale": "relative", "synthetic": True})
        result = analyze(lc, config=cfg, group_id=group)
        candidates = result["catalog"].copy()
        if len(candidates):
            features = candidates[feature_columns].astype(float).replace([np.inf,-np.inf],np.nan)
            candidates["anomaly_score"] = -model.score_samples(features)
        else:
            candidates["anomaly_score"] = pd.Series(dtype=float)
        candidates["high_priority"] = candidates.anomaly_score >= threshold
        candidates["simulation_kind"] = kind
        candidates["injected_peak_over_noise"] = strength
        predictions.append(candidates)
        for stage, table in [("Extraction", candidates), ("Extraction + ML", candidates[candidates.high_priority])]:
            matched = match_events(table, truth, tolerance_s=35)
            records.append({
                "trial": group, "kind": kind, "strength": strength, "shape": shape,
                "stage": stage, "injected": len(truth), "recovered": matched["tp"],
                "missed": matched["fn"], "false_triggers": matched["fp"],
                "processed_seconds": float(result["manifest"]["processed_exposure_s"]),
            })
        print(f"Simulation {trial_id+1}/{len(trials)} finished", flush=True)

    metrics = pd.DataFrame(records)
    metrics.to_csv(out / "per_trial_results.csv", index=False)
    nonempty = [table for table in predictions if len(table)]
    combined = pd.concat(nonempty, ignore_index=True) if nonempty else predictions[0].iloc[:0]
    combined.to_csv(out / "simulated_candidate_scores.csv", index=False)
    pd.concat(truths, ignore_index=True).to_csv(out / "injected_event_truth.csv", index=False)
    recovery = metrics[metrics.kind == "burst"].groupby(["stage", "strength"])[["injected", "recovered", "missed", "false_triggers"]].sum().reset_index()
    recovery["recovery_percent"] = 100*recovery.recovered/recovery.injected
    recovery.to_csv(out / "recovery_by_strength.csv", index=False)
    controls = metrics[metrics.kind != "burst"].groupby(["stage", "kind"])[["false_triggers", "processed_seconds"]].sum().reset_index()
    controls["false_triggers_per_simulated_hour"] = controls.false_triggers/(controls.processed_seconds/3600)
    controls.to_csv(out / "negative_control_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for stage, color, marker in [("Extraction", "#2474b5", "o"), ("Extraction + ML", "#df7b22", "s")]:
        sub = recovery[recovery.stage == stage]
        axes[0].plot(sub.strength, sub.recovery_percent, marker=marker, color=color, label=stage)
    axes[0].set(xlabel="Injected peak / simulated noise scale", ylabel="Injected bursts recovered (%)", title="Recovery on new simulations", ylim=(-5,105), xticks=[3,5,8,12])
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for stage, color, offset in [("Extraction", "#2474b5", -.18), ("Extraction + ML", "#df7b22", .18)]:
        sub = controls[controls.stage == stage].set_index("kind").loc[["noise_only", "artifacts_only"]]
        bars = axes[1].bar(np.arange(2)+offset, sub.false_triggers_per_simulated_hour, width=.36, color=color, label=stage)
        axes[1].bar_label(bars, fmt="%.1f", padding=3)
    axes[1].set(xticks=[0,1], xticklabels=["Noise only", "Noise + artifacts"], ylabel="False triggers / simulated hour", title="Controls with no injected bursts")
    axes[1].set_ylim(0,max(1,float(controls.false_triggers_per_simulated_hour.max())*1.3))
    axes[1].legend()
    fig.suptitle("Controlled simulation — frozen extraction and anomaly model", fontsize=14)
    fig.text(.5,.015,"4 independent trials per strength; 4 per control type. Simulation performance is not real-XSM accuracy.",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.94])
    fig.savefig(out / "simulation_results.png", dpi=160, bbox_inches="tight")
    print("\nSaved simulation outputs:", out)
    print("\nRecovery counts (do not interpret these as real-data accuracy):")
    print(recovery.to_string(index=False))
    return metrics, recovery, controls, fig
