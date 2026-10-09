"""Synthetic demonstration only; never a claim of real-XSM validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import erf
from .schema import LightCurve, PipelineConfig
from .physics import fred_bin_mean, fred_parameters
from .pipeline import analyze
from .evaluation import match_events, far_stats
from .models import train_models
from .serialization import write_json, save_analysis


def synthetic_observation(seed=42, n=1000):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)+0.5
    width = np.ones(n)
    noise = rng.uniform(1.5, 3.0)
    background = rng.uniform(15, 25) + rng.uniform(-0.001, 0.001)*t
    y = background+rng.normal(0, noise, n)
    truth = []
    # Vary amplitude/time independently across all observing-period partitions.
    onset = rng.uniform(130, 160)
    amplitude, tr, td = rng.uniform(35, 80), rng.uniform(3, 12), rng.uniform(15, 35)
    y += fred_bin_mean(t-0.5, t+0.5, amplitude, onset, tr, td)
    pars = fred_parameters(amplitude, onset, tr, td)
    truth.append({"observed_start_s": pars["start_5pct_s"], "observed_peak_s": pars["peak_s"],
                  "observed_end_s": pars["end_5pct_s"], "morphology_label": "fast_rise_slow_decay",
                  "target_rise_s": pars["rise_duration_s"], "target_decay_constant_s": td,
                  "target_duration_s": pars["duration_s"], "target_net_peak": pars["net_peak"],
                  "regression_target_reliable": True})
    # Different signal family prevents testing only the fitted FRED template.
    center, sigma, peak = rng.uniform(490, 530), rng.uniform(8, 18), rng.uniform(15, 45)
    y += peak*sigma*np.sqrt(np.pi/2)*(erf((t+0.5-center)/(np.sqrt(2)*sigma))-erf((t-0.5-center)/(np.sqrt(2)*sigma)))
    w = sigma*np.sqrt(2*np.log(20))
    truth.append({"observed_start_s": center-w, "observed_peak_s": center,
                  "observed_end_s": center+w, "morphology_label": "approximately_symmetric",
                  "target_rise_s": np.nan, "target_decay_constant_s": np.nan,
                  "target_duration_s": np.nan, "target_net_peak": np.nan,
                  "regression_target_reliable": False})
    # Known simulated instrument artifacts, not arbitrary unusual solar morphologies.
    y[350] += rng.uniform(30, 90)
    y[760:775] += rng.uniform(10, 30)
    lc = LightCurve(t, y, np.full(n, noise), width, width.copy(), np.zeros(n, int),
                    {"quantity": "count_rate", "unit": "count / s", "epoch": "relative",
                     "time_scale": "relative", "instrument": "synthetic_demo",
                     "synthetic": True, "generator_seed": seed})
    return lc, pd.DataFrame(truth)


def run_demo(directory, groups=40, seed=42):
    if groups < 30:
        raise ValueError("Use at least 30 independent synthetic periods for the five-way demo split")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    # Split original observing periods before candidate extraction or augmentation.
    splits = (["train"]*int(groups*0.4)+["tune"]*int(groups*0.15)+
              ["calibration"]*int(groups*0.15)+["threshold"]*int(groups*0.15))
    splits += ["test"]*(groups-len(splits))
    tables, exposures, references = [], [], []
    cfg = PipelineConfig()
    for i, split in enumerate(splits):
        lc, truth = synthetic_observation(seed+1009*i)
        group = f"synthetic_period_{i:03d}"
        truth["group_id"], truth["split"] = group, split
        result = analyze(lc, cfg, group_id=group)
        table = result["catalog"]
        table["label"], table["split"] = 0, split
        table["morphology_label"] = pd.Series([None]*len(table), dtype=object)
        table["regression_target_reliable"] = False
        for name in ("target_rise_s", "target_decay_constant_s", "target_duration_s", "target_net_peak"):
            table[name] = np.nan
        match = match_events(table, truth, tolerance_s=35)
        for pi, ri in match["pairs"]:
            table.loc[pi, "label"] = 1
            for name in ("morphology_label", "regression_target_reliable", "target_rise_s", "target_decay_constant_s", "target_duration_s", "target_net_peak"):
                table.loc[pi, name] = truth.iloc[ri][name]
        tables.append(table)
        exposures.append({"group_id": group, "split": split,
                          "processed_exposure_s": result["manifest"]["processed_exposure_s"], "fully_reviewed": True})
        references.append(truth)
        if i == 0:
            pd.DataFrame({"time": lc.time, "rate": lc.value, "error": lc.error,
                          "width": lc.bin_width, "exposure": lc.exposure, "quality": lc.quality}).to_csv(out/"example_observation.csv", index=False)
            write_json(out/"example_mapping.json", {"time": "time", "value": "rate", "error": "error",
                       "bin_width": "width", "exposure": "exposure", "quality": "quality",
                       "quantity": "count_rate", "unit": "count / s", "epoch": "relative",
                       "time_scale": "relative", "time_unit_s": 1, "time_position": "center",
                       "instrument": "synthetic_demo"})
            save_analysis(result, out/"example_analysis")
        print(f"Synthetic period {i+1}/{groups}: {len(table)} candidates", flush=True)
    frame, exposure, reference = pd.concat(tables, ignore_index=True), pd.DataFrame(exposures), pd.concat(references, ignore_index=True)
    frame.to_csv(out/"training_candidates.csv", index=False)
    exposure.to_csv(out/"exposure.csv", index=False)
    reference.to_csv(out/"reference_events.csv", index=False)
    bundle, report = train_models(frame, exposure, target_far=1.0, n_estimators=250,
                                  seed=seed, domain_status="SYNTHETIC_DEMO_ONLY")
    bundle.save(out/"synthetic_demo_model.joblib")
    test = frame[frame.split == "test"]
    scored = bundle.score(test)
    scored.to_csv(out/"test_predictions.csv", index=False)
    ref = reference[reference.split == "test"]
    full = match_events(scored[scored.status == "accepted"], ref, tolerance_s=35)
    full.pop("pairs")
    full.update(far_stats(full["fp"], exposure.loc[exposure.split == "test", "processed_exposure_s"].sum()))
    proposed = match_events(test, ref, tolerance_s=35)
    report = {"domain": "SYNTHETIC_DEMO_ONLY", "candidate_classification": report,
              "full_pipeline_detection": full, "proposal_recall": proposed["recall"],
              "warning": "Small synthetic demonstration, not real-XSM accuracy or an operational FAR guarantee."}
    write_json(out/"demo_metrics.json", report)
    return report
