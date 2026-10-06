"""
End-to-end pipeline integration tests on synthetic demonstration observations.
"""

import os
import pytest
from solarburst.core.schema import AnalysisConfig
from solarburst.io.xsm_adapter import load_lightcurve
from solarburst.engine import SolarBurstEngine
from solarburst.evaluation.injection import create_synthetic_demo_lightcurve
from solarburst.evaluation.matching import match_events_one_to_one
from solarburst.evaluation.metrics import compute_detection_metrics


def test_end_to_end_analysis():
    # 1. Create synthetic demo observation with 4 injected bursts
    lc, injected = create_synthetic_demo_lightcurve(duration_s=3600.0, rng_seed=42)

    # 2. Execute SolarBurstEngine
    engine = SolarBurstEngine()
    cfg = AnalysisConfig(name="balanced")
    results = engine.run_analysis(lc, cfg)

    assert results["status"] == "success"
    assert results["total_candidates"] > 0
    assert results["total_bursts"] > 0
    assert len(results["bursts"]) == results["total_bursts"]

    # 3. Match against ground truth
    from solarburst.core.schema import BurstResult
    burst_objs = [BurstResult(**b) for b in results["bursts"]]

    match_res = match_events_one_to_one(burst_objs, injected, tolerance_sec=60.0)
    assert len(match_res["matched_pairs"]) >= 2  # Recovers at least the strong and impulsive bursts

    # 4. Metrics
    metrics = compute_detection_metrics(match_res, valid_exposure_s=3600.0)
    assert metrics["precision"] > 0.0
    assert metrics["recall"] > 0.0
    assert "far_per_24h" in metrics
