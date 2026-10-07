"""
High-level end-to-end SolarBurst analysis pipeline orchestrator.
Powers the CLI, background jobs, and Node.js API.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from .core.schema import LightCurve, Candidate, BurstResult, AnalysisConfig
from .preprocess.gti import compute_coverage_summary
from .preprocess.binning import regularize_detection_series
from .preprocess.noise import estimate_robust_noise, hampel_spike_screen, compute_standardized_series
from .preprocess.baseline import iterative_masked_baseline
from .detection.proposals import generate_candidate_proposals
from .ml.features import extract_features_table
from .ml.models import BurstClassifier
from .fitting.emg_fit import fit_emg_profile
from .fitting.multi_component import fit_multicomponent_emg
from .catalog.parameters import synthesize_burst_result


class SolarBurstEngine:
    """
    Unified analysis engine for solar X-ray light curves.
    """
    def __init__(self, classifier: Optional[BurstClassifier] = None):
        if classifier is None:
            import os
            from .ml.persistence import load_classifier
            for path_try in [
                "./models/pretrained/rf_baseline.joblib",
                "../models/pretrained/rf_baseline.joblib",
                "models/pretrained/rf_baseline.joblib",
                os.path.join(os.path.dirname(__file__), "../../../models/pretrained/rf_baseline.joblib")
            ]:
                if os.path.exists(path_try):
                    try:
                        classifier = load_classifier(path_try)
                        break
                    except Exception:
                        pass
        self.classifier = classifier

    def run_analysis(
        self,
        lc: LightCurve,
        config: Optional[AnalysisConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute full scientific pipeline:
        1. Preprocessing & regularized detection series
        2. Iterative masked baseline estimation
        3. Multiscale proposal generation (matched filter + CWT + derivative)
        4. Standardized feature extraction
        5. ML Candidate scoring (Random Forest or XGBoost)
        6. Physical profile fitting (EMG convolution & joint baseline)
        7. Classifications, uncertainties, and catalog synthesis
        """
        start_wall_time = time.time()
        config = config or AnalysisConfig()

        if progress_callback:
            progress_callback(10, "Preprocessing and checking GTIs...")

        # 1. Preprocessing & Coverage
        coverage = compute_coverage_summary(lc)
        reg = regularize_detection_series(lc)
        
        t_grid = reg["time_grid_met"]
        val_grid = reg["value_grid"]
        err_grid = reg["error_grid"]
        mask_valid = reg["mask_valid"]
        dt_s = reg["target_dt_s"]

        if np.sum(mask_valid) < 20:
            raise ValueError("Insufficient valid observation points for analysis (<20 points).")

        if progress_callback:
            progress_callback(25, "Estimating robust iterative baseline (AsLS)...")

        # 2. Baseline and Noise
        base_res = iterative_masked_baseline(
            val_grid,
            mask_valid,
            lam=config.asls_lambda,
            p=config.asls_p,
            iterations=config.baseline_iterations
        )
        baseline = base_res["baseline"]
        noise_sigma = base_res["noise_sigma"]

        # Flag spikes
        spikes = hampel_spike_screen(val_grid, mask_valid)

        if progress_callback:
            progress_callback(45, "Running multiscale candidate detectors...")

        # 3. Candidate Proposals
        candidates = generate_candidate_proposals(
            time_met=t_grid,
            rate=val_grid,
            baseline=baseline,
            noise_sigma=noise_sigma,
            dt_s=dt_s,
            matched_filter_thresh=config.matched_filter_snr_threshold,
            cwt_thresh=config.cwt_snr_threshold,
            derivative_thresh=config.derivative_threshold,
            hysteresis_seed=config.hysteresis_seed_sigma,
            hysteresis_grow=config.hysteresis_grow_sigma
        )

        if progress_callback:
            progress_callback(60, f"Extracting candidate features ({len(candidates)} candidates)...")

        # 4. Feature Extraction & ML Scoring
        X_features, feature_names = extract_features_table(
            candidates,
            time_met=t_grid,
            rate=val_grid,
            baseline=baseline,
            noise_sigma=noise_sigma,
            dt_s=dt_s
        )

        if len(candidates) > 0:
            if config.model_family != "classical_only" and self.classifier is not None:
                scores, decisions = self.classifier.predict_decisions(X_features)
                for idx, cand in enumerate(candidates):
                    cand.ml_score = float(scores[idx])
                    cand.decision = decisions[idx]
            else:
                # Classical scoring based on peak SNR
                for cand in candidates:
                    score = float(1.0 / (1.0 + np.exp(-0.5 * (cand.seed_snr - 4.0))))
                    cand.ml_score = score
                    if score >= config.ml_accept_threshold:
                        cand.decision = "accepted"
                    elif score >= config.ml_review_threshold:
                        cand.decision = "review"
                    else:
                        cand.decision = "rejected"

        if progress_callback:
            progress_callback(75, "Fitting physical profiles and computing uncertainties...")

        # 5. Physical Profile Fitting for Non-Rejected Candidates
        burst_results: List[BurstResult] = []
        lo_grid = t_grid - dt_s / 2.0
        hi_grid = t_grid + dt_s / 2.0

        fitting_candidates = [c for c in candidates if c.decision != "rejected"]
        if len(fitting_candidates) > 60:
            fitting_candidates.sort(key=lambda c: c.seed_snr, reverse=True)
            fitting_candidates = fitting_candidates[:60]
            fitting_candidates.sort(key=lambda c: c.start_time_met)

        for idx, cand in enumerate(fitting_candidates):
            # Extract candidate context slice
            mask_ctx = (t_grid >= cand.context_start_met) & (t_grid <= cand.context_end_met) & mask_valid
            sub_lo = lo_grid[mask_ctx]
            sub_hi = hi_grid[mask_ctx]
            sub_y = val_grid[mask_ctx]
            sub_err = err_grid[mask_ctx]

            if len(sub_y) < 8:
                continue

            try:
                # Joint profile fit
                fit_res = fit_multicomponent_emg(
                    sub_lo, sub_hi, sub_y, sub_err,
                    max_components=config.max_components
                )
                
                burst = synthesize_burst_result(
                    cand=cand,
                    fit_res=fit_res,
                    time_met=t_grid[mask_ctx],
                    lc=lc,
                    fast_uncertainty=config.fast_uncertainty
                )
                burst_results.append(burst)
            except Exception as e:
                # Flag fit failure
                continue

        if progress_callback:
            progress_callback(95, "Synthesizing final event catalog...")

        total_runtime_s = time.time() - start_wall_time

        # Downsample light curve for responsive web display (max 4000 points)
        plot_dict = lc.to_dict(max_points=4000)

        # Add downsampled baseline to plot_dict
        step = plot_dict["step"]
        sub_indices = np.arange(0, len(baseline), step)
        plot_dict["baseline"] = [float(b) if np.isfinite(b) else None for b in baseline[sub_indices]]

        return {
            "status": "success",
            "runtime_s": total_runtime_s,
            "coverage": coverage,
            "noise_sigma": float(noise_sigma),
            "total_candidates": len(candidates),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "start_time_met": float(c.start_time_met) if np.isfinite(c.start_time_met) else 0.0,
                    "peak_time_met": float(c.peak_time_met) if np.isfinite(c.peak_time_met) else 0.0,
                    "end_time_met": float(c.end_time_met) if np.isfinite(c.end_time_met) else 0.0,
                    "seed_snr": float(c.seed_snr) if np.isfinite(c.seed_snr) else 0.0,
                    "ml_score": float(c.ml_score) if np.isfinite(c.ml_score) else 0.0,
                    "decision": c.decision,
                    "generator_flags": c.generator_flags
                }
                for c in candidates
            ],
            "total_bursts": len(burst_results),
            "bursts": [b.to_dict() for b in burst_results],
            "summary": {
                "accepted_count": sum(1 for b in burst_results if b.decision == "accepted"),
                "review_count": sum(1 for b in burst_results if b.decision == "review"),
                "rejected_count": sum(1 for b in candidates if b.decision == "rejected"),
                "reliable_count": sum(1 for b in burst_results if b.reliability == "reliable"),
                "poor_fit_count": sum(1 for b in burst_results if b.reliability == "poor_fit"),
                "truncated_count": sum(1 for b in burst_results if b.reliability == "truncated"),
            },
            "lightcurve": plot_dict,
            "config": {
                "preset": config.name,
                "model_family": config.model_family,
                "fit_profile": config.fit_profile,
                "asls_lambda": config.asls_lambda,
                "asls_p": config.asls_p,
                "accept_threshold": config.ml_accept_threshold,
                "review_threshold": config.ml_review_threshold,
            }
        }
