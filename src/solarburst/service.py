"""
FastAPI scientific engine microservice for local high-performance IPC with Node.js backend.
"""

from typing import Dict, Any, Optional, List
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core.schema import AnalysisConfig
from .io.xsm_adapter import inspect_file, load_lightcurve
from .detection.wavelet_cwt import compute_scalogram
from .engine import SolarBurstEngine
from .ml.models import train_candidate_models
from .ml.persistence import save_classifier, load_classifier
from .evaluation.injection import create_synthetic_demo_lightcurve

api_app = FastAPI(
    title="SolarBurst Scientific Service",
    description="Python scientific analysis engine for ISRO XSM Solar Burst Detection",
    version="1.0.0"
)

# Enable CORS for local Node.js backend communication
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SolarBurstEngine()


class InspectRequest(BaseModel):
    filepath: str


class AnalyzeRequest(BaseModel):
    filepath: str
    preset: str = "balanced"
    model_family: str = "random_forest"
    asls_lambda: float = 1e6
    asls_p: float = 0.01
    accept_threshold: float = 0.65
    review_threshold: float = 0.35
    max_components: int = 3
    fast_uncertainty: bool = True
    column_mapping: Optional[Dict[str, str]] = None


class WaveletRequest(BaseModel):
    filepath: str
    start_met: Optional[float] = None
    end_met: Optional[float] = None
    wavelet: str = "morlet"
    num_scales: int = 40


class TrainRequest(BaseModel):
    n_synthetic_samples: int = 150
    output_dir: str = "./models/pretrained"


@api_app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "SolarBurst",
        "version": "1.0.0",
        "python_env": "Python 3.14 cp314-win_amd64",
        "scientific_stack": {
            "numpy": np.__version__,
            "scipy": "available",
            "astropy": "available",
            "sklearn": "available",
            "xgboost": "available",
            "pywt": "available"
        }
    }


@api_app.post("/inspect")
def inspect_dataset(req: InspectRequest):
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {req.filepath}")
    try:
        return inspect_file(req.filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def clean_json_floats(obj):
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json_floats(v) for v in obj]
    return obj


@api_app.post("/analyze")
def run_analysis(req: AnalyzeRequest):
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {req.filepath}")

    try:
        # Load light curve with optional column mapping
        options = req.column_mapping or {}
        lc = load_lightcurve(req.filepath, options=options)

        # Assemble configuration
        cfg = AnalysisConfig(
            name=req.preset,
            model_family=req.model_family,
            asls_lambda=req.asls_lambda,
            asls_p=req.asls_p,
            ml_accept_threshold=req.accept_threshold,
            ml_review_threshold=req.review_threshold,
            max_components=req.max_components,
            fast_uncertainty=req.fast_uncertainty
        )

        # Run analysis
        result = engine.run_analysis(lc, cfg)
        return clean_json_floats(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api_app.post("/wavelet")
def compute_wavelet_scalogram(req: WaveletRequest):
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {req.filepath}")

    try:
        lc = load_lightcurve(req.filepath)
        time_met = lc.met_seconds
        rate = lc.value

        # Slice to requested time window
        if req.start_met is not None and req.end_met is not None:
            mask = (time_met >= req.start_met) & (time_met <= req.end_met) & lc.valid_mask
        else:
            # Default to first 1800 seconds
            mask = (time_met <= time_met[0] + 1800.0) & lc.valid_mask

        sub_t = time_met[mask]
        sub_r = rate[mask]

        if len(sub_t) < 10:
            raise HTTPException(status_code=400, detail="Window has fewer than 10 valid points.")

        # Subsample if too dense for web payload (max 800 points)
        if len(sub_t) > 800:
            step = int(np.ceil(len(sub_t) / 800))
            sub_t = sub_t[::step]
            sub_r = sub_r[::step]

        scalogram = compute_scalogram(
            signal=sub_r,
            time_sec=sub_t,
            wavelet=req.wavelet,
            num_scales=req.num_scales
        )
        return scalogram
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.post("/demo")
def generate_demo():
    """Generate synthetic demonstration observation and run analysis."""
    try:
        lc, injected = create_synthetic_demo_lightcurve(duration_s=7200.0)
        cfg = AnalysisConfig(name="balanced")
        result = engine.run_analysis(lc, cfg)
        result["injected_ground_truth"] = injected
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.post("/train")
def train_models(req: TrainRequest):
    """
    Train Random Forest and XGBoost candidate scoring models on synthetic injection partition.
    """
    try:
        # Generate training observation and extraction
        lc_train, inj_train = create_synthetic_demo_lightcurve(duration_s=14400.0, rng_seed=101)
        lc_val, inj_val = create_synthetic_demo_lightcurve(duration_s=7200.0, rng_seed=202)

        # Preprocess and detect proposals on both splits
        from .preprocess.binning import regularize_detection_series
        from .preprocess.baseline import iterative_masked_baseline
        from .detection.proposals import generate_candidate_proposals
        from .ml.features import extract_features_table
        from .evaluation.matching import match_events_one_to_one

        def get_split_features(lc, inj):
            reg = regularize_detection_series(lc)
            base_res = iterative_masked_baseline(reg["value_grid"], reg["mask_valid"])
            cands = generate_candidate_proposals(
                reg["time_grid_met"], reg["value_grid"],
                base_res["baseline"], base_res["noise_sigma"], reg["target_dt_s"],
                matched_filter_thresh=2.5, cwt_thresh=2.0, hysteresis_seed=3.0
            )
            X, _ = extract_features_table(
                cands, reg["time_grid_met"], reg["value_grid"],
                base_res["baseline"], base_res["noise_sigma"], reg["target_dt_s"]
            )
            # Label candidates based on overlap with injected truth
            y = np.zeros(len(cands), dtype=int)
            for i, c in enumerate(cands):
                for fl in inj:
                    if abs(c.peak_time_met - fl["peak_time_met"]) < 60.0:
                        y[i] = 1
                        break
            return X, y

        X_train, y_train = get_split_features(lc_train, inj_train)
        X_val, y_val = get_split_features(lc_val, inj_val)

        # Train both models
        res = train_candidate_models(X_train, y_train, X_val, y_val)
        
        # Save both models
        os.makedirs(req.output_dir, exist_ok=True)
        rf_meta = save_classifier(res["rf_wrapper"], req.output_dir, "rf_baseline")
        xgb_meta = save_classifier(res["xgb_wrapper"], req.output_dir, "xgb_challenger")

        # Set active model in engine
        engine.classifier = res["rf_wrapper"]

        return {
            "status": "trained_successfully",
            "n_train_samples": len(y_train),
            "n_val_samples": len(y_val),
            "train_positives": int(np.sum(y_train == 1)),
            "train_negatives": int(np.sum(y_train == 0)),
            "val_positives": int(np.sum(y_val == 1)),
            "val_negatives": int(np.sum(y_val == 0)),
            "rf_artifact": rf_meta,
            "xgb_artifact": xgb_meta
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
