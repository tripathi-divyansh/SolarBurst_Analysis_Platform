"""XGBoost models with explicit train/tune/calibration/threshold/test partitions."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import json
import platform
import joblib
import numpy as np
import pandas as pd
from .serialization import write_json
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from .features import FEATURES, FEATURE_VERSION, REGRESSION_FEATURES, REGRESSION_TARGETS
from .evaluation import classification_metrics, select_threshold, far_stats

SPLITS = ("train", "tune", "calibration", "threshold", "test")


@dataclass
class FeatureTransform:
    columns: list
    imputer: object = None

    def _array(self, frame):
        missing = set(self.columns)-set(frame)
        if missing: raise ValueError(f"Missing model features: {sorted(missing)}")
        x = frame.loc[:, self.columns].to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x

    def fit_transform(self, frame):
        x = self._array(frame)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        return np.column_stack([self.imputer.fit_transform(x), np.isnan(x).astype(float)])

    def transform(self, frame):
        if self.imputer is None: raise ValueError("Transformer is not fitted")
        x = self._array(frame)
        return np.column_stack([self.imputer.transform(x), np.isnan(x).astype(float)])


def validate_training_table(frame):
    required = set(FEATURES) | {"candidate_id", "group_id", "source_hash", "pipeline_hash", "feature_version", "quantity", "unit", "label", "split"}
    if required-set(frame): raise ValueError(f"Training table missing {sorted(required-set(frame))}")
    if frame.candidate_id.isna().any() or frame.candidate_id.duplicated().any():
        raise ValueError("candidate_id must be unique and nonmissing")
    for col in ("group_id", "source_hash", "split", "pipeline_hash", "quantity", "unit"):
        if frame[col].isna().any() or (frame[col].astype(str).str.len() == 0).any():
            raise ValueError(f"Missing {col}")
    if set(frame.split) != set(SPLITS):
        raise ValueError(f"Explicit partitions required: {SPLITS}")
    if not set(frame.label) <= {0, 1}:
        raise ValueError("Only reviewed 0/1 labels allowed; omit uncertain intervals from evaluation exposure")
    for col in ("group_id", "source_hash"):
        if (frame.groupby(col).split.nunique() > 1).any():
            raise ValueError(f"Leakage: {col} crosses partitions")
    for col in ("pipeline_hash", "feature_version", "quantity", "unit"):
        if frame[col].nunique() != 1:
            raise ValueError(f"Mixed {col}: train compatible observations/configurations separately")
    if frame.feature_version.iloc[0] != FEATURE_VERSION:
        raise ValueError("Unsupported feature version")
    for split in SPLITS:
        part = frame[frame.split == split]
        if part.group_id.nunique() < 2 or set(part.label) != {0, 1}:
            raise ValueError(f"{split}: at least two independent groups and both classes required")
    # Counts are engineering checks, not proof of adequate statistical sample size.


def validate_exposure(exposure, frame):
    needed = {"group_id", "split", "processed_exposure_s", "fully_reviewed"}
    if needed-set(exposure): raise ValueError(f"Exposure table missing {sorted(needed-set(exposure))}")
    if exposure.group_id.duplicated().any(): raise ValueError("Exposure groups must be unique")
    ex = exposure.copy()
    if not np.isfinite(ex.processed_exposure_s).all() or (ex.processed_exposure_s <= 0).any():
        raise ValueError("Exposure must be positive finite seconds")
    if not ex.fully_reviewed.map(lambda x: x is True or str(x).lower() == "true").all():
        raise ValueError("Operational thresholds require fully reviewed continuous intervals")
    mapping = dict(zip(ex.group_id, ex.split))
    if any(mapping.get(g) != sp for g, sp in frame[["group_id", "split"]].drop_duplicates().itertuples(index=False, name=None)):
        raise ValueError("Candidate and exposure groups/partitions disagree")
    if not set(ex.split) <= set(SPLITS): raise ValueError("Unknown exposure split")
    return {sp: float(ex.loc[ex.split == sp, "processed_exposure_s"].sum()) for sp in SPLITS}


def _classifier(seed, n_estimators, weight=1.0, multiclass=False):
    options = dict(n_estimators=n_estimators, max_depth=3, learning_rate=0.05,
                   min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=5, tree_method="hist", device="cpu",
                   n_jobs=2, random_state=seed, early_stopping_rounds=20)
    options.update(objective="multi:softprob" if multiclass else "binary:logistic",
                   eval_metric="mlogloss" if multiclass else "aucpr")
    if not multiclass: options["scale_pos_weight"] = weight
    return XGBClassifier(**options)


def _logit(score):
    s = np.clip(score, 1e-6, 1-1e-6)
    return np.log(s/(1-s)).reshape(-1, 1)


@dataclass
class BurstModels:
    transform: FeatureTransform
    burst: object
    calibrator: object
    threshold: float
    metadata: dict
    morphology: object = None
    morphology_classes: list | None = None
    regression_transform: FeatureTransform | None = None
    regressors: dict | None = None

    def score(self, frame):
        for col in ("pipeline_hash", "quantity", "unit"):
            if col not in frame or (len(frame) and not (frame[col] == self.metadata[col]).all()):
                raise ValueError(f"Inference {col} differs from training")
        if "feature_version" not in frame or (len(frame) and not (frame.feature_version == FEATURE_VERSION).all()):
            raise ValueError("Inference feature schema version mismatch")
        out = frame.copy()
        if not len(out): return out
        x = self.transform.transform(out)
        raw = self.burst.predict_proba(x)[:, 1]
        probability = self.calibrator.predict_proba(_logit(raw))[:, 1]
        out["burst_raw_score"] = raw
        out["burst_probability"] = probability
        out["probability_scope"] = self.metadata["domain_status"]
        out["status"] = np.where(probability >= self.threshold, "accepted", "review_below_threshold")
        accepted = probability >= self.threshold
        out["learned_morphology"] = "unavailable_or_not_accepted"
        out["morphology_score_uncalibrated"] = np.nan
        if self.morphology is not None and accepted.any():
            p = self.morphology.predict_proba(x[accepted])
            c = np.argmax(p, axis=1)
            out.loc[accepted, "learned_morphology"] = np.array(self.morphology_classes)[c]
            out.loc[accepted, "morphology_score_uncalibrated"] = p.max(axis=1)
        if self.regressors and accepted.any():
            xr = self.regression_transform.transform(out.loc[accepted])
            for key, reg in self.regressors.items():
                out.loc[accepted, "ml_"+key.removeprefix("target_")] = np.maximum(np.expm1(reg.predict(xr)), 0)
            # Correlated estimate discrepancy, not independent evidence of truth.
            if "ml_duration_s" in out:
                ratio = out.ml_duration_s/out.fit_duration_s.replace(0, np.nan)
                out["duration_crosscheck_flag"] = (ratio > 2) | (ratio < 0.5)
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        # Companion human-readable provenance. Joblib must only load trusted artifacts.
        write_json(path.with_suffix(".json"), self.metadata)

    @classmethod
    def load(cls, path, trusted=False):
        if not trusted:
            raise ValueError("Serialized models can execute code; explicitly confirm trusted=True")
        model = joblib.load(path)
        if not isinstance(model, cls): raise ValueError("Not an XSM model bundle")
        if model.metadata["feature_version"] != FEATURE_VERSION: raise ValueError("Model schema mismatch")
        return model


def train_models(frame, exposure, target_far=1.0, n_estimators=400, seed=42,
                 class_weight=False, domain_status="unvalidated_real_domain"):
    """Run once after fixing design; do not repeatedly tune against returned test metrics."""
    validate_training_table(frame)
    duration = validate_exposure(exposure, frame)
    parts = {sp: frame.loc[frame.split == sp].copy() for sp in SPLITS}
    transform = FeatureTransform(FEATURES.copy())
    matrices = {"train": transform.fit_transform(parts["train"])}
    matrices.update({sp: transform.transform(parts[sp]) for sp in SPLITS if sp != "train"})
    ys = {sp: parts[sp].label.to_numpy(int) for sp in SPLITS}
    weight = float(np.sum(ys["train"] == 0)/np.sum(ys["train"] == 1)) if class_weight else 1.0
    model = _classifier(seed, n_estimators, weight)
    model.fit(matrices["train"], ys["train"], eval_set=[(matrices["tune"], ys["tune"])], verbose=False)
    cal_raw = model.predict_proba(matrices["calibration"])[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
    calibrator.fit(_logit(cal_raw), ys["calibration"])
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Calibration reverses burst ranking: inspect label quality/domain shift before deploying")
    th = calibrator.predict_proba(_logit(model.predict_proba(matrices["threshold"])[:, 1]))[:, 1]
    threshold, threshold_stats = select_threshold(ys["threshold"], th, duration["threshold"], target_far)
    meta = {"feature_version": FEATURE_VERSION, "features": FEATURES,
            "pipeline_hash": frame.pipeline_hash.iloc[0], "quantity": frame.quantity.iloc[0], "unit": frame.unit.iloc[0],
            "seed": seed, "domain_status": domain_status, "threshold": threshold,
            "threshold_target_far": target_far, "threshold_validation": threshold_stats,
            "probability_method": "sigmoid_on_separate_observing_periods",
            "evaluation_scope": "candidate_labels_only; full detection recall requires reference-event matching",
            "python": platform.python_version(),
            "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost-cpu", "joblib")},
            "partition_counts": {sp: {"candidates": len(parts[sp]), "groups": parts[sp].group_id.nunique(),
                                        "positive": int(ys[sp].sum()), "exposure_s": duration[sp]} for sp in SPLITS},
            "morphology_status": "no_supervised_labels", "regression_status": "no_valid_targets"}
    bundle = BurstModels(transform, model, calibrator, threshold, meta)
    # Morphology requires explicit reviewed positive labels; fallback rules are separate.
    if "morphology_label" in frame:
        tr = parts["train"].loc[(parts["train"].label == 1) & parts["train"].morphology_label.notna()]
        va = parts["tune"].loc[(parts["tune"].label == 1) & parts["tune"].morphology_label.notna()]
        classes = sorted(tr.morphology_label.unique())
        if len(classes) >= 2 and set(va.morphology_label) == set(classes):
            mapping = {name: i for i, name in enumerate(classes)}
            mc = _classifier(seed, n_estimators, multiclass=len(classes) > 2)
            mc.set_params(eval_metric="mlogloss" if len(classes) > 2 else "logloss")
            mc.fit(transform.transform(tr), tr.morphology_label.map(mapping).to_numpy(int),
                   eval_set=[(transform.transform(va), va.morphology_label.map(mapping).to_numpy(int))], verbose=False)
            bundle.morphology, bundle.morphology_classes = mc, classes
            meta["morphology_status"] = "trained_uncalibrated_class_scores"
    # Regression excludes fitted features and only accepts targets explicitly marked reliable.
    if set(REGRESSION_TARGETS+["regression_target_reliable"]) <= set(frame):
        masks = {}
        for sp in ("train", "tune"):
            a = parts[sp]
            target = a[REGRESSION_TARGETS].to_numpy(float)
            masks[sp] = (a.label.to_numpy() == 1) & (a.regression_target_reliable.to_numpy() == True) & np.isfinite(target).all(axis=1) & (target > 0).all(axis=1)
        if masks["train"].sum() >= 20 and masks["tune"].sum() >= 5:
            rt = FeatureTransform(REGRESSION_FEATURES.copy())
            a, b = parts["train"].loc[masks["train"]], parts["tune"].loc[masks["tune"]]
            ra, rb = rt.fit_transform(a), rt.transform(b)
            bundle.regression_transform, bundle.regressors = rt, {}
            for key in REGRESSION_TARGETS:
                reg = XGBRegressor(n_estimators=n_estimators, max_depth=3, learning_rate=0.05,
                                   min_child_weight=3, reg_lambda=5, tree_method="hist", device="cpu",
                                   n_jobs=2, random_state=seed, early_stopping_rounds=20,
                                   objective="reg:squarederror", eval_metric="rmse")
                reg.fit(ra, np.log1p(a[key]), eval_set=[(rb, np.log1p(b[key]))], verbose=False)
                bundle.regressors[key] = reg
            meta["regression_status"] = "trained_correlated_crosscheck_not_independent_measurement"
    # Open locked test only after training, calibration and threshold have finished.
    raw_test = model.predict_proba(matrices["test"])[:, 1]
    test_prob = calibrator.predict_proba(_logit(raw_test))[:, 1]
    report = classification_metrics(ys["test"], test_prob, threshold)
    report.update(far_stats(report["fp"], duration["test"]))
    report["uncalibrated_brier"] = classification_metrics(ys["test"], raw_test)["brier"]
    report["scope"] = meta["evaluation_scope"]
    report["domain_status"] = domain_status
    if bundle.morphology is not None:
        part = parts["test"]
        selected = (part.label == 1) & part.morphology_label.isin(bundle.morphology_classes)
        if selected.any():
            pred = bundle.morphology.predict(transform.transform(part.loc[selected])).astype(int)
            names = np.array(bundle.morphology_classes)[pred]
            truth = part.loc[selected, "morphology_label"]
            report["morphology_on_labeled_positive_candidates"] = {
                "n": int(selected.sum()), "accuracy": float(accuracy_score(truth, names)),
                "macro_f1": float(f1_score(truth, names, average="macro", zero_division=0)),
                "probabilities_calibrated": False}
    if bundle.regressors:
        part = parts["test"]
        targets = part[REGRESSION_TARGETS].to_numpy(float)
        selected = (part.label.to_numpy() == 1) & (part.regression_target_reliable.to_numpy() == True) & np.isfinite(targets).all(axis=1) & (targets > 0).all(axis=1)
        report["regression_on_reliable_positive_candidates"] = {}
        if selected.any():
            x = bundle.regression_transform.transform(part.loc[selected])
            for key, reg in bundle.regressors.items():
                prediction = np.maximum(np.expm1(reg.predict(x)), 0)
                truth = part.loc[selected, key].to_numpy(float)
                report["regression_on_reliable_positive_candidates"][key] = {
                    "n": int(selected.sum()), "mae": float(mean_absolute_error(truth, prediction)),
                    "rmse": float(np.sqrt(mean_squared_error(truth, prediction)))}
    meta["locked_test_candidate_metrics"] = report
    return bundle, report
