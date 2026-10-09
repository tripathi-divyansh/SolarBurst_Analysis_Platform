"""XSM review and preliminary three-way training, for the EXISTING notebook.

Quick start (after copying this file into PROJECT):
    import sys
    sys.path.insert(0, str(PROJECT))
    from xsm_review_training import ReviewSession
    review = ReviewSession(BATCH_RUN, review_folder)
    review.show()

After completing independent annotation:
    review.status()
    model_folder = review.train()

This add-on does not change the original feature CSVs or the strict five-way
production trainer. It trains a preliminary candidate classifier, not a solar
forecast, morphology classifier, physical-parameter regressor, or calibrated
operational detector. Candidate-level recall cannot measure missed events.
All code-generated scores and fit flags are evidence to inspect, not labels.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os

import numpy as np
import pandas as pd

VERSION = "xsm-review-baseline-1.0"
SPLIT_PLAN = {
    "20260716": "train", "20260913": "train",
    "20260916": "train", "20260917": "train",
    "20260919": "validation", "20260920": "test",
}
FEATURES = [
    "peak_snr", "net_peak_observed", "background_at_peak", "peak_background_ratio",
    "rise_observed_s", "decay_observed_s", "duration_observed_s", "fwhm_observed_s",
    "fluence_observed", "max_rise_rate", "max_decay_rate", "skewness", "kurtosis",
    "wavelet_energy", "wavelet_max", "dominant_scale_s", "n_peaks", "cadence_s",
    "sigma_evidence", "wavelet_evidence", "bb_evidence", "edge_candidate",
    "fit_success", "fit_r2", "fit_rmse_over_noise", "fit_reduced_statistic",
    "fit_rise_constant_s", "fit_decay_constant_s", "fit_duration_s", "fit_net_peak",
    "fit_residual_autocorrelation", "fit_censored",
]
ANNOTATION_COLUMNS = ["candidate_id", "decision", "evidence", "reviewer", "reviewed_at"]
DECISIONS = {"unreviewed", "positive", "negative", "uncertain"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def validate_inputs(frame, groups):
    required = set(FEATURES) | {"candidate_id", "group_id", "source_hash", "pipeline_hash",
                              "feature_version", "quantity", "unit", "label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Candidate table is missing: {sorted(missing)}")
    needed = {"group_id", "candidate_count", "processed_exposure_s", "fully_reviewed"}
    if needed - set(groups):
        raise ValueError(f"Observation table is missing: {sorted(needed-set(groups))}")
    for table, key in [(frame, "candidate_id"), (groups, "group_id")]:
        if table[key].isna().any() or table[key].astype(str).str.strip().eq("").any() or table[key].duplicated().any():
            raise ValueError(f"Missing or duplicate {key}")
    if set(groups.group_id) != set(SPLIT_PLAN) or set(frame.group_id) != set(SPLIT_PLAN):
        raise ValueError("This split is for the six reviewed dataset dates only. Keep new dates separate until a new plan is defined.")
    actual = frame.groupby("group_id").size().sort_index()
    expected = groups.set_index("group_id").candidate_count.sort_index()
    if not np.array_equal(actual.to_numpy(), expected.to_numpy()):
        raise ValueError("Candidate counts disagree with observation_groups.csv")
    for col in ["pipeline_hash", "feature_version", "quantity", "unit"]:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any() or frame[col].nunique() != 1:
            raise ValueError(f"Missing or mixed {col}")
    if frame.feature_version.iloc[0] != "fred-features-2":
        raise ValueError("Feature schema differs from the reviewed v0.2 package")
    if frame.source_hash.isna().any() or frame.source_hash.astype(str).str.strip().eq("").any():
        raise ValueError("Missing source hash")
    if (frame.groupby("source_hash").group_id.nunique() > 1).any():
        raise ValueError("Same source appears in different observation groups")
    if (frame.groupby("group_id").source_hash.nunique() != 1).any():
        raise ValueError("Multiple source versions in one observation group")
    if not np.isfinite(groups.processed_exposure_s).all() or (groups.processed_exposure_s <= 0).any():
        raise ValueError("Invalid exposure seconds")
    for col in FEATURES:
        values = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {col}")
    known = frame.label.dropna()
    if not known.isin([0, 1]).all():
        raise ValueError("Existing labels must be blank, 0, or 1")


class ReviewSession:
    def __init__(self, batch_run, review_folder):
        self.batch = Path(batch_run).resolve()
        self.folder = Path(review_folder).resolve()
        self.candidate_path = self.folder / "training_candidates.csv"
        self.group_path = self.folder / "observation_groups.csv"
        if not self.candidate_path.is_file() or not self.group_path.is_file():
            raise FileNotFoundError("Use the review_folder created by cell 5, containing both CSVs.")
        self.frame = pd.read_csv(self.candidate_path, dtype={"candidate_id": str, "group_id": str})
        self.groups = pd.read_csv(self.group_path, dtype={"group_id": str})
        validate_inputs(self.frame, self.groups)
        self.frame["split"] = self.frame.group_id.map(SPLIT_PLAN)
        self.work = self.folder / "review_baseline"
        self.work.mkdir(exist_ok=True)
        self.annotation_path = self.work / "annotations.csv"
        self.fingerprint = {
            "version": VERSION,
            "candidate_csv_sha256": _sha(self.candidate_path),
            "group_csv_sha256": _sha(self.group_path),
            "split_plan": SPLIT_PLAN,
            "threshold": 0.5,
            "selection": "validation average precision; ties prefer random forest",
        }
        manifest = self.work / "review_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.fingerprint:
                raise ValueError("Inputs or split plan changed. Existing reviews are protected; use a separate review folder.")
        else:
            _json(manifest, self.fingerprint)
        if not self.annotation_path.exists():
            a = pd.DataFrame({"candidate_id": self.frame.candidate_id,
                              "decision": self.frame.label.map({0: "negative", 1: "positive"}).fillna("unreviewed"),
                              "evidence": self.frame.get("review_notes", pd.Series("", index=self.frame.index)).fillna(""),
                              "reviewer": "", "reviewed_at": ""})
            _save_csv(a, self.annotation_path)
        plan = self.groups.copy()
        plan["split"] = plan.group_id.map(SPLIT_PLAN)
        _save_csv(plan, self.work / "split_plan.csv")
        self._read_annotations()
        self.plot_index = {}
        for path in sorted(self.batch.glob("*/review_candidates.csv")):
            table = pd.read_csv(path, dtype={"candidate_id": str})
            for i, row in table.iterrows():
                cid = row.candidate_id
                if cid not in set(self.frame.candidate_id):
                    continue
                match = self.frame.loc[self.frame.candidate_id == cid].iloc[0]
                if row.source_hash != match.source_hash or row.pipeline_hash != match.pipeline_hash:
                    raise ValueError("Review image provenance does not match candidate table")
                if cid in self.plot_index:
                    raise ValueError(f"Duplicate review products for {cid}")
                self.plot_index[cid] = (path.parent, i + 1)

    def _read_annotations(self):
        a = pd.read_csv(self.annotation_path, dtype=str, keep_default_na=False)
        if set(a.columns) != set(ANNOTATION_COLUMNS) or a.candidate_id.duplicated().any() or set(a.candidate_id) != set(self.frame.candidate_id):
            raise ValueError("Annotation IDs/schema do not match the original features")
        if not set(a.decision) <= DECISIONS:
            raise ValueError("Invalid annotation decision")
        return a

    def table(self):
        if _sha(self.candidate_path) != self.fingerprint["candidate_csv_sha256"] or _sha(self.group_path) != self.fingerprint["group_csv_sha256"]:
            raise ValueError("Source CSV changed during review. Restore the original before continuing.")
        out = self.frame.drop(columns=["label"]).merge(self._read_annotations(), on="candidate_id", validate="one_to_one")
        out["label"] = out.decision.map({"positive": 1, "negative": 0})
        return out

    def annotate(self, candidate_id, decision, evidence, reviewer):
        """Record independent evidence; never derives labels from model features."""
        if (self.work / "baseline_model").exists():
            raise ValueError("This evaluation is frozen. Preserve it and use fresh observations for further development.")
        self.table()
        if decision not in DECISIONS:
            raise ValueError("Use positive, negative, uncertain, or unreviewed")
        if decision != "unreviewed" and (not str(evidence).strip() or not str(reviewer).strip()):
            raise ValueError("Enter reviewer name/initials and evidence. If unsure, choose uncertain and explain why.")
        a = self._read_annotations()
        where = a.candidate_id.eq(str(candidate_id))
        if where.sum() != 1:
            raise ValueError("Unknown candidate ID")
        record = dict(candidate_id=str(candidate_id), decision=decision,
                      evidence=str(evidence).strip(), reviewer=str(reviewer).strip(),
                      reviewed_at=datetime.now(timezone.utc).isoformat())
        for key, value in record.items():
            a.loc[where, key] = value
        with (self.work / "review_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        _save_csv(a, self.annotation_path)

    def status(self):
        out = pd.crosstab(self.table().split, self.table().decision)
        out = out.reindex(index=["train", "validation", "test"], columns=["positive", "negative", "uncertain", "unreviewed"], fill_value=0)
        print(out.to_string())
        print("Reviews saved in:", self.annotation_path)
        print("Download a project backup after reviewing. Uncertain candidates are never assigned 0 automatically.")
        return out

    def readiness(self):
        frame = self.table()
        problems = []
        for split in ["train", "validation", "test"]:
            part = frame[frame.split == split]
            unreviewed = int(part.decision.eq("unreviewed").sum())
            if unreviewed:
                problems.append(f"{split}: {unreviewed} candidates still unreviewed")
            reviewed = part[part.decision != "unreviewed"]
            if reviewed[["evidence", "reviewer"]].apply(lambda s: s.str.strip().eq("")).any().any():
                problems.append(f"{split}: some reviewed candidates lack reviewer/evidence")
            labels = part.label.dropna()
            counts = labels.value_counts()
            minimum = 5 if split == "train" else 2
            if any(int(counts.get(label, 0)) < minimum for label in [0, 1]):
                problems.append(f"{split}: need at least {minimum} independently reviewed examples of EACH class")
        return problems

    def show(self):
        """Review widget. A plot or fit alone does not establish solar origin."""
        import ipywidgets as w
        from IPython.display import display, Image, clear_output
        table = self.table().sort_values(["group_id", "observed_peak_s"]).reset_index(drop=True)
        first = np.flatnonzero(table.decision.eq("unreviewed").to_numpy())
        idx = w.BoundedIntText(value=int(first[0] + 1) if len(first) else 1, min=1, max=len(table), description="Candidate:")
        person = w.Text(description="Reviewer:", placeholder="Name or initials")
        decision = w.Dropdown(options=[("Choose after review", "unreviewed"), ("1: supported solar burst", "positive"), ("0: supported non-burst", "negative"), ("Uncertain / needs expert", "uncertain")], description="Decision:", layout=w.Layout(width="460px"))
        notes = w.Textarea(description="Evidence:", placeholder="Reference event/time, expert judgement, or reason uncertain; no invented labels.", layout=w.Layout(width="95%", height="90px"))
        save = w.Button(description="Save and next", button_style="success")
        intervals = w.Button(description="Show day intervals")
        output = w.Output()
        messages = w.Output()

        def paint(change=None):
            row = table.iloc[idx.value-1]
            current = self._read_annotations().set_index("candidate_id").loc[row.candidate_id]
            decision.value, notes.value = current.decision, current.evidence
            if current.reviewer:
                person.value = current.reviewer
            with output:
                clear_output(wait=True)
                print(f"Candidate {idx.value}/{len(table)} | observation {row.group_id} | ID {row.candidate_id}")
                print(f"Peak: {row.observed_peak_s:.3f} s from this file's saved time origin")
                print("Quality flags:", row.quality_flags if pd.notna(row.quality_flags) else "none")
                print("Flags and the plotted FRED fit are not class labels.")
                info = self.plot_index.get(row.candidate_id)
                if not info:
                    print("Plot missing: restore this batch's review products from your backup.")
                    return
                directory, number = info
                manifest = json.loads((directory / "manifest.json").read_text())
                meta = manifest.get("metadata", {})
                print("Source:", meta.get("source_file", directory.name))
                if all(k in meta for k in ["original_mjdref", "time_origin_met_s", "original_time_scale"]):
                    from astropy.time import Time, TimeDelta
                    utc = (Time(float(meta["original_mjdref"]), format="mjd", scale=meta["original_time_scale"].lower()) + TimeDelta(float(meta["time_origin_met_s"]) + row.observed_peak_s, format="sec")).utc
                    print("Peak UTC:", utc.isot, "(for matching independent evidence)")
                png = directory / "review_plots" / f"candidate_{number:03d}.png"
                if png.exists():
                    display(Image(filename=str(png)))
                else:
                    print("Candidate image missing:", png)

        def save_next(button):
            with messages:
                clear_output(wait=True)
                try:
                    row = table.iloc[idx.value-1]
                    if decision.value == "unreviewed":
                        raise ValueError("Select a reviewed decision, or choose uncertain if you cannot decide.")
                    self.annotate(row.candidate_id, decision.value, notes.value, person.value)
                    print("Saved. Download a backup before leaving Colab.")
                    if idx.value < len(table):
                        idx.value += 1
                    else:
                        self.status()
                except (ValueError, OSError) as exc:
                    print(str(exc))

        def show_intervals(button):
            info = self.plot_index.get(table.iloc[idx.value-1].candidate_id)
            with output:
                if not info:
                    print("Restore review images first.")
                    return
                print("Day overview. Review images may be too coarse to identify every missed event.")
                for path in sorted((info[0] / "review_plots").glob("interval_*.png")):
                    display(Image(filename=str(path)))

        idx.observe(paint, names="value")
        save.on_click(save_next)
        intervals.on_click(show_intervals)
        display(w.VBox([w.HTML("<b>Independent candidate review</b><br>Positive = evidence supports a solar burst. Negative = evidence supports noise, artifact, or another non-burst cause. Unsure? Choose uncertain. Save before changing candidate number."), idx, person, output, decision, notes, w.HBox([save, intervals]), messages]))
        paint()
        return {"candidate": idx, "reviewer": person, "decision": decision, "evidence": notes, "save": save}

    def train(self):
        """Fit two fixed baselines; select using validation, evaluate test once."""
        problems = self.readiness()
        if problems:
            raise ValueError("Training is not ready:\n- " + "\n- ".join(problems) + "\nDo not invent labels to bypass these checks.")
        out = self.work / "baseline_model"
        if out.exists():
            raise FileExistsError(f"A frozen evaluation already exists at {out}. Read its report; do not repeatedly tune on this test day.")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
        from xgboost import XGBClassifier
        import xgboost
        import joblib

        all_rows = self.table()
        known = all_rows[all_rows.label.notna()].copy()
        train = known[known.split == "train"]
        validation = known[known.split == "validation"]
        test = known[known.split == "test"]
        # Select columns ONLY using the training partition, then freeze their order.
        columns = [col for col in FEATURES if train[col].notna().any()]
        if not columns:
            raise ValueError("No usable training features")
        specs = {
            "random_forest": RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=2),
            "xgboost": XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.03, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=42),
        }

        def metrics(y, score):
            predicted = (score >= 0.5).astype(int)
            return {
                "average_precision": float(average_precision_score(y, score)),
                "roc_auc": float(roc_auc_score(y, score)),
                "precision_at_0.5": float(precision_score(y, predicted, zero_division=0)),
                "recall_at_0.5": float(recall_score(y, predicted, zero_division=0)),
                "f1_at_0.5": float(f1_score(y, predicted, zero_division=0)),
                "confusion_matrix_true_rows_predicted_columns_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
                "n": int(len(y)), "positive_fraction": float(np.mean(y)),
            }

        models, validation_metrics = {}, {}
        for name, estimator in specs.items():
            model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("classifier", estimator)])
            model.fit(train[columns], train.label.astype(int))
            scores = model.predict_proba(validation[columns])[:, 1]
            validation_metrics[name] = metrics(validation.label.astype(int), scores)
            models[name] = model
        # Python insertion order gives the declared random-forest tie preference.
        selected = max(specs, key=lambda name: validation_metrics[name]["average_precision"])
        model = models[selected]
        # Create the directory BEFORE opening the held-out evaluation. Reruns do
        # not replace results silently, even if a later write fails.
        out.mkdir()
        metadata = {
            **self.fingerprint,
            "selected_model": selected, "feature_columns": columns,
            "excluded_all_missing_training_features": [c for c in FEATURES if c not in columns],
            "validation": validation_metrics,
            "annotation_sha256": _sha(self.annotation_path),
            "compatibility": {c: str(all_rows[c].iloc[0]) for c in ["pipeline_hash", "feature_version", "quantity", "unit"]},
            "versions": {**{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]}, "xgboost": xgboost.__version__},
            "scope": "Preliminary classification among reviewed, generated candidates; not end-to-end detection accuracy.",
            "limitations": ["Scores are uncalibrated, including predict_proba output.", "Only one validation day and one test day; nearby days may be correlated.", "Uncertain candidates are excluded, so metrics describe the adjudicated subset only.", "No continuous-interval truth: no false-alarm-per-hour or missed-event sensitivity claim.", "No learned morphology, physical rise/decay regression, or forecasting.", "Minimum class counts are software gates, not evidence of statistical adequacy."],
        }
        _json(out / "frozen_selection.json", metadata)
        joblib.dump({"model": model, "metadata": metadata}, out / "candidate_model.joblib")
        score = model.predict_proba(test[columns])[:, 1]
        metadata["test"] = metrics(test.label.astype(int), score)
        metadata["review_coverage"] = {}
        for split in ["train", "validation", "test"]:
            rows = all_rows[all_rows.split == split]
            metadata["review_coverage"][split] = {
                "total_candidates": int(len(rows)), "binary_labelled": int(rows.label.notna().sum()),
                "uncertain_excluded": int(rows.decision.eq("uncertain").sum()),
                "binary_label_fraction": float(rows.label.notna().mean()),
            }
        predicted = test[["candidate_id", "group_id", "label"]].copy()
        predicted["uncalibrated_score"] = score
        predicted["predicted_label_at_0.5"] = (score >= 0.5).astype(int)
        _save_csv(predicted, out / "test_predictions.csv")
        _save_csv(all_rows, out / "reviewed_training_snapshot.csv")
        _json(out / "evaluation.json", metadata)
        print("Saved model and evaluation:", out)
        print("Selected on validation:", selected)
        print("Held-out candidate metrics:", json.dumps(metadata["test"], indent=2))
        print("These are preliminary candidate-level results, not end-to-end solar-burst detection accuracy.")
        return out


def score_candidates(model_path, frame, *, trusted=False):
    """Use only your own trusted joblib; validate extraction compatibility."""
    if not trusted:
        raise ValueError("Use trusted=True only for your own model file; joblib loading executes serialized code.")
    import joblib
    bundle = joblib.load(model_path)
    meta = bundle["metadata"]
    for col, value in meta["compatibility"].items():
        if col not in frame or not frame[col].astype(str).eq(value).all():
            raise ValueError(f"Inference {col} differs from training")
    columns = meta["feature_columns"]
    if set(columns)-set(frame):
        raise ValueError("Missing inference features")
    result = frame.copy()
    if len(result):
        x = result[columns].apply(pd.to_numeric, errors="raise")
        if np.isinf(x.to_numpy(dtype=float)).any():
            raise ValueError("Infinite inference features")
        result["uncalibrated_score"] = bundle["model"].predict_proba(x)[:, 1]
        result["candidate_prediction"] = (result.uncalibrated_score >= meta["threshold"]).astype(int)
    else:
        result["uncalibrated_score"] = pd.Series(dtype=float)
        result["candidate_prediction"] = pd.Series(dtype=int)
    return result
