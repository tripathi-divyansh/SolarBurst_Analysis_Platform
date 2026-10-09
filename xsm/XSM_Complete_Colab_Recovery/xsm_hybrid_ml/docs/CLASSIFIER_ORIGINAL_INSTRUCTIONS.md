# Final workflow: keep extraction, compare supervised simulation classifiers

Use the current Colab notebook. Do not delete previous cells or change your real
review labels. This is a new experiment; do not run the original `review.train()`
for it. The original real-data classifier still needs reviewed labels.

The supplied module uses your existing v0.2 XSM project. Your trained Isolation
Forest remains a frozen comparison baseline. Background/noise scales come only
from the original real training days. Random Forest and XGBoost learn from known
simulated bursts and artifacts; they do NOT learn confirmed labels on ISRO data.

The experiment generates independent full observations before extracting
candidates: 96 training, 36 validation, and 60 test observations, each 900 seconds.
Every split includes bursts, noise-only observations, and artifact-only controls.
There are four burst shape families, varying strengths and widths, Gaussian or
correlated noise, drifting backgrounds, narrow spikes, and softened rectangular
artifacts. This is still a simplified simulation, not a complete instrument model.

Existing extraction and measured features are retained. Five raw-context features
are added: peak concentration, roughness, flat-top fraction, longest sustained
excess, and positive area relative to peak. Physical fitted parameters missing
from your real candidate data are not classifier inputs. There is no automatic
rule that every short event is false.

## Cell 1 — upload the code add-on

Download `XSM_Classifier_Comparison.zip` from the chat. Run this new cell at the
bottom of the current notebook and select the ZIP. No path changes are needed.

```python
from pathlib import Path
from google.colab import files
import sys, io, zipfile

required = ["PROJECT", "demo_data", "anomaly_model", "feature_columns", "review_threshold"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError("Run the earlier Isolation Forest setup/training cells first. Missing: " + ", ".join(missing))

uploaded = files.upload()
installed = False
for name, data in uploaded.items():
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "xsm_classifier_demo.py" in archive.namelist():
                (Path(PROJECT) / "xsm_classifier_demo.py").write_bytes(archive.read("xsm_classifier_demo.py"))
                installed = True
if not installed:
    raise RuntimeError("Select XSM_Classifier_Comparison.zip.")

sys.path.insert(0, str(Path(PROJECT) / "src"))
sys.path.insert(0, str(PROJECT))
from xsm_classifier_demo import run_experiment
print("Ready. Run Cell 2 next.")
```

All dependencies are already in the original v0.2 project's tested requirements.
If XGBoost alone is missing, install it in another cell before Cell 2:

```python
# Only if you receive ModuleNotFoundError for xgboost:
# %pip install -q xgboost-cpu==3.4.1
```

## Cell 2 — generate simulations, train, select, and test

```python
comparison_folder = Path(PROJECT) / "runs" / "synthetic_classifier_v1"

comparison_folder = run_experiment(
    frame=demo_data,
    isolation_model=anomaly_model,
    isolation_columns=feature_columns,
    isolation_threshold=review_threshold,
    output_dir=comparison_folder
)
```

This can take several minutes. Progress prints every six simulated observations.
Wait for the final results table. The folder is protected against overwriting;
if you already finished this experiment, continue at Cell 3 rather than retraining.
An interrupted run can leave a partial folder. Keep it for diagnosis; restore
the running session or create a separately named recovery run only if necessary.
Reruns using the same seeds are repetitions, not new independent evidence.

The models use training-only missing-value imputation. Random Forest has 300
trees, maximum depth 6 and minimum leaf size 3. XGBoost has maximum depth 3,
learning rate 0.05 and a 600-round maximum, with early stopping after 30 rounds
without validation log-loss improvement. These are fixed starting settings,
not values guaranteed to be optimal.

Both classifier thresholds are fixed at 0.5. The classifier with the best
validation end-to-end event F1 is selected; ties prefer Random Forest. Selection
and saved models are frozen BEFORE test simulations are generated. Test results
for both models and the old baseline are then reported for comparison; do not
choose a different winner simply because it scored better on this test sample.

## Cell 3 — display the graphs and table

```python
from IPython.display import display, Image
import pandas as pd
import json

comparison_folder = Path(PROJECT) / "runs" / "synthetic_classifier_v1"

display(Image(filename=str(comparison_folder / "training_validation_loss.png")))
display(Image(filename=str(comparison_folder / "heldout_comparison.png")))
display(pd.read_csv(comparison_folder / "test_comparison.csv"))
display(pd.read_csv(comparison_folder / "test_recovery_by_strength.csv"))

report = json.loads((comparison_folder / "report.json").read_text())
print("Selected using validation:", report["selected_model"])
print("Selected XGBoost boosting round:", report["best_xgboost_round"])
print("Scope:", report["scope"])
```

### Read the graphs

- **Loss:** lower log loss means better candidate-label predictions on these
  simulations. If training loss keeps falling while validation loss rises, that
  suggests overfitting to the simulated training examples. Early stopping uses
  validation to choose the number of rounds.
- **Both losses high:** compare them against the constant-prediction baseline.
  This could reflect underfitting, ambiguous examples, insufficient features, or
  a difficult simulation. A small gap alone does not prove a good model.
- **Recovery:** higher is better, especially for weak bursts. Look at counts as
  well as percentages. Events never proposed by extraction count as missed.
- **False triggers:** lower is better. This graph uses separate zero-burst
  controls, including known artificial artifacts. It is a simulated rate only.
- **Precision and recall:** the report also counts unmatched detections in
  burst-containing observations. Matching requires interval overlap and peak
  separation of at most 35 seconds, one-to-one; duplicate detections count false.

The two curves refer to XGBoost training and validation simulation candidates.
The old Isolation Forest does not have a comparable supervised loss curve.
The test graph uses new observations, independent of training and validation.
All splits share the same broad simulation generator; real-world transfer is
still unverified. Do not report these numbers as real ISRO detection accuracy.

## Cell 4 — download the model, graphs, scores, and protocol

```python
import shutil
from google.colab import files

result_zip = shutil.make_archive(
    "/content/XSM_classifier_results", "zip",
    root_dir=str(comparison_folder.parent),
    base_dir=comparison_folder.name
)
files.download(result_zip)
```

Then rerun your existing full-project backup cell and download the notebook.

### Main outputs

- `selected_model.joblib`: selected classifier, training-fitted imputer, feature
  order, compatibility information, threshold and synthetic-only scope.
- `random_forest.joblib`, `xgboost.joblib`: both fitted candidate classifiers.
- `xgboost_model.json`: native booster export; the imputer is still required.
- `xgboost_loss.csv`: actual training/validation loss values by boosting round.
- `training_validation_loss.png`, `heldout_comparison.png`: graphs.
- `frozen_selection.json`, `report.json`, `protocol.json`: selection, results,
  assumptions, fixed seeds, software versions and limitations.
- `train/`, `validation/`, `test/`: candidate features, known injected-event
  truth, and observation exposures. These are explicitly simulation-labelled.
- `test_predictions.csv`, `test_recovery_by_strength.csv`, `test_comparison.csv`:
  held-out model outputs and detection metrics.

## Optional Cell 5 — exploratory ranking of your real saved XSM batch

This requires the original batch's `input_arrays.npz` and `baseline.npz`, as well
as its CSVs. The original CSV alone cannot supply the added context features.

```python
from xsm_classifier_demo import score_real_batch

real_scores = score_real_batch(
    model_path=comparison_folder / "selected_model.joblib",
    batch_run=BATCH_RUN,
    output_csv=comparison_folder / "real_candidates_exploratory.csv",
    trusted=True   # Only load the model you just created yourself.
)

display(real_scores.sort_values("synthetic_classifier_score", ascending=False).head(10))
```

These are experimental rankings, not confirmed event labels or calibrated solar
burst probabilities. The function preserves your original extraction CSVs and
review annotations. Genuine real-data evaluation still requires independent
reviewed events and observing intervals. Download Cell 4 again after this step
to include the exploratory rankings.

## References

XGBoost evaluation sets and early stopping:
https://xgboost.readthedocs.io/en/stable/python/python_intro.html

Random Forest classifier controls:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## Included example results

The `example_results/` folder contains the completed local test graphs and report,
using your supplied training candidate scales and the earlier Isolation Forest
settings. They are simulation results. Both models are exercised by the code,
and the selected model is saved when you run Cell 2. Software versions are in
`example_results/protocol.json`; numerical results can vary with versions.
Only two weakest-band events occurred in this test: do not treat their 2/2
recovery as reliable evidence of weak-burst sensitivity.
