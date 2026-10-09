# Continue in your current Colab notebook

This is an add-on for XSM_Hybrid_ML_Code v0.2.0. Keep using your existing notebook.
It expects the variables `PROJECT`, `BATCH_RUN`, and `review_folder` from your
previous six cells. If the runtime was lost, restore your latest project backup
first; the two CSVs alone cannot restore the light curves or review images.

## What the uploaded CSVs show

- 280 unique candidates across six observation dates; all 280 labels are blank.
- No duplicate candidate IDs or infinite numeric feature values.
- 144 candidates have the `single_bin_candidate` flag.
- 135 fits report optimizer success, but all 280 candidates have missing
  reliable physical rise, decay, duration, and peak fit features.
- Neither a successful fit nor a warning determines a true event label.

The following split is fixed before annotation and model evaluation:

| Role | Observation dates | Candidates |
|---|---|---:|
| Train | 2026-07-16, 2026-09-13, 2026-09-16, 2026-09-17 | 130 |
| Validation | 2026-09-19 | 96 |
| Test | 2026-09-20 | 54 |

## Cell A — upload this small add-on ZIP

Download `XSM_Review_Training_Addon.zip` from the chat to your computer. Add this
cell below the existing cells and run it. Select that add-on ZIP when prompted.

```python
from pathlib import Path
import sys, subprocess, io, zipfile
from google.colab import files

required = ["PROJECT", "BATCH_RUN", "review_folder"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Missing notebook variables: " + ", ".join(missing)
        + ". Restore your project backup and the existing notebook setup first."
    )
PROJECT = Path(PROJECT)
if not (PROJECT / "run_xsm.py").is_file():
    raise RuntimeError("PROJECT does not point to the restored v0.2 XSM project.")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ipywidgets>=8,<9"])
uploaded = files.upload()
installed = False
for name, data in uploaded.items():
    if not name.lower().endswith(".zip"):
        continue
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if "xsm_review_training.py" not in archive.namelist():
            continue
        for member in ["xsm_review_training.py", "START_HERE.md", "VALIDATION.txt"]:
            if member in archive.namelist():
                # Read only the exact known members; never extract arbitrary paths.
                (PROJECT / member).write_bytes(archive.read(member))
        installed = True
if not installed:
    raise RuntimeError("Select XSM_Review_Training_Addon.zip, not the dataset ZIP.")
print("Installed. Run Cell B next.")
```

## Cell B — open the review tool

```python
import sys
from google.colab import output
output.enable_custom_widget_manager()
sys.path.insert(0, str(PROJECT))
from xsm_review_training import ReviewSession

review = ReviewSession(BATCH_RUN, review_folder)
review.status()
review.show()
```

Enter your name/initials. Inspect the candidate plot and, where needed, its day
intervals and independent evidence. The tool shows a UTC timestamp for matching
an event catalogue or an expert's annotation. Use the same review criteria for
every day. Model scores are not shown during annotation.

- **1: supported solar burst:** independent evidence/expert review supports the
  event being a solar burst. Record the evidence and timing match.
- **0: supported non-burst:** evidence supports noise, an instrumental artifact,
  or another non-burst cause. Absence from a catalogue alone is insufficient.
- **Uncertain / needs expert:** evidence is insufficient. This is the appropriate
  choice if you do not know how to classify it. Add a short explanation.

Do not use an SNR cutoff, a fit flag, or the fitted curve itself as ground-truth
labels. Do not mark everything negative. The supplied FITS metadata and XMLs do
not supply event class labels. If you cannot judge the events, ask your ISRO
contact/supervisor for an annotated event list or a qualified review of the plots.
No code can create independent training truth from these unlabelled files alone.

Click **Save and next**. To revisit a candidate, change its number at the top;
save before navigating. Reviews are saved after every click. Rerunning Cell B
resumes the first unreviewed candidate and preserves saved decisions.

Original feature CSVs remain unchanged. Reviews, the fixed split, and an edit
history are stored under `review_folder/review_baseline/`. Nothing is sent to
another person. The `fully_reviewed` exposure flags are not automatically changed:
reviewing generated candidates is not a full search for missed events.

If widgets fail to render, a direct annotation call also works after reviewing
the corresponding evidence:

```python
# Example ONLY: use an actual ID from review.frame, with your real review.
# review.annotate(candidate_id, "uncertain", "Reason evidence is insufficient", "Your initials")
```

## Cell C — download a backup after reviewing

Run this even if review is incomplete. It includes the add-on, annotations,
source data, and images. Keep the downloaded ZIP; runtime storage is temporary.
You can also rerun your original project-backup cell instead.

```python
from datetime import datetime, timezone
import shutil
from google.colab import files

stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
backup = shutil.make_archive(
    f"/content/XSM_work_backup_{stamp}", "zip",
    root_dir=PROJECT.parent, base_dir=PROJECT.name
)
files.download(backup)
```

## Cell D — check whether training is ready

```python
review.status()
problems = review.readiness()
print("\n".join(problems) if problems else "Ready for preliminary training.")
```

Every candidate must have a review decision and reviewer/evidence recorded.
Uncertain candidates are excluded from fitting and metrics, with exclusion counts
reported. Training needs at least 5 positive and 5 negative examples in training,
and at least 2 of each class in validation and test. These are minimal software
checks, not a claim that such small samples establish accuracy. If a split lacks
one class, obtain more independently labelled observations and revise the study
plan; do not invent labels or move individual candidates across days.

## Cell E — train only when Cell D says ready

```python
model_folder = review.train()
```

The add-on fits a Random Forest and XGBoost with fixed settings on training days.
Missing-value imputation and removal of entirely missing columns use training
data only. It selects the model using validation average precision; ties prefer
Random Forest. The decision threshold is fixed at 0.5 before evaluation. Only
the selected model is evaluated on the held-out day, and outputs are protected
against replacement. The model remains fitted on the training days.

The imputer and classifier are saved together in `candidate_model.joblib`.
`evaluation.json` includes validation results, held-out metrics, class counts,
uncertain exclusions, software versions, and limitations. `test_predictions.csv`
and a snapshot of the reviewed table are included. The original strict five-way
trainer is unchanged; use this add-on's `review.train()` for this preliminary run.

Results apply only to the labelled subset of generated candidates. They cannot
measure missed solar events or operational false alarms per observing hour.
Scores are **uncalibrated**. One validation day and one test day cannot establish
generalization. Do not tune and rerun against the same test day. Do not claim
physical-parameter regression or learned morphology from this classifier.

## Cell F — download the trained model

```python
model_zip = shutil.make_archive(
    "/content/XSM_preliminary_model", "zip",
    root_dir=model_folder.parent, base_dir=model_folder.name
)
files.download(model_zip)
```

Then rerun Cell C to back up the full project with the trained model included.
To inspect a saved report without retraining:

```python
import json
report_path = review.work / "baseline_model" / "evaluation.json"
print(json.dumps(json.loads(report_path.read_text()), indent=2))
```

## Use the trained classifier on new extracted candidates

First process new Level-2 `.lc` and matching `.gti` files with the same v0.2
pipeline/settings. This helper accepts extracted features, not raw FITS bytes:

```python
from xsm_review_training import score_candidates
# new_candidates = pd.read_csv("PATH_TO_NEW_CANDIDATES_CSV")
# predictions = score_candidates(
#     model_folder / "candidate_model.joblib", new_candidates, trusted=True
# )
```

Change the commented path to your new candidate CSV and uncomment after extraction.
Load only your own trusted joblib files. Future input must match the saved feature
schema, units, and pipeline fingerprint. Keep fresh independent observations for
subsequent evaluation.

Implementation references: scikit-learn RandomForestClassifier API
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
and ipywidgets widget events API
https://ipywidgets.readthedocs.io/en/latest/reference/ipywidgets.html .
