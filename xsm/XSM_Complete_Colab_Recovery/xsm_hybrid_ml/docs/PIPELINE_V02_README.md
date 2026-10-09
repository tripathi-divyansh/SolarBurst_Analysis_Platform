# XSM Hybrid ML 0.2.0 — corrected detector and Colab recovery

**Start with [COLAB_START_HERE.md](COLAB_START_HERE.md) and `Colab_Recovery.ipynb`.** Your July 16 LC/GTI and completed corrected results are included. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for measured results, changes, and limitations.

```bash
python run_xsm.py --input data/july16 --out runs/my_new_run
```

The runner protects existing results. Choose a new output directory for each run.

## Architecture

Implements the supplied **ISRO XSM Solar Burst Architecture**:

**Input → quality control → robust background → Sigma/MAD + Mexican-hat wavelets + Bayesian Blocks → candidate segmentation → causal FRED fitting and signal features → XGBoost burst classifier → supervised morphology classifier → optional parameter regression → quality flags and catalog.**

This is an ML/scientific backend and CLI, ready to connect to a GUI. It is not a completed website. This release includes actual July 16 observations but no reviewed event labels and no pretrained weights. The synthetic demo command can train demonstration models; those models are not XSM-validated. No real-data detection accuracy, zero-error guarantee, or production false-alarm guarantee is claimed.

## 1. Quick start

Tested on Linux x86-64, Python 3.12. Use a virtual environment; Windows users can run this tested Linux route in WSL2. Other platforms need separate dependency/installation verification.

```bash
cd xsm_hybrid_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

The pinned file records the actual tested package versions. For a fresh dependency resolution within supported bounds, use `python -m pip install -e ".[test]"` instead. The tested requirements file is not a cryptographic/hash lock.

Run the complete synthetic example (CPU; fitting many candidates can take several minutes):

```bash
python -m xsm_burst.cli demo --out my_demo --groups 60
```

It creates synthetic light curves, extracts real pipeline features, trains XGBoost, calibrates scores on separate periods, selects an operating threshold, saves a model, and evaluates held-out synthetic periods. No old synthetic demonstration weights are bundled with this release. Do not use a newly generated synthetic model to make claims about real solar observations.

## 2. Analyze your own observation

For standard XSMDAS 1.50 Level-2 `.lc` files with matching external `.gti`, prefer `run_xsm.py` or `xsm_burst.xsm.load_xsm_level2`. It resolves the documented bin-start convention and exposure. The generic reader below still requires explicit mapping and does not replace the instrument adapter.


Create a JSON mapping from `examples/input_mapping.json`. Replace the epoch/time scale placeholders with verified product values. Explicitly check column names, bin widths, exposure, units, quality-bit semantics, time offsets, energy band, and whether timestamps represent bin centers.

```bash
python -m xsm_burst.cli extract observation.fits \
  --mapping my_mapping.json \
  --group observing_period_001 \
  --out extracted/period_001
```

Output:

| File | Contents |
|---|---|
| `candidates.csv` | Candidate features, preliminary boundaries, quality flags, and provenance |
| `fits.json` | Physical-fit parameters, predictions, residuals, diagnostics, optional bootstrap intervals |
| `manifest.json` | Source hash, quantity/units, configuration, warnings, valid and processed exposure |
| `baseline.npz` | Time and estimated background arrays |

Without a trained model, every proposed event remains `review_unscored`. Candidate existence is not confirmation of a real burst.

Supported parsers: FITS/.lc/.fits.gz, comma CSV, TSV, whitespace ASCII, XLS and XLSX. FITS column mapping is explicit. It is a generic reader, not a complete automated PRADAN calibration pipeline. CDF and arbitrary archive detection are not implemented in this version. Legacy XLS uses `xlrd`; an XLS fixture was not supplied for the verification run.

For CSV/ASCII, numeric time is converted using `time_unit_s` and `time_offset_s`. Width and exposure columns/scalars must already be **seconds**. FITS users must resolve mission reference-time keywords and TIMEZERO into the explicit mapping. The library validates an absolute epoch but analyzes relative elapsed seconds; it does not guess or automatically convert an arbitrary mission MET convention to calendar time. Do not interpret an arbitrary numeric timestamp as Unix time.

For rates and flux, positive measurement errors are required. If only raw integer counts are available, set `quantity` to `counts`, supply exposure, and omit the error field. Detection uses a Poisson-error approximation; physical fitting uses the Poisson count likelihood. Never relabel corrected values as raw counts.

Any nonzero quality flag is invalid. Map mission flags appropriately before use; mark saturation and unresolved state transitions invalid. Full bins contained in FITS `GTI`/`STDGTI` intervals are retained; partial bins are conservatively excluded. GTIs must share the table's mapped time reference and units. Exposure is assumed uniformly distributed within each included bin. Arbitrary sub-bin live-time structure needs an instrument-specific extension.

## 3. Prepare real training data

Extract features from contiguous independent observing periods, then combine their candidate tables. Review complete time intervals, including missed events, not only proposed candidates.

Add these columns:

| Column | Meaning |
|---|---|
| `label` | Reviewed genuine burst = 1; reviewed non-burst/artifact = 0 |
| `split` | Exactly one of `train`, `tune`, `calibration`, `threshold`, `test` |
| `group_id` | Original observing-period group shared by all related candidates/variants |
| `morphology_label` | Optional reviewed class name on positive candidates; missing otherwise |
| `regression_target_reliable` | Optional true/false flag confirming trustworthy parameter targets |
| `target_rise_s` | Defined rise duration, not the FRED rise constant |
| `target_decay_constant_s` | Effective exponential decay constant |
| `target_duration_s` | Defined burst duration |
| `target_net_peak` | Background-subtracted peak, in the feature table's signal unit |

Do not generate real labels by thresholding the same features you then claim to validate. Uncertain labels must not become automatic negatives. Use real reviewed periods and varied synthetic injections for training; preserve label provenance outside the numeric feature columns.

Split **before extraction, injection or augmentation**. Assign every parent flare, neighboring processing context and background-derived variant to the same group. Apply guard intervals at least as wide as the largest analysis context. The code rejects a group or source hash appearing in different partitions, but it cannot infer undocumented relationships between different source files. Those must be expressed in `group_id`.

The five roles are separate:

1. `train`: fit preprocessing and trees.
2. `tune`: early stopping and model-development decisions.
3. `calibration`: sigmoid probability calibration on natural candidate prevalence.
4. `threshold`: operating threshold for the declared false-alarm target.
5. `test`: final metrics after choices are fixed. Do not repeatedly choose configurations from this report.

Each partition must contain both burst and non-burst candidates and at least two groups. This is only a software guard; it is not a statistical sample-size recommendation. Real work generally needs many more independently reviewed events and observing days. With limited data, carefully designed grouped cross-fitting is preferable, but is not implemented here.

Create `exposure.csv`, one row per fully reviewed observing period:

```csv
group_id,split,processed_exposure_s,fully_reviewed
observing_period_001,train,7200,true
observing_period_002,train,5400,true
```

Include periods with zero proposals. Use the **processed valid exposure**, excluding invalid bins, discarded short segments, and unreviewed intervals. Do not multiply exposure by the number of candidates. If ambiguous portions are omitted, remove their exposure and re-extract the remaining reviewed segments.

## 4. Train and predict

```bash
python -m xsm_burst.cli train training_candidates.csv \
  --exposure exposure.csv \
  --model models/xsm_burst.joblib \
  --target-far 1.0

python -m xsm_burst.cli predict extracted/period_001/candidates.csv \
  --model models/xsm_burst.joblib \
  --trust-model \
  --out predictions.csv
```

`--trust-model` is required because Python/joblib model deserialization can execute code. Only load artifacts produced by your own trusted training workflow, with matching dependency versions.

Save/reload includes the imputer, all missingness indicators, feature order, classifiers, calibration, thresholds, optional regressors and metadata. Physical quantity, unit string, pipeline configuration hash and feature schema must match training. Retrain or explicitly validate a new model after changing feature definitions, energy-band treatment, preprocessing or candidate settings. Mixed units are rejected, not automatically converted.

The probability field describes the fitted calibration relationship for the stated domain. Small calibration sets and domain shift remain limitations even though a calibrator exists. The synthetic bundle's `probability_scope` says `SYNTHETIC_DEMO_ONLY`.

The final status is `accepted` or `review_below_threshold`. These are classifier decisions; fit reliability and instrument-quality flags remain separate. A low score does not prove an instrument artifact, and accepted events with poor fits must not be presented as reliable measurements.

## 5. Morphology and regression

When positive training/tuning examples have suitable `morphology_label` values, a second XGBoost model is trained. It returns **uncalibrated class scores**, not validated morphology probabilities. If labels are absent or classes lack tuning support, the learned morphology model remains unavailable.

An always-available, explicitly named operational fallback uses fitted rise/decay asymmetry and observed peak count. Morphology classes are team-defined and do not identify unique physical mechanisms. A single FRED function cannot represent every morphology.

The optional regressors train only when all four positive targets are reliable, with at least 20 training and 5 tuning examples. They exclude physical-fit feature columns to reduce circular target leakage. Their predictions still share the observations and signal features with the physics fit, so they are a **correlated consistency check**, not independent confirmation. Targets derived from biased physical fits reproduce those biases; trustworthy synthetic truth or independently reviewed estimates are preferable.

## 6. Scientific definitions and error reduction

- The FRED signal is zero before onset. Its amplitude parameter differs from net peak.
- The analytic peak is `t0 + tau_r * log1p(tau_d / tau_r)`.
- Primary fitted boundaries are 5% of fitted net peak. Preliminary candidate boundaries are exported separately.
- Bin-integrated predictions are used for fitting; original observations are not smoothed or interpolated into synthetic measurements.
- Fit constant `tau_r` differs from measured rise duration; `tau_d` differs from decay duration.
- Fluence is background-subtracted. Full-model fluence can be extrapolated when an event is censored; check flags.
- Wavelets are skipped and flagged for substantially irregular sampling. No gap interpolation is used to create wavelet evidence.
- Bayesian Blocks uses Gaussian point-measurement fitness on residual rates with supplied/estimated errors. For raw low counts this is an approximate proposal generator, not an exact Poisson event-arrival analysis.
- Bayesian Blocks is capped through contiguous inverse-variance aggregation to bound its quadratic cost. Coarsening is flagged.
- Duplicate timestamps are rejected by default. Optional `identical` handling only drops exact repeated records; it never averages conflicting observations automatically.
- Nonfinite values, zero exposure and nonzero quality flags break segments. Independent time segments cannot be merged into a burst across a gap.
- A robust excess-noise estimate subtracts measurement variance before adding any extra component, avoiding blind double counting.
- Fits are multistart and bounded; failed, censored, unresolved, correlated-residual and ill-conditioned cases are flagged.
- Measurement likelihood and fit quality are distinct from detection probability. A high R² alone is insufficient.
- Train-only imputation, feature-order checking, disjoint calibration, and disjoint threshold selection reduce software and evaluation errors.

Set `bootstrap_samples` in a pipeline configuration for parametric model-based 95% intervals. At least 20 successful refits are required; approximately 200 is a more useful starting point than 20. This refits the constant background and FRED together but assumes the model and independent measurement noise. Correlated noise and detection/model-selection uncertainty are not captured. Do not interpret the intervals as fully calibrated until coverage is tested.

## 7. Evaluation and limits

The training report's classification metrics are **candidate-level**. They exclude bursts never proposed by signal processing. Use `xsm_burst.evaluation.match_events` with a reviewed event catalog to measure complete-pipeline recall. It enforces one-to-one overlap/peak matching and penalizes duplicate predictions. The synthetic demo runs both evaluations.

False alarms are normalized by valid processed exposure, not file duration or candidate count. The report includes a simple Poisson upper 95% FAR bound; zero observed false positives in a short test is not evidence of zero operational risk. Real false alarms can be correlated, so day/block resampling is needed for a stronger uncertainty analysis.

Current boundaries:

- Requires real labeled data before scientific deployment; the supplied July day is unlabelled development data.
- Uses a robust local-linear detection background and constant FRED fit background, with a separate linear-background comparison. Extremely broad bursts, strongly sloped backgrounds and high duty cycles can bias these estimates.
- Fits one FRED component per candidate. Multi-peak candidates are flagged and retained, not claimed to be resolved. Joint multi-component inference is future work.
- The illustrative synthetic regression targets cover the FRED family only; their validity does not extend to all real morphologies.
- No automatic GOES classification from XSM counts/rates; no spectral calibration or instrument response inversion.
- No saturation-recovery or censored-count likelihood. Mark saturated measurements invalid; peak reconstruction from missing data is unreliable.
- No complete web GUI, cross-platform packaging guarantee, or formal validation on all input formats/instruments.
- No model-selection search against the test set. Hyperparameters are conservative starting points.

## 8. Files

| Module | Role |
|---|---|
| `schema.py` | Input validation, bins, exposure, duplicates and gap segmentation |
| `io.py` | Explicit FITS/text/Excel mapping |
| `detection.py` | Robust background and three proposal generators |
| `physics.py` | Causal FRED, bin means, likelihood fitting and bootstrap |
| `features.py` | Versioned candidate features and operational morphology |
| `pipeline.py` | End-to-end candidate extraction and quality flags |
| `models.py` | Burst/morphology classifiers, optional regression, calibration and persistence |
| `evaluation.py` | Candidate metrics, FAR and one-to-one event matching |
| `synthetic.py` | Reproducible synthetic observations and end-to-end demo |
| `cli.py` | Extract/train/predict/demo commands |
| `tests/` | Numerical, scientific and model integration checks |

## 9. Python integration

```python
from xsm_burst import analyze, PipelineConfig
from xsm_burst.io import load_lightcurve
from xsm_burst.models import BurstModels

curve = load_lightcurve("observation.csv", "mapping.json")
result = analyze(curve, PipelineConfig(), group_id="period_001")
model = BurstModels.load("models/xsm_burst.joblib", trusted=True)
catalog = model.score(result["catalog"])
catalog.to_csv("catalog.csv", index=False)
# GUI integrations can retrieve original samples, result['fits'] and baseline.
```

## 10. Sources

The user-supplied architecture is preserved in `docs/input_architecture.txt`.

- [Astropy Bayesian Blocks](https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html): point-measurement fitness and change points.
- [XGBoost sklearn estimator interface](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html): training, early stopping and inference.
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): score calibration and evaluation.

See `TEST_REPORT.md` for the tests and demonstration actually executed with this package.
