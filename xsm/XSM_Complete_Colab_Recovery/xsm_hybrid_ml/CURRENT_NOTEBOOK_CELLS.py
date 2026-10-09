# %% [markdown]
# # XSM complete Colab recovery — 10 October 2026
# Run cells **1–6 in order**. No Google Drive and no path edits are required.
# Use a CPU runtime. Select `XSM_Complete_Colab_Recovery.zip` when Cell 1 asks.
# After a later reset, select your latest downloaded **full backup ZIP** instead.
# 
# This restores the corrected v0.2 pipeline, 5 added context features, Isolation Forest,
# RF/XGBoost simulation training, saved models, evaluation plots, and early stopping.
# The original XGBoost settings are retained because stronger regularization did not improve validation.
# 
# **Scope:** saved models/results were reproduced in the assistant's environment from the shared
# candidate CSVs. They are not a snapshot of your inaccessible Colab runtime.
# Included: 280 unlabelled real candidates across six dates, July 16 LC/GTI, prior review assets,
# and 96/36/60 train/validation/test simulations. Missing: other original observation files and
# any reviews or edits saved only in your Colab session. Real labels are never invented.

# %% [markdown]
# ## Cell 1 — upload and restore
# Choose the complete recovery ZIP (or a later full backup made by Cell 6).
# Restores into a new folder so existing work is not overwritten.

# %%
from pathlib import Path, PurePosixPath
import io, zipfile, tempfile, sys, subprocess, os, json
from google.colab import files

uploaded=files.upload()
archives=[(name,data) for name,data in uploaded.items() if name.lower().endswith('.zip')]
if len(archives)!=1:
    raise ValueError('Upload exactly one complete recovery or full backup ZIP.')
destination=Path(tempfile.mkdtemp(prefix='xsm_recovery_',dir='/content'))
with zipfile.ZipFile(io.BytesIO(archives[0][1])) as archive:
    for item in archive.infolist():
        p=PurePosixPath(item.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in item.filename:
            raise ValueError('Unsafe archive path: '+item.filename)
        if (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError('Symbolic links are not accepted in this backup.')
    archive.extractall(destination)
roots=[p.parent for p in destination.rglob('recovery_actions.py')
       if (p.parent/'src/xsm_burst').is_dir()]
if len(roots)!=1:
    raise ValueError('This ZIP is not the complete recovery package. Upload XSM_Complete_Colab_Recovery.zip.')
PROJECT=roots[0]
comparison_folder=PROJECT/'runs/synthetic_classifier_v1'
BATCH_RUN=PROJECT/'results/july16_v02'
Path('/content/xsm_project_path.txt').write_text(str(PROJECT))
print('Restored:',PROJECT)
print('Run Cell 2 next. No paths need changing.')
del uploaded,archives


# %% [markdown]
# ## Cell 2 — install dependencies in an isolated environment
# This avoids the NumPy import conflict encountered earlier. Scientific code runs in a separate
# Python process; the notebook's imported NumPy is not replaced. Internet is needed for installation.
# Requires Colab Python 3.12 or newer. Re-run this cell after a runtime reset, following Cell 1.

# %%
from pathlib import Path
import sys, subprocess, os, hashlib
if 'PROJECT' not in globals():
    PROJECT=Path('/content/xsm_project_path.txt').read_text().strip()
PROJECT=Path(PROJECT)
if sys.version_info<(3,12):
    raise RuntimeError('Select a Colab runtime with Python 3.12 or newer.')
ENV=Path('/content/xsm_recovery_python')
PYTHON=ENV/'bin/python'
if not PYTHON.exists():
    subprocess.run([sys.executable,'-m','venv',str(ENV)],check=True)
requirements=PROJECT/'requirements-recovery.txt'
digest=hashlib.sha256(requirements.read_bytes()).hexdigest()
marker=ENV/'installed_requirements.sha256'
if not marker.exists() or marker.read_text()!=digest:
    subprocess.run([str(PYTHON),'-m','pip','install','--disable-pip-version-check',
                    '-r',str(requirements)],check=True)
    marker.write_text(digest)
def run_xsm_action(action):
    env=os.environ.copy()
    env['MPLBACKEND']='Agg'
    subprocess.run([str(PYTHON),str(PROJECT/'recovery_actions.py'),action],
                   cwd=PROJECT,env=env,check=True)
print('Dependencies ready. Run Cell 3.')


# %% [markdown]
# ## Cell 3 — restore the baseline and verify early stopping
# Rebuilds the unsupervised Isolation Forest using only the fixed training days.
# Loads the saved XGBoost model; it does not retrain it. Expected best round: 205; stop: 235.
# Only use trusted recovery archives: saved joblib models contain executable Python serialization.

# %%
run_xsm_action('restore')


# %% [markdown]
# ## Cell 4 — evaluate the saved test and show the graphs
# This reproduces the **existing** test report, not a new untouched evaluation.
# Models were originally selected on validation. Do not choose new settings from these test results.
# Loss is candidate-level; burst recovery uses one-to-one event matching, including missed proposals.
# False triggers/hour is measured on simulated zero-burst controls, not real exposure.

# %%
run_xsm_action('evaluate')
from IPython.display import display, Image
comparison_folder=PROJECT/'runs/synthetic_classifier_v1'
for filename in ['training_validation_loss.png','heldout_comparison.png']:
    display(Image(filename=str(comparison_folder/filename)))


# %% [markdown]
# ## Cell 5 — repeat the validation-only regularization check
# Compares the current settings with three predefined alternatives. Leaves saved models unchanged.
# Expected: original/shallow recover 20/24 bursts; stronger alternatives recover 18/24.
# A smaller train–validation loss gap is not sufficient reason to change the model.

# %%
run_xsm_action('regularization')


# %% [markdown]
# ## Cell 6 — download your full project backup
# Run this before closing Colab and after any optional cells. The ZIP includes everything inside
# PROJECT: code, data, trained models, review files, plots, and this notebook. Files elsewhere in
# `/content` and unsaved notebook-cell edits are not included; save notebook edits separately with
# **File → Download → Download .ipynb**. Keep both files on your computer.

# %%
from datetime import datetime, timezone
import shutil
from google.colab import files
stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
backup=shutil.make_archive(f'/content/XSM_full_backup_{stamp}','zip',
                         root_dir=PROJECT.parent,base_dir=PROJECT.name)
files.download(backup)
print('Keep the downloaded ZIP. Restore it with Cell 1 next time.')


# %% [markdown]
# ## Optional Cell 7 — reproduce training from the beginning
# Leave the switch False for normal recovery. Set True only to reproduce the original simulation
# experiment. It generates the same seeded simulations, fits RF and XGBoost, selects on validation,
# and writes a new results folder. This is not an independent new test. Original models stay intact.
# Current XGBoost: 600 maximum rounds, learning_rate=0.05, depth=3, min_child_weight=3,
# subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5, patience=30, logloss.
# After any new model development, use a new frozen evaluation protocol and fresh test observations.

# %%
RETRAIN_FROM_SCRATCH=False
if RETRAIN_FROM_SCRATCH:
    run_xsm_action('retrain')
else:
    print('Skipped: your trained models are already restored.')


# %% [markdown]
# ## Optional Cell 8 — score the included July observation
# Exploratory review priorities only. It does not create reviewed labels or measure real accuracy.
# Re-run Cell 6 afterwards to keep the new scores.

# %%
SCORE_JULY=False
if SCORE_JULY:
    run_xsm_action('score-july')
else:
    print('Skipped optional real-data scoring.')


# %% [markdown]
# ## Other retained capabilities
# The corrected FITS/LC loader, quality controls, extraction, FRED features and plots are in `src/`
# and `run_xsm.py`. Human-review code is `xsm_review_training.py` with its guide in
# `docs/REVIEW_ADDON.md`. The real supervised trainer still requires independent reviewed labels;
# the default recovery workflow uses simulation truth only.
# 
# To continue in an existing notebook, copy Cells 1–6 from this notebook beneath your existing
# cells and run them in order. The same cells are also in `CURRENT_NOTEBOOK_CELLS.py`.
# Historical guides under `docs/` describe older setup cells; use this recovery notebook for setup.