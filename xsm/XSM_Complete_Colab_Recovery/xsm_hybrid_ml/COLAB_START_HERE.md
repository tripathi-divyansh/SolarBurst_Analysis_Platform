# XSM complete Colab recovery

1. Extract this ZIP on your computer and locate `xsm_hybrid_ml/Colab_Recovery.ipynb`.
2. Open it using Google Colab → File → Upload notebook.
   To keep your current notebook, copy cells 1–6 into it instead.
3. Run cells 1–6 in order. When cell 1 asks, upload the original complete ZIP.
4. Keep the full backup downloaded by cell 6. After any later reset, upload that
   backup in cell 1, so later saved results and reviews are restored too.

No Drive and no path edits. CPU runtime, Python 3.12+. Cell 2 needs package-download
internet access. Dependencies live in a separate Python environment so NumPy
imports in the notebook do not conflict. Scientific operations run in subprocesses.
Colab browser installation has not been tested live; local validation is recorded
in RECOVERY_VALIDATION.txt.

Included changes: corrected v0.2 FITS/LC pipeline; quality/exposure handling;
5 extra context features (peak fraction, roughness, flat-top fraction, longest
above-2 interval, positive-area/peak); fixed day splits; training-only preprocessing;
unsupervised Isolation Forest baseline; simulation-trained RF/XGBoost; saved
models; loss graph; one-to-one event evaluation; strength recovery; control false
triggers/hour; early stopping verified at round 205 (training ends 235); validation
regularization comparison; review add-on and backup workflow.

Original selected model remains unchanged. Stronger regularization did not improve
validation. XGBoost predictions use its best iteration, not the last trained round.

The supervised training is SYNTHETIC ONLY. No real accuracy is established.
The saved test has only two weak bursts. Zero false triggers in five simulated
control hours does not establish a zero real false-alarm rate.

Included real inputs: shared 280-candidate CSV, six observation groups, July 16
LC/GTI/XML and original/corrected review assets. Other original observation files,
latest Colab-only edits and reviews were not shared and cannot be recovered here.
Preserve any still-live Colab work separately before switching to this package.

Cell 6 backs up PROJECT only. Save notebook edits separately as .ipynb and put
new data/results you want backed up inside PROJECT. Do not include virtual
environments; cell 2 recreates them. Keep backups on your own computer.
