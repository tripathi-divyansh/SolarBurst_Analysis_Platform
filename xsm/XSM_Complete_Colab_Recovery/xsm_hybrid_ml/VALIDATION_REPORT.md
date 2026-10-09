# XSM pipeline 0.2.0 — July 16 review and software correction

## What was supplied

The user supplied a Level-2 light curve, matching GTI, and XML metadata for 16 July 2026, plus the original 42-candidate review archive. No independently reviewed event catalogue or class labels were supplied. The energy selection remains unverified: the LC HISTORY contains unusable negative energy bounds and the XML does not establish a usable selected energy range.

The light curve has 6,163 rows. One fails rate/error/exposure validity checks; nine additional bins are not fully contained in a GTI. The retained 6,153 bins form 11 continuous intervals, with 6,120.199950397015 seconds of effective exposure. The 10 GTI intervals and one invalid row account for the interval splitting. Missing periods are never interpolated into measured observations.

LC SHA-256: `733be91d8a060eba9d73b371efebda5f0c22676322dc7c226384b6359c1ef75c`.

GTI SHA-256: `93afb88286c57750f9bc284f42580fa4842bdd021c411cb195f3e019448249e5`.

## Reproduction and defects

The unchanged 0.1 engine reproduced all 42 candidate times, the 22 single-bin candidates, the detection routes, and processed exposure from the uploaded review. Numerical optimizer outputs can vary slightly between environments. Of the original candidates, 41 were seeded only by Bayesian Blocks, one by the pointwise sigma route, and none by wavelets. In the uploaded fit results, 30 of 40 numerically successful fits placed their fitted peak outside the detected candidate interval.

The earlier background algorithm used an accumulating positive-outlier mask and endpoint-padded median smoothing. This could underestimate gradually rising interval ends. A positive Bayesian Block then seeded individual pointwise excursions inside it, fragmenting a broad offset into many small candidates. The unconstrained FRED fit could describe a background trend rather than the candidate it was supposed to characterize.

## Changes

- A symmetric robust local-linear background retains a local slope at boundaries and avoids the accumulating positive-only mask. It is an offline method; broad events comparable with its smoothing window may still be partly absorbed.
- Significant Bayesian Block support stays connected during segmentation. A constant offset with no inferred change point does not create a BB-only event. Edge blocks may still generate candidates, flagged for boundary review.
- Candidate-associated FRED peaks are constrained to the candidate interval. Single-bin events are retained for review but receive no resolved FRED parameters.
- FRED fits are compared with a linear-background model on the same samples and likelihood. A nonpositive BIC improvement is flagged; BIC is a diagnostic, not a calibrated event probability.
- Unreliable physical-fit parameters become missing values in ML features; full diagnostic fit output remains available for review. A successful optimizer is not equivalent to a trustworthy event characterization.
- The new explicit XSM LC adapter checks file completeness, bin-time convention, units, time references, exposure and external GTIs. Rates and measurement errors are not corrected a second time.
- Implementation and feature versions change pipeline fingerprints, preventing silent reuse of old candidate tables and trained bundles.
- A fresh-session Colab notebook restores the included data and protects previous review files by writing each run to a new directory. Its backup cell saves the entire project.

Numeric detection thresholds and the 301-second background window were unchanged. Changes were selected after examining this observation, so this day is development data and must not be presented as an independent evaluation set.

## Actual-data result

| Measure | Original run | Corrected run |
|---|---:|---:|
| Candidates | 42 | 2 |
| Single-bin candidates | 22 | 1 |
| BB-only candidates | 41 | 0 |
| Processed exposure (seconds) | 6120.20 | 6120.20 |
| Confirmed event labels | None supplied | None assigned |

The corrected candidates peak at 749.5 and 7427.5 seconds relative to MET origin 300933166.64688903 seconds. Their detected intervals last three seconds and one second, respectively. The three-second candidate has unresolved fitted timescales and does not improve BIC over a linear background (delta BIC approximately -1.824). The single-bin candidate has insufficient event bins for physical timing estimates. Both remain `review_unscored` with blank labels. Neither supplies a reliable physical-parameter target for regression.

Reducing candidate count does not establish better real-data accuracy, prove the removed candidates were false, or prove that no other solar events occurred. Gradual trends and short edge intervals remain scientifically ambiguous without instrument-state review and independent event references.

## Verification performed

- **45 tests passed** on Linux, Python 3.12, using `requirements-tested.txt`; see `test_run_v02.txt`.
- Tests include slope recovery at interval boundaries, constant-offset rejection by BB, preservation of broad weak BB intervals without fragmentation, physical peak constraints, single-bin handling, gap/exposure preservation, actionable truncation errors, numerical FRED integration and fitting, and ML leakage/provenance/save-load checks.
- Controlled signal checks used 16 generated observations with 32 known planted events: all 32 matched a proposal within 35 seconds. There were also 39 other proposals, including known simulated artifacts. This is not an event-classification accuracy result.
- Twenty additional curved-trend noise controls, totalling 16,000 seconds, produced five proposals with no injected events. These results demonstrate that the revised detector still proposes noise excursions. They are not an operational real-XSM false-alarm estimate.
- The July runner was executed on the supplied LC/GTI and produced the included results and plots. The notebook Python cells were syntax-checked; the Colab-specific upload/download UI was not exercised in a live Colab account. Dependencies and engine were executed under Python 3.12; the user's Python 3.13 Colab runtime was not independently tested here.

## Before supervised training

Obtain reviewed positive and negative examples across independent observing periods. Review whole intervals for missed events, not just proposals. Do not label uncertain cases as non-bursts. Do not use detector thresholds or these fit warnings as ground truth. Keep all related observations/augmentations in the same partition. This implementation requires separate train/tune/calibration/threshold/test partitions with at least two candidate-bearing groups and both classes in each partition. One day with two unlabelled candidates cannot satisfy this.

No real-data classifier has been trained by this correction. Old version-0.1 synthetic weights are intentionally omitted from the new package. The supervised training API and synthetic demonstration generator remain available for later validated use.

## Sources and implementation references

- PRL XSM data products and analysis software: https://www.prl.res.in/ch2xsm/data_analysis
- XSMDAS 1.50 distribution, whose standard LC implementation stores bin-start times and exposure-corrected rates: https://www.prl.res.in/ch2xsm/static/ch2_xsmdas_20251121_v1.5.zip
- Astropy Bayesian Blocks API: https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html
- Local-linear robust smoothing background/reference: https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html (the package uses its own error-weighted fixed-time-window implementation, not the statsmodels function).
- Colab runtime persistence: https://research.google.com/colaboratory/faq.html
