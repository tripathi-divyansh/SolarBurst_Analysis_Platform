import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def classification_metrics(y, score, threshold=0.5):
    y, score = np.asarray(y, int), np.asarray(score, float)
    pred = score >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, score)),
            "log_loss": float(log_loss(y, np.clip(score, 1e-7, 1-1e-7), labels=[0, 1]))}


def far_stats(n_false, exposure_s):
    if not np.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError("Positive valid processed observing exposure required")
    days = exposure_s/86400
    return {"false_alarms_per_valid_day": n_false/days,
            "far_poisson_upper_95": float(0.5*chi2.ppf(0.95, 2*(n_false+1))/days),
            "exposure_s": float(exposure_s)}


def select_threshold(y, score, exposure_s, target_far):
    """Candidate-label operating point; does not account for never-proposed bursts."""
    if not np.isfinite(target_far) or target_far < 0:
        raise ValueError("target_far must be finite and nonnegative")
    y, score = np.asarray(y, int), np.asarray(score, float)
    thresholds = np.r_[np.nextafter(1.0, 2.0), np.unique(score)]
    eligible = []
    for threshold in thresholds:
        pred = score >= threshold
        fp = int(np.sum(pred & (y == 0)))
        stats = far_stats(fp, exposure_s)
        if stats["false_alarms_per_valid_day"] <= target_far:
            tp = int(np.sum(pred & (y == 1)))
            eligible.append((tp, -fp, float(threshold), stats))
    best = max(eligible, key=lambda x: (x[0], x[1], x[2]))
    return best[2], best[3]


def match_events(predicted, reference, tolerance_s=30.0):
    """One-to-one maximum-cardinality matching; overlap AND peak tolerance.

    Frames need group_id, observed_start_s, observed_peak_s, observed_end_s.
    All predictions passed here count as accepted. Duplicates become false positives.
    """
    cols = ["group_id", "observed_start_s", "observed_peak_s", "observed_end_s"]
    if tolerance_s <= 0:
        raise ValueError("Peak tolerance must be positive")
    for frame in (predicted, reference):
        if set(cols)-set(frame): raise ValueError("Missing event boundary columns")
        if not np.isfinite(frame[cols[1:]].to_numpy(float)).all(): raise ValueError("Nonfinite event times")
    pairs = []
    # Use positional indices independent of caller DataFrame index labels.
    for group in set(predicted.group_id) | set(reference.group_id):
        pi = np.flatnonzero(predicted.group_id.to_numpy() == group)
        ri = np.flatnonzero(reference.group_id.to_numpy() == group)
        if not len(pi) or not len(ri): continue
        p, r = predicted.iloc[pi], reference.iloc[ri]
        delta = np.abs(p.observed_peak_s.to_numpy()[:, None]-r.observed_peak_s.to_numpy()[None, :])
        overlap = np.minimum(p.observed_end_s.to_numpy()[:, None], r.observed_end_s.to_numpy()[None, :]) > np.maximum(p.observed_start_s.to_numpy()[:, None], r.observed_start_s.to_numpy()[None, :])
        ok = overlap & (delta <= tolerance_s)
        cost = np.where(ok, -1+delta/tolerance_s*1e-3, 0)
        a, b = linear_sum_assignment(cost)
        pairs.extend((int(pi[i]), int(ri[j])) for i, j in zip(a, b) if ok[i, j])
    tp = len(pairs)
    return {"pairs": pairs, "tp": tp, "fp": len(predicted)-tp, "fn": len(reference)-tp,
            "precision": tp/len(predicted) if len(predicted) else 0.0,
            "recall": tp/len(reference) if len(reference) else 0.0}
