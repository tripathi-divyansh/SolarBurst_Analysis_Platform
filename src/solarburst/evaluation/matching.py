"""
One-to-one event matching between reference ground-truth events and predicted bursts.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from ..core.schema import BurstResult


def match_events_one_to_one(
    predicted_bursts: List[BurstResult],
    reference_events: List[Dict[str, Any]],
    tolerance_sec: float = 45.0
) -> Dict[str, Any]:
    """
    Perform greedy one-to-one matching between predicted burst results and reference events.
    
    Criteria:
      - |t_peak_pred - t_peak_ref| <= tolerance_sec
      - Nonzero temporal interval overlap
    
    Returns:
    --------
    dict containing:
      - 'matched_pairs': list of (pred_burst, ref_event)
      - 'unmatched_predictions': list of false alarm BurstResult
      - 'unmatched_references': list of missed reference event dicts
    """
    matched_pairs = []
    unmatched_pred = list(predicted_bursts)
    unmatched_ref = list(reference_events)

    if not predicted_bursts or not reference_events:
        return {
            "matched_pairs": [],
            "unmatched_predictions": unmatched_pred,
            "unmatched_references": unmatched_ref
        }

    # Build cost matrix based on peak time absolute error
    costs = []
    for p in predicted_bursts:
        row = []
        for r in reference_events:
            dt = abs(p.peak_time_met - r["peak_time_met"])
            # Check overlap
            r_start = r.get("start_time_met", r["peak_time_met"] - 30.0)
            r_end = r.get("end_time_met", r["peak_time_met"] + 90.0)
            overlap = max(0.0, min(p.end_time_met, r_end) - max(p.start_time_met, r_start))
            
            if dt <= tolerance_sec and overlap > 0.0:
                row.append(dt)
            else:
                row.append(np.inf)
        costs.append(row)

    cost_mat = np.array(costs)
    
    # Greedy lowest-cost matching
    used_pred = set()
    used_ref = set()

    flat_indices = np.argsort(cost_mat.flatten())
    n_refs = len(reference_events)

    for idx in flat_indices:
        p_idx = idx // n_refs
        r_idx = idx % n_refs
        if cost_mat[p_idx, r_idx] == np.inf:
            break
        if p_idx not in used_pred and r_idx not in used_ref:
            used_pred.add(p_idx)
            used_ref.add(r_idx)
            matched_pairs.append({
                "predicted": predicted_bursts[p_idx],
                "reference": reference_events[r_idx],
                "delta_t_peak_s": float(predicted_bursts[p_idx].peak_time_met - reference_events[r_idx]["peak_time_met"])
            })

    unmatched_predictions = [p for i, p in enumerate(predicted_bursts) if i not in used_pred]
    unmatched_references = [r for j, r in enumerate(reference_events) if j not in used_ref]

    return {
        "matched_pairs": matched_pairs,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_references": unmatched_references,
    }
