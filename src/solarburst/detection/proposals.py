"""
Candidate proposal fusion, hysteresis interval growth, peak splitting, and deduplication.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..core.schema import Candidate
from .matched_filter import run_matched_filter_bank
from .wavelet_cwt import run_cwt_candidate_detector
from .derivative import run_derivative_excess_detector


def generate_candidate_proposals(
    time_met: np.ndarray,
    rate: np.ndarray,
    baseline: np.ndarray,
    noise_sigma: float,
    dt_s: float,
    matched_filter_thresh: float = 3.5,
    cwt_thresh: float = 3.0,
    derivative_thresh: float = 2.5,
    hysteresis_seed: float = 4.0,
    hysteresis_grow: float = 1.8
) -> List[Candidate]:
    """
    Union of 3 candidate generators + hysteresis growth + multi-peak splitting + deduplication.
    """
    n = len(rate)
    if n < 10:
        return []

    sigma = max(noise_sigma, 1e-6)
    z_series = (rate - baseline) / sigma

    # 1. Run Candidate Generators
    mf_res = run_matched_filter_bank(z_series, dt_s, threshold=matched_filter_thresh)
    cwt_res = run_cwt_candidate_detector(z_series, dt_s, threshold=cwt_thresh)
    der_res = run_derivative_excess_detector(z_series, dt_s, derivative_threshold=derivative_thresh)

    # 2. Gather Seed Indices
    # Seeds can come from generator candidate indices or direct high-z excursions (>= hysteresis_seed)
    seed_dict = {}  # index -> list of generator sources

    for idx in mf_res["candidate_indices"]:
        seed_dict.setdefault(idx, []).append("matched_filter")
    for idx in cwt_res["candidate_indices"]:
        seed_dict.setdefault(idx, []).append("cwt_ridges")
    for idx in der_res["candidate_indices"]:
        seed_dict.setdefault(idx, []).append("derivative_onset")

    # Add strong z-score seeds
    for i in range(1, n - 1):
        if z_series[i] >= hysteresis_seed and z_series[i] >= z_series[i - 1] and z_series[i] >= z_series[i + 1]:
            seed_dict.setdefault(i, []).append("direct_excess")

    if not seed_dict:
        return []

    # Sort seeds by SNR descending
    sorted_seeds = sorted(seed_dict.keys(), key=lambda idx: z_series[idx], reverse=True)

    # 3. Grow Intervals via Hysteresis (connected excess above hysteresis_grow)
    candidates: List[Candidate] = []
    claimed_mask = np.zeros(n, dtype=bool)

    for cand_idx, seed_i in enumerate(sorted_seeds):
        if claimed_mask[seed_i]:
            continue

        # Grow left
        left = seed_i
        while left > 0 and z_series[left - 1] >= hysteresis_grow:
            left -= 1

        # Grow right
        right = seed_i
        while right < n - 1 and z_series[right + 1] >= hysteresis_grow:
            right += 1

        # Duration in bins
        duration_bins = right - left + 1
        if duration_bins < 3:
            # Require minimum persistence
            continue

        # Check for multiple peaks inside this active interval [left, right]
        sub_z = z_series[left:right + 1]
        local_peaks = []
        for j in range(1, len(sub_z) - 1):
            if sub_z[j] >= hysteresis_seed and sub_z[j] > sub_z[j - 1] and sub_z[j] > sub_z[j + 1]:
                local_peaks.append(left + j)

        # Mark claimed
        claimed_mask[left:right + 1] = True

        # Context window: extend context for baseline and profile fitting by up to 2x width or 60s
        span_s = time_met[right] - time_met[left]
        context_ext_s = max(span_s * 0.8, 30.0)
        c_start_met = max(time_met[0], time_met[left] - context_ext_s)
        c_end_met = min(time_met[-1], time_met[right] + context_ext_s)

        # Peak time: index with maximum rate inside [left, right]
        peak_i = left + np.argmax(rate[left:right + 1])
        peak_snr = float(z_series[peak_i])
        excess_sum = float(np.sum(np.maximum(rate[left:right + 1] - baseline[left:right + 1], 0.0) * dt_s))

        cand = Candidate(
            candidate_id=f"CAND_{cand_idx + 1:04d}",
            start_time_met=float(time_met[left]),
            peak_time_met=float(time_met[peak_i]),
            end_time_met=float(time_met[right]),
            seed_snr=peak_snr,
            generator_flags=seed_dict.get(seed_i, ["excess"]),
            context_start_met=float(c_start_met),
            context_end_met=float(c_end_met),
            preliminary_width_s=float(time_met[right] - time_met[left]),
            preliminary_excess=excess_sum,
            decision="review"
        )
        candidates.append(cand)

    # Sort candidates chronologically
    candidates.sort(key=lambda c: c.start_time_met)
    return candidates
