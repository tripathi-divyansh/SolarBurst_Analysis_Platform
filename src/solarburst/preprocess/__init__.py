"""
Preprocessing module exports for SolarBurst.
"""

from .gti import find_contiguous_segments, compute_coverage_summary
from .binning import regularize_detection_series
from .noise import estimate_robust_noise, hampel_spike_screen, compute_standardized_series
from .baseline import asymmetric_least_squares, iterative_masked_baseline

__all__ = [
    "find_contiguous_segments",
    "compute_coverage_summary",
    "regularize_detection_series",
    "estimate_robust_noise",
    "hampel_spike_screen",
    "compute_standardized_series",
    "asymmetric_least_squares",
    "iterative_masked_baseline",
]
