"""
Detection module exports for SolarBurst.
"""

from .matched_filter import generate_template, run_matched_filter_bank
from .wavelet_cwt import run_cwt_candidate_detector, compute_scalogram
from .derivative import run_derivative_excess_detector
from .proposals import generate_candidate_proposals

__all__ = [
    "generate_template",
    "run_matched_filter_bank",
    "run_cwt_candidate_detector",
    "compute_scalogram",
    "run_derivative_excess_detector",
    "generate_candidate_proposals",
]
