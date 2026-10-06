"""
Evaluation module exports for SolarBurst.
"""

from .injection import inject_flare_counts, generate_synthetic_flare_profile, create_synthetic_demo_lightcurve
from .matching import match_events_one_to_one
from .metrics import compute_detection_metrics, compute_parameter_recovery_errors

__all__ = [
    "inject_flare_counts",
    "generate_synthetic_flare_profile",
    "create_synthetic_demo_lightcurve",
    "match_events_one_to_one",
    "compute_detection_metrics",
    "compute_parameter_recovery_errors",
]
