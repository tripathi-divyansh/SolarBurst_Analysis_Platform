"""
Core module exports for SolarBurst.
"""

from .schema import LightCurve, Candidate, BurstResult, AnalysisConfig
from .quality import QualityFlag, is_valid_observation, create_quality_mask
from .time_utils import (
    XSM_EPOCH_ISO,
    XSM_EPOCH_TIME,
    met_to_astropy_time,
    astropy_time_to_met,
    parse_time_array
)

__all__ = [
    "LightCurve",
    "Candidate",
    "BurstResult",
    "AnalysisConfig",
    "QualityFlag",
    "is_valid_observation",
    "create_quality_mask",
    "XSM_EPOCH_ISO",
    "XSM_EPOCH_TIME",
    "met_to_astropy_time",
    "astropy_time_to_met",
    "parse_time_array",
]
