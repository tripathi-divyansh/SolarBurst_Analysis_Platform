"""
Quality bitmasks and Good Time Interval (GTI) helpers.
"""

from enum import IntFlag
import numpy as np


class QualityFlag(IntFlag):
    GOOD = 0
    INVALID = 1 << 0              # General invalid measurement
    GAP_FILL = 1 << 1             # Missing bin / synthetic fill
    FILTER_TRANSITION = 1 << 2    # Be-filter transition or detector mode switch
    SATURATED = 1 << 3            # Detector saturation / high dead-time
    OCCULTED = 1 << 4             # Sun occulted / orbital night
    HK_ANOMALY = 1 << 5           # Housekeeping / temperature anomaly
    EDGE_INTERVAL = 1 << 6        # Near observation boundary
    SPIKE_ARTIFACT = 1 << 7       # Single-bin MAD spike anomaly


def is_valid_observation(quality_mask: np.ndarray) -> np.ndarray:
    """Return boolean mask where observation is science-valid (no severe flags)."""
    # Exclude INVALID, OCCULTED, and GAP_FILL from scientific fitting
    severe_mask = (
        QualityFlag.INVALID
        | QualityFlag.OCCULTED
        | QualityFlag.GAP_FILL
    )
    return (quality_mask & int(severe_mask)) == 0


def create_quality_mask(
    length: int,
    nan_indices: np.ndarray = None,
    filter_indices: np.ndarray = None,
    saturated_indices: np.ndarray = None
) -> np.ndarray:
    """Construct an initial uint32 quality array."""
    mask = np.zeros(length, dtype=np.uint32)
    if nan_indices is not None and len(nan_indices) > 0:
        mask[nan_indices] |= int(QualityFlag.INVALID)
    if filter_indices is not None and len(filter_indices) > 0:
        mask[filter_indices] |= int(QualityFlag.FILTER_TRANSITION)
    if saturated_indices is not None and len(saturated_indices) > 0:
        mask[saturated_indices] |= int(QualityFlag.SATURATED)
    return mask
