"""
Canonical schemas for SolarBurst: LightCurve, Candidate, BurstResult, and AnalysisConfig.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import numpy as np
from astropy.time import Time
from .quality import QualityFlag, is_valid_observation
from .time_utils import met_to_astropy_time, astropy_time_to_met


@dataclass
class LightCurve:
    """Canonical representation of an X-ray solar light curve."""
    time: Time                 # astropy.time.Time (bin centers or timestamps)
    value: np.ndarray          # original physical units (counts, count_rate, or energy_flux)
    error: np.ndarray          # 1-sigma uncertainty (NaN if unavailable)
    quality: np.ndarray        # internal uint32 bitmask
    bin_start: Time            # start of each observation bin
    bin_end: Time              # end of each observation bin
    exposure_s: np.ndarray     # effective live exposure time per bin in seconds
    metadata: Dict[str, Any]   # standardized metadata dictionary

    def __post_init__(self):
        n = len(self.value)
        if len(self.time) != n:
            raise ValueError(f"Time length {len(self.time)} does not match value length {n}")
        if self.error is None:
            self.error = np.full(n, np.nan, dtype=np.float64)
        if self.quality is None:
            self.quality = np.zeros(n, dtype=np.uint32)
        if self.exposure_s is None:
            # Estimate exposure from bin width if available
            dt = (self.bin_end - self.bin_start).to_value("sec") if self.bin_start is not None else np.ones(n)
            self.exposure_s = np.asarray(dt, dtype=np.float64)

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of valid, non-gap, finite observations."""
        finite = np.isfinite(self.value)
        valid_q = is_valid_observation(self.quality)
        return finite & valid_q

    @property
    def met_seconds(self) -> np.ndarray:
        """XSM Mission Elapsed Time in seconds."""
        return astropy_time_to_met(self.time)

    def to_dict(self, max_points: Optional[int] = None) -> Dict[str, Any]:
        """Convert to serializable dictionary for JSON API responses."""
        n = len(self.value)
        step = 1
        if max_points is not None and n > max_points:
            step = int(np.ceil(n / max_points))

        indices = np.arange(0, n, step)
        
        # Subsample arrays
        sub_time = self.time[indices]
        sub_val = self.value[indices]
        sub_err = self.error[indices]
        sub_qual = self.quality[indices]
        sub_exp = self.exposure_s[indices]
        sub_met = astropy_time_to_met(sub_time)

        return {
            "total_points": n,
            "sampled_points": len(indices),
            "step": step,
            "met_seconds": sub_met.tolist(),
            "time_iso": [t.isot for t in sub_time],
            "value": [float(x) if np.isfinite(x) else None for x in sub_val],
            "error": [float(x) if np.isfinite(x) else None for x in sub_err],
            "quality": [int(x) for x in sub_qual],
            "exposure_s": [float(x) if np.isfinite(x) else None for x in sub_exp],
            "metadata": self.metadata,
        }


@dataclass
class Candidate:
    """A proposed burst candidate before physical fitting."""
    candidate_id: str
    start_time_met: float
    peak_time_met: float
    end_time_met: float
    seed_snr: float
    generator_flags: List[str]  # e.g. ['matched_filter', 'cwt', 'derivative']
    context_start_met: float
    context_end_met: float
    preliminary_width_s: float = 0.0
    preliminary_excess: float = 0.0
    ml_score: Optional[float] = None
    decision: str = "review"     # 'accepted' | 'review' | 'rejected'
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class BurstResult:
    """Fully analyzed, fitted, and classified solar burst."""
    burst_id: str
    candidate_id: str
    parent_id: Optional[str]
    
    # Timing (MET seconds and ISO strings)
    peak_time_met: float
    peak_time_iso: str
    heating_center_met: float
    start_time_met: float
    start_time_iso: str
    end_time_met: float
    end_time_iso: str
    
    # Durations (seconds)
    rise_time_s: float
    decay_duration_s: float
    decay_tau_s: float
    fwhm_s: float
    duration_s: float
    
    # Amplitudes and Physical Quantities
    net_peak: float
    total_peak: float
    background_at_peak: float
    background_slope: float
    peak_snr: float
    fluence: float
    
    # Morphology and Classifications
    asymmetry_rho: float
    morphology_class: str       # 'Fast-rise/slow-decay' | 'Approximately symmetric' | 'Slow-rise/fast-decay'
    duration_class: str         # 'Short' | 'Intermediate' | 'Long'
    intensity_class: str        # Calibrated GOES band or uncalibrated count-rate tier
    complexity: str             # 'isolated' | 'overlapping' | 'multi-peak' | 'unresolved'
    reliability: str            # 'reliable' | 'poor_fit' | 'truncated' | 'low_snr'
    
    # Model and Fitting Diagnostics
    fit_model: str              # 'emg_convolution' | 'piecewise_gaussian_exp'
    reduced_chi2: float
    aicc: float
    bic: float
    r_squared: float
    ml_score: float
    decision: str               # 'accepted' | 'review' | 'rejected'
    
    # Uncertainties and flags
    uncertainties: Dict[str, float] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    audit_status: str = "auto"   # 'auto' | 'approved' | 'rejected'
    audit_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisConfig:
    """Analysis configuration preset settings."""
    name: str = "balanced"
    # Preprocessing
    asls_lambda: float = 1e6
    asls_p: float = 0.01
    baseline_iterations: int = 3
    # Candidate proposal thresholds
    matched_filter_snr_threshold: float = 3.5
    cwt_snr_threshold: float = 3.0
    derivative_threshold: float = 2.5
    hysteresis_seed_sigma: float = 4.0
    hysteresis_grow_sigma: float = 1.8
    # ML Scoring
    model_family: str = "random_forest"  # 'random_forest' | 'xgboost' | 'classical_only'
    ml_accept_threshold: float = 0.65
    ml_review_threshold: float = 0.35
    # Fitting
    fit_profile: str = "emg_convolution"  # 'emg_convolution' | 'piecewise_gaussian_exp'
    max_components: int = 3
    bootstrap_samples: int = 100
    fast_uncertainty: bool = True
