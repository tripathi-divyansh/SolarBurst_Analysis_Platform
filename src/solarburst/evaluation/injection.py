"""
Noise-correct synthetic solar flare injection into real observational backgrounds.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from astropy.time import Time, TimeDelta
from ..core.schema import LightCurve
from ..core.time_utils import met_to_astropy_time
from ..fitting.models import piecewise_gauss_exp, emg_convolution_bin_averaged


def inject_flare_counts(
    real_counts: np.ndarray,
    exposure_s: np.ndarray,
    flare_rate_mean: np.ndarray,
    valid_mask: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Noise-correct injection of Poisson counts into real background observations.
    CRITICAL: Does NOT resample the background (which already contains its own noise).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    out = np.asarray(real_counts, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool)
    
    mu = np.maximum(flare_rate_mean[valid], 0.0) * exposure_s[valid]
    poisson_counts = rng.poisson(mu)
    out[valid] += poisson_counts
    return out


def generate_synthetic_flare_profile(
    time_met: np.ndarray,
    peak_time_met: float,
    amplitude: float,
    rise_s: float,
    decay_s: float,
    shape_family: str = "emg"
) -> np.ndarray:
    """
    Generate synthetic flare count rate profile for injection.
    
    Shape families:
      - 'emg': Gaussian heating convolved with exponential cooling
      - 'piecewise': Gaussian rise + exponential decay
      - 'multi_pulse': Two closely spaced pulses
      - 'broad': Long-duration gradual event
    """
    dt = np.median(np.diff(time_met)) if len(time_met) > 1 else 1.0
    n = len(time_met)
    rate = np.zeros(n, dtype=np.float64)

    if shape_family == "emg":
        # Fluence roughly amplitude * (rise_s + decay_s)
        fluence = amplitude * (rise_s * np.sqrt(np.pi / 2.0) + decay_s)
        lo = time_met - dt / 2.0
        hi = time_met + dt / 2.0
        # Use bin-averaged EMG
        rate = emg_convolution_bin_averaged(
            lo, hi, fluence, mu=peak_time_met - rise_s * 0.5,
            sigma_h=rise_s, tau_c=decay_s, b0=0.0, b1=0.0
        )

    elif shape_family == "piecewise":
        rate = piecewise_gauss_exp(
            time_met, amplitude, peak_time_met,
            sigma_rise=rise_s, tau_decay=decay_s, b0=0.0, b1=0.0
        )

    elif shape_family == "multi_pulse":
        p1 = piecewise_gauss_exp(
            time_met, amplitude, peak_time_met,
            sigma_rise=rise_s, tau_decay=decay_s, b0=0.0, b1=0.0
        )
        p2 = piecewise_gauss_exp(
            time_met, amplitude * 0.6, peak_time_met + decay_s * 0.8,
            sigma_rise=rise_s * 0.8, tau_decay=decay_s * 1.2, b0=0.0, b1=0.0
        )
        rate = p1 + p2

    else:
        # Fallback to piecewise
        rate = piecewise_gauss_exp(
            time_met, amplitude, peak_time_met,
            sigma_rise=rise_s, tau_decay=decay_s, b0=0.0, b1=0.0
        )

    return np.maximum(rate, 0.0)


def create_synthetic_demo_lightcurve(
    duration_s: float = 7200.0,
    cadence_s: float = 1.0,
    background_rate: float = 150.0,
    rng_seed: int = 42
) -> Tuple[LightCurve, List[Dict[str, Any]]]:
    """
    Generate a realistic synthetic Chandrayaan-2 XSM observation with known injected flares.
    Includes quiet background, Poisson noise, orbital trend, and diverse flare morphologies.
    """
    rng = np.random.default_rng(rng_seed)
    n_points = int(duration_s / cadence_s)
    
    t_start_met = 150000000.0  # arbitrary XSM MET ~ 2021
    time_met = t_start_met + np.arange(n_points) * cadence_s

    # Slowly varying orbital background (~120 min orbit)
    orbit_period_s = 7200.0
    orbital_trend = 15.0 * np.sin(2.0 * np.pi * (time_met - t_start_met) / orbit_period_s)
    bg_rate_series = background_rate + orbital_trend

    # Generate synthetic background counts
    raw_counts = rng.poisson(bg_rate_series * cadence_s).astype(np.float64)

    # Injected flares catalog
    injected_flares = [
        # 1. Strong isolated impulsive flare (Fast-rise/slow-decay)
        {
            "id": "INJ_01",
            "type": "isolated_impulsive",
            "peak_time_met": t_start_met + 1200.0,
            "amplitude": 450.0,
            "rise_s": 25.0,
            "decay_s": 180.0,
            "shape_family": "emg"
        },
        # 2. Overlapping double flare complex
        {
            "id": "INJ_02",
            "type": "overlapping_complex",
            "peak_time_met": t_start_met + 3200.0,
            "amplitude": 320.0,
            "rise_s": 35.0,
            "decay_s": 220.0,
            "shape_family": "multi_pulse"
        },
        # 3. Weak near-threshold flare (SNR ~ 4)
        {
            "id": "INJ_03",
            "type": "weak_threshold",
            "peak_time_met": t_start_met + 5000.0,
            "amplitude": 65.0,
            "rise_s": 15.0,
            "decay_s": 90.0,
            "shape_family": "emg"
        },
        # 4. Broad gradual flare (Duration > 10 min)
        {
            "id": "INJ_04",
            "type": "broad_gradual",
            "peak_time_met": t_start_met + 6200.0,
            "amplitude": 200.0,
            "rise_s": 150.0,
            "decay_s": 500.0,
            "shape_family": "piecewise"
        }
    ]

    valid_mask = np.ones(n_points, dtype=bool)
    # Add a short 60s telemetry gap
    gap_idx = np.where((time_met >= t_start_met + 4200.0) & (time_met <= t_start_met + 4260.0))[0]
    valid_mask[gap_idx] = False
    raw_counts[gap_idx] = np.nan

    # Inject flares with noise-correct Poisson counts
    for fl in injected_flares:
        flare_profile = generate_synthetic_flare_profile(
            time_met,
            peak_time_met=fl["peak_time_met"],
            amplitude=fl["amplitude"],
            rise_s=fl["rise_s"],
            decay_s=fl["decay_s"],
            shape_family=fl["shape_family"]
        )
        raw_counts = inject_flare_counts(
            real_counts=raw_counts,
            exposure_s=np.full(n_points, cadence_s),
            flare_rate_mean=flare_profile,
            valid_mask=valid_mask,
            rng=rng
        )

    # Compute observed rate and error
    rate = raw_counts / cadence_s
    error = np.sqrt(np.maximum(raw_counts, 1.0)) / cadence_s

    # Convert to Astropy Time
    t_obj = met_to_astropy_time(time_met)
    half_bin = TimeDelta(cadence_s / 2.0, format="sec")
    bin_start = t_obj - half_bin
    bin_end = t_obj + half_bin
    exposure_arr = np.where(valid_mask, cadence_s, 0.0)

    quality = np.zeros(n_points, dtype=np.uint32)
    quality[~valid_mask] = 1  # INVALID / GAP

    metadata = {
        "instrument": "XSM",
        "telescope": "Chandrayaan-2",
        "quantity": "count_rate",
        "unit": "count / s",
        "energy_band_keV": [1.0, 15.0],
        "time_scale": "utc",
        "original_epoch": "2017-01-01T00:00:00",
        "processing_version": "SolarBurst-SyntheticDemo-v1.0",
        "calibration_version": "v1.5",
        "error_kind": "poisson",
        "source_sha256": "demo_synthetic_sha256_verified",
        "source_filepath": "synthetic_xsm_demo.fits",
        "is_synthetic_demonstration": True,
        "injected_flares_count": len(injected_flares),
    }

    lc = LightCurve(
        time=t_obj,
        value=rate,
        error=error,
        quality=quality,
        bin_start=bin_start,
        bin_end=bin_end,
        exposure_s=exposure_arr,
        metadata=metadata
    )

    return lc, injected_flares
