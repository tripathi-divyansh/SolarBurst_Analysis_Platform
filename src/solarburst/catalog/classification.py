"""
Physically justified operational classifications: intensity, temporal morphology, duration, and reliability.
"""

from typing import Tuple, Dict, Any, Optional


def classify_intensity(
    peak_flux: float,
    quantity: str = "count_rate",
    unit: str = "count / s",
    calibrated_goes_band: bool = False
) -> str:
    """
    Intensity classification with scientific safeguards.
    CRITICAL: Never apply GOES classes (A, B, C, M, X) directly to raw XSM counts/s!
    """
    if calibrated_goes_band and quantity == "energy_flux" and "W" in unit:
        # Calibrated 0.1 - 0.8 nm solar irradiance in W/m^2
        if peak_flux < 1e-8:
            return f"< A1.0 ({peak_flux:.2e} W/m²)"
        elif peak_flux < 1e-7:
            sub = peak_flux / 1e-8
            return f"A{sub:.1f}"
        elif peak_flux < 1e-6:
            sub = peak_flux / 1e-7
            return f"B{sub:.1f}"
        elif peak_flux < 1e-5:
            sub = peak_flux / 1e-6
            return f"C{sub:.1f}"
        elif peak_flux < 1e-4:
            sub = peak_flux / 1e-5
            return f"M{sub:.1f}"
        else:
            sub = peak_flux / 1e-4
            return f"X{sub:.1f}"
    else:
        # XSM count rate or uncalibrated signal
        # Classify by logarithmic count-rate tier without false GOES claim
        if peak_flux <= 0:
            return "Below baseline"
        log_val = int(peak_flux)
        return f"Rate Tier ~{log_val:d} {unit} (Uncalibrated XSM count rate)"


def classify_morphology(asymmetry_rho: float) -> str:
    """
    Temporal morphology label based on operational asymmetry ratio rho:
      rho = (t_peak - t_rise,10%) / (t_decay,10% - t_peak)
    """
    if asymmetry_rho < 0.5:
        return "Fast-rise/slow-decay"
    elif asymmetry_rho <= 2.0:
        return "Approximately symmetric"
    else:
        return "Slow-rise/fast-decay"


def classify_duration(duration_s: float) -> str:
    """
    Operational duration bins:
      - Short: < 5 min (300 s)
      - Intermediate: 5 - 30 min (300 - 1800 s)
      - Long: > 30 min (1800 s)
    """
    if duration_s < 300.0:
        return "Short (<5 min)"
    elif duration_s <= 1800.0:
        return "Intermediate (5-30 min)"
    else:
        return "Long (>30 min)"


def classify_reliability(
    chi2_reduced: float,
    peak_snr: float,
    edge_proximity_s: float,
    duration_s: float
) -> str:
    """
    Assign reliability status: reliable, poor_fit, low_snr, or truncated.
    """
    if edge_proximity_s < duration_s * 0.1:
        return "truncated"
    if peak_snr < 3.0:
        return "low_snr"
    if chi2_reduced > 5.0 or chi2_reduced < 0.1:
        return "poor_fit"
    return "reliable"
