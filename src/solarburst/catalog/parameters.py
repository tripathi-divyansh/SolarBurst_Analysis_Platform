"""
Parameter synthesis and BurstResult object construction.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from astropy.time import Time
from ..core.schema import Candidate, BurstResult, LightCurve
from ..core.time_utils import met_to_astropy_time
from ..fitting.uncertainty import compute_profile_crossings, run_bootstrap_uncertainties
from .classification import (
    classify_intensity,
    classify_morphology,
    classify_duration,
    classify_reliability
)


def synthesize_burst_result(
    cand: Candidate,
    fit_res: Dict[str, Any],
    time_met: np.ndarray,
    lc: LightCurve,
    fast_uncertainty: bool = True
) -> BurstResult:
    """
    Construct canonical BurstResult from candidate proposals, profile fits, and classifications.
    """
    # Dense grid for crossing calculation
    tref = fit_res["tref_met"]
    dt = float(np.median(np.diff(time_met))) if len(time_met) > 1 else 1.0
    t_dense = np.linspace(cand.context_start_met, cand.context_end_met, 1500)
    
    # Evaluate signal component on dense grid
    from scipy.stats import exponnorm
    sigma_h = max(fit_res["heating_sigma_s"], 1e-4)
    tau_c = max(fit_res["decay_tau_s"], 1e-4)
    K = max(tau_c / sigma_h, 1e-4)
    dist = exponnorm(K=K, loc=fit_res["heating_center_met"] - tref, scale=sigma_h)
    signal_dense = fit_res["fluence"] * dist.pdf(t_dense - tref)

    crossings = compute_profile_crossings(
        t_dense=t_dense,
        signal_dense=signal_dense,
        t_peak=fit_res["fitted_peak_time_met"],
        net_peak=fit_res["net_peak"]
    )

    # Uncertainty
    uncertainties = dict(fit_res.get("uncertainties", {}))
    if not fast_uncertainty:
        # Run bootstrap realizations if requested
        lo_cand = time_met[:-1]
        hi_cand = time_met[1:]
        boot_res = run_bootstrap_uncertainties(lo_cand, hi_cand, fit_res, n_bootstrap=30)
        uncertainties.update(boot_res)

    # Classifications
    quantity = lc.metadata.get("quantity", "count_rate")
    unit = lc.metadata.get("unit", "count / s")
    calibrated_goes = bool(quantity == "energy_flux" and "W" in unit)

    int_class = classify_intensity(fit_res["net_peak"], quantity, unit, calibrated_goes)
    morph_class = classify_morphology(crossings["asymmetry_rho"])
    dur_class = classify_duration(crossings["duration_s"])
    
    edge_proximity = min(cand.start_time_met - time_met[0], time_met[-1] - cand.end_time_met)
    noise_sigma = float(np.nanmedian(lc.error)) if np.any(np.isfinite(lc.error)) else 1.0
    peak_snr = float(fit_res["net_peak"] / max(noise_sigma, 1e-6))
    
    rel_status = classify_reliability(
        chi2_reduced=fit_res["chi2_reduced"],
        peak_snr=peak_snr,
        edge_proximity_s=edge_proximity,
        duration_s=crossings["duration_s"]
    )

    # Timing ISO strings
    t_peak_iso = met_to_astropy_time(fit_res["fitted_peak_time_met"]).isot
    t_start_iso = met_to_astropy_time(crossings["start_time_met"]).isot
    t_end_iso = met_to_astropy_time(crossings["end_time_met"]).isot

    burst_id = f"BST_{cand.candidate_id.split('_')[-1]}"

    return BurstResult(
        burst_id=burst_id,
        candidate_id=cand.candidate_id,
        parent_id=None,
        peak_time_met=float(fit_res["fitted_peak_time_met"]),
        peak_time_iso=t_peak_iso,
        heating_center_met=float(fit_res["heating_center_met"]),
        start_time_met=float(crossings["start_time_met"]),
        start_time_iso=t_start_iso,
        end_time_met=float(crossings["end_time_met"]),
        end_time_iso=t_end_iso,
        rise_time_s=float(crossings["rise_time_s"]),
        decay_duration_s=float(crossings["decay_duration_s"]),
        decay_tau_s=float(fit_res["decay_tau_s"]),
        fwhm_s=float(crossings["fwhm_s"]),
        duration_s=float(crossings["duration_s"]),
        net_peak=float(fit_res["net_peak"]),
        total_peak=float(fit_res["total_peak"]),
        background_at_peak=float(fit_res["total_peak"] - fit_res["net_peak"]),
        background_slope=float(fit_res["background_slope"]),
        peak_snr=float(peak_snr),
        fluence=float(fit_res["fluence"]),
        asymmetry_rho=float(crossings["asymmetry_rho"]),
        morphology_class=morph_class,
        duration_class=dur_class,
        intensity_class=int_class,
        complexity="isolated" if fit_res.get("num_components", 1) == 1 else "overlapping",
        reliability=rel_status,
        fit_model="emg_convolution",
        reduced_chi2=float(fit_res["chi2_reduced"]),
        aicc=float(fit_res["aicc"]),
        bic=float(fit_res["bic"]),
        r_squared=float(fit_res["r_squared"]),
        ml_score=float(cand.ml_score if cand.ml_score is not None else 0.5),
        decision=cand.decision,
        uncertainties=uncertainties,
        quality_flags=[]
    )
