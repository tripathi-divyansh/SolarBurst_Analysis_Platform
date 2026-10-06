"""
Tests for physical profile models, bin-averaged EMG fitting, and crossings.
"""

import numpy as np
import pytest
from solarburst.fitting.models import piecewise_gauss_exp, emg_convolution_bin_averaged
from solarburst.fitting.emg_fit import fit_emg_profile
from solarburst.fitting.uncertainty import compute_profile_crossings


def test_fit_emg_profile_recovery():
    # Construct a synthetic EMG flare profile
    dt = 1.0
    t = np.arange(0, 400, dt)
    lo = t - 0.5
    hi = t + 0.5
    
    true_fluence = 5000.0
    true_mu = 100.0
    true_sigma_h = 15.0
    true_tau_c = 45.0
    true_b0 = 25.0
    true_b1 = 0.01

    clean_signal = emg_convolution_bin_averaged(
        lo, hi,
        fluence=true_fluence,
        mu=true_mu,
        sigma_h=true_sigma_h,
        tau_c=true_tau_c,
        b0=true_b0,
        b1=true_b1,
        t_ref=200.0
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.5, size=len(t))
    noisy_y = clean_signal + noise
    err = np.full_like(noisy_y, 1.5)

    res = fit_emg_profile(lo, hi, noisy_y, err)

    assert res["optimizer_success"]
    assert abs(res["heating_center_met"] - true_mu) < 5.0
    assert abs(res["decay_tau_s"] - true_tau_c) < 10.0
    assert abs(res["fluence"] - true_fluence) / true_fluence < 0.15
    assert res["chi2_reduced"] < 2.0


def test_crossings_calculation():
    t_dense = np.linspace(0, 200, 1000)
    # Peak at 50, rise 20, decay 60
    sig = 100.0 * np.exp(-((t_dense - 50.0) / 20.0)**2)
    sig[t_dense > 50.0] = 100.0 * np.exp(-(t_dense[t_dense > 50.0] - 50.0) / 40.0)

    res = compute_profile_crossings(t_dense, sig, t_peak=50.0, net_peak=100.0)

    assert res["start_time_met"] < 50.0
    assert res["end_time_met"] > 50.0
    assert res["duration_s"] > 40.0
    assert res["asymmetry_rho"] < 1.0  # Fast-rise/slow-decay
