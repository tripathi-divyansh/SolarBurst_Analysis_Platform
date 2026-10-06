"""
Tests for noise estimation, AsLS baseline, and regularized detection series.
"""

import numpy as np
import pytest
from solarburst.preprocess.noise import estimate_robust_noise, hampel_spike_screen
from solarburst.preprocess.baseline import asymmetric_least_squares, iterative_masked_baseline


def test_robust_noise_estimation():
    rng = np.random.default_rng(42)
    # Pure Gaussian noise with sigma = 3.5
    true_sigma = 3.5
    noise = rng.normal(0, true_sigma, size=1000)
    signal = 100.0 + noise
    mask_valid = np.ones(1000, dtype=bool)

    est_sigma = estimate_robust_noise(signal, mask_valid)
    assert abs(est_sigma - true_sigma) < 0.3


def test_asls_baseline_downweights_peaks():
    # Constant baseline at 50.0 + single huge flare peak up to 500.0
    x = np.linspace(0, 100, 500)
    bg_true = 50.0 + 2.0 * np.sin(x / 10.0)
    signal = bg_true.copy()
    
    # Add flare
    flare_center = 250
    signal[flare_center-20:flare_center+40] += 300.0 * np.exp(-((np.arange(60)-20)/15.0)**2)

    mask_valid = np.ones(500, dtype=bool)
    b_est = asymmetric_least_squares(signal, mask_valid, lam=1e5, p=0.01)

    # Baseline should NOT be dragged up to 300; it should stay close to 50
    assert np.max(b_est) < 80.0
    assert np.min(b_est) > 40.0
