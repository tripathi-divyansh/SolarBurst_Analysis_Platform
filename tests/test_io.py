"""
Tests for I/O format detection, Astropy time conversion, and XSM MET adapters.
"""

import os
import pytest
import numpy as np
from astropy.time import Time
from solarburst.core.time_utils import met_to_astropy_time, astropy_time_to_met, XSM_EPOCH_ISO
from solarburst.io.detector import detect_file_format
from solarburst.io.xsm_adapter import load_lightcurve


def test_xsm_met_epoch_conversion():
    # 0 MET seconds must equal 2017-01-01T00:00:00 UTC exactly
    t0 = met_to_astropy_time(0.0)
    assert t0.isot.startswith("2017-01-01T00:00:00")
    
    # Round-trip conversion test
    seconds = np.array([0.0, 1000.0, 86400.0, 1e8])
    t_obj = met_to_astropy_time(seconds)
    round_trip = astropy_time_to_met(t_obj)
    np.testing.assert_allclose(round_trip, seconds, atol=1e-6)


def test_synthetic_csv_and_fits_detection():
    csv_path = "./data/examples/synthetic_xsm_demo.csv"
    fits_path = "./data/examples/synthetic_xsm_demo.fits"

    if os.path.exists(csv_path):
        det_csv = detect_file_format(csv_path)
        assert det_csv["format"] in ["csv", "text", "ascii"]

    if os.path.exists(fits_path):
        det_fits = detect_file_format(fits_path)
        assert det_fits["format"] == "fits"


def test_load_lightcurve_validity():
    csv_path = "./data/examples/synthetic_xsm_demo.csv"
    if not os.path.exists(csv_path):
        pytest.skip("synthetic_xsm_demo.csv not found")

    lc = load_lightcurve(csv_path)
    assert len(lc.value) > 100
    assert len(lc.time) == len(lc.value)
    assert lc.metadata["instrument"] == "XSM"
    assert np.any(lc.valid_mask)
