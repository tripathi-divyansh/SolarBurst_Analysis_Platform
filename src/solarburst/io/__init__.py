"""
I/O module for SolarBurst.
"""

from .detector import detect_file_format, compute_sha256
from .fits_reader import read_fits_lightcurve, inspect_fits_file
from .text_reader import read_text_lightcurve, inspect_text_file
from .excel_reader import read_excel_lightcurve, inspect_excel_file
from .cdf_reader import read_cdf_lightcurve, inspect_cdf_file
from .xsm_adapter import load_xsm_fits, inspect_file, load_lightcurve

__all__ = [
    "detect_file_format",
    "compute_sha256",
    "read_fits_lightcurve",
    "inspect_fits_file",
    "read_text_lightcurve",
    "inspect_text_file",
    "read_excel_lightcurve",
    "inspect_excel_file",
    "read_cdf_lightcurve",
    "inspect_cdf_file",
    "load_xsm_fits",
    "inspect_file",
    "load_lightcurve",
]
