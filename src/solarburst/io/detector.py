"""
Format auto-detection and file inspection based on magic bytes and structural signatures.
"""

import os
import zipfile
import hashlib
from typing import Dict, Any, List


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file for provenance."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def detect_file_format(filepath: str) -> Dict[str, Any]:
    """
    Detect the physical file format using magic bytes, archive contents, and fallback extensions.
    
    Returns:
    --------
    dict with keys: 'format', 'is_compressed', 'sha256', 'size_bytes', 'details'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    size_bytes = os.path.getsize(filepath)
    sha256 = compute_sha256(filepath)
    
    with open(filepath, "rb") as f:
        header = f.read(2048)

    # 1. Check FITS
    # Standard FITS starts with "SIMPLE  ="
    if header.startswith(b"SIMPLE  ="):
        return {
            "format": "fits",
            "is_compressed": False,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "details": "Standard uncompressed FITS"
        }

    # 2. Check Gzip compressed FITS
    if header.startswith(b"\x1f\x8b"):
        # Check if internal content is FITS
        import gzip
        try:
            with gzip.open(filepath, "rb") as gz:
                inner_header = gz.read(80)
                if inner_header.startswith(b"SIMPLE  ="):
                    return {
                        "format": "fits_gz",
                        "is_compressed": True,
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "details": "Gzip-compressed FITS (.fits.gz / .lc.gz)"
                    }
        except Exception:
            pass

    # 3. Check NASA CDF
    # CDF magic numbers: 0x0000FFFF or 0xCDF0001
    if header.startswith(b"\x00\x00\xff\xff") or header.startswith(b"\xcd\xf0\x00\x01") or header.startswith(b"\xcd\xf2\x60\x02"):
        return {
            "format": "cdf",
            "is_compressed": False,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "details": "NASA Common Data Format (CDF)"
        }

    # 4. Check OLE Compound File (Legacy Excel .xls)
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return {
            "format": "xls",
            "is_compressed": False,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "details": "Legacy Microsoft Excel Binary (.xls)"
        }

    # 5. Check ZIP archive (could be XLSX or generic zip)
    if header.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                namelist = zf.namelist()
                if "xl/workbook.xml" in namelist or "[Content_Types].xml" in namelist:
                    return {
                        "format": "xlsx",
                        "is_compressed": False,
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "details": "Microsoft Excel OpenXML (.xlsx)"
                    }
        except Exception:
            pass

    # 6. Fallback to text inspection (CSV / TSV / ASCII)
    try:
        text_sample = header.decode("utf-8", errors="replace")
        # Check for typical CSV / TSV patterns
        lines = [line.strip() for line in text_sample.splitlines() if line.strip() and not line.startswith("#")]
        if len(lines) >= 2:
            first = lines[0]
            if "," in first:
                fmt = "csv"
            elif "\t" in first:
                fmt = "tsv"
            else:
                fmt = "ascii"
            return {
                "format": fmt,
                "is_compressed": False,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "details": f"Plain text ({fmt.upper()})"
            }
    except Exception:
        pass

    # Fallback based on extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".fits", ".fit", ".lc"]:
        fmt = "fits"
    elif ext == ".csv":
        fmt = "csv"
    elif ext in [".tsv", ".tab"]:
        fmt = "tsv"
    elif ext == ".xlsx":
        fmt = "xlsx"
    elif ext == ".xls":
        fmt = "xls"
    elif ext == ".cdf":
        fmt = "cdf"
    else:
        fmt = "unknown"

    return {
        "format": fmt,
        "is_compressed": False,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "details": f"Extension-inferred format ({ext})"
    }
