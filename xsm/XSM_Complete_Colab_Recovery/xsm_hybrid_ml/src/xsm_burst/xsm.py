"""Explicit adapter for standard XSMDAS 1.50 Level-2 light curves.

XSMDAS 1.50 source: xsmgenlc.cpp stores bin starts; FRACEXP includes the
applied dead-time factor. Rates/errors are already corrected. No L1 decoder.
"""
from pathlib import Path
import hashlib
import numpy as np
from astropy.io import fits
from .schema import LightCurve


def _read_table(path, extension):
    path = Path(path)
    try:
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[extension]
            info = hdu.fileinfo()
            if path.suffix.lower() != '.gz' and info:
                required = info['datLoc'] + info['datSpan']
                if path.stat().st_size < required:
                    raise ValueError(f'Truncated FITS file: {path.name}; re-extract or re-upload the original file.')
            header = hdul[0].header.copy()
            header.update(hdu.header)
            table = {name.upper().strip(): np.array(hdu.data[name], dtype=float)
                     for name in hdu.columns.names}
            return header, table
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f'Could not read {extension} table in {path.name}: {exc}') from exc


def _reference(h):
    unit = str(h.get('TIMEUNIT', 's')).strip().lower()
    scale = str(h.get('TIMESYS', '')).strip().lower()
    if unit != 's' or scale != 'utc':
        raise ValueError('This adapter requires UTC mission elapsed time in seconds.')
    mjd = h.get('MJDREF')
    if mjd is None:
        if 'MJDREFI' not in h:
            raise ValueError('Missing MJD time reference')
        mjd = h['MJDREFI'] + h.get('MJDREFF', 0.)
    return float(mjd), scale, float(h.get('TIMEZERO', 0.))


def load_xsm_level2(path, gti_path=None):
    path = Path(path)
    if gti_path is None:
        base = path.with_suffix('') if path.suffix.lower() == '.gz' else path
        gti_path = base.with_suffix('.gti')
    gti_path = Path(gti_path)
    h, data = _read_table(path, 'RATE')
    gh, gti = _read_table(gti_path, 'GTI')
    if not {'TIME', 'RATE', 'ERROR', 'FRACEXP'} <= set(data):
        raise ValueError('Expected a Level-2 LC with TIME, RATE, ERROR, FRACEXP; raw L1 is not supported.')
    mjd, scale, tz = _reference(h)
    gmjd, gscale, gtz = _reference(gh)
    if mjd != gmjd or scale != gscale:
        raise ValueError('LC and GTI have different time references.')
    dt = float(h.get('TIMEDEL', np.nan))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Missing/invalid TIMEDEL.')
    if 'TIMEPIXR' in h:
        pixr = float(h['TIMEPIXR'])
    elif (str(h.get('CREATOR', '')).strip().lower() == 'xsmgenlc'
          and str(h.get('XSMDASVE', '')).strip() in {'1.5', '1.50'}
          and any('lctype = standard' in str(v) for v in h.get('HISTORY', []))):
        pixr = 0.
    else:
        raise ValueError('Unverified bin-time convention. Supply a documented TIMEPIXR or a standard XSMDAS 1.50 product.')
    if not np.isfinite(pixr) or not 0 <= pixr <= 1:
        raise ValueError('Invalid TIMEPIXR.')
    time = data['TIME'] + tz
    y, err, frac = (data[k] for k in ('RATE', 'ERROR', 'FRACEXP'))
    if any(a.ndim != 1 or a.shape != time.shape for a in (time, y, err, frac)):
        raise ValueError('LC columns must be equal-length vectors.')
    if not np.isfinite(time).all():
        raise ValueError('Non-finite timestamps.')
    starts, stops = gti['START']+gtz, gti['STOP']+gtz
    if not (np.isfinite(starts).all() and np.isfinite(stops).all() and (stops > starts).all()):
        raise ValueError('Invalid GTI boundaries.')
    order = np.argsort(starts)
    if np.any(starts[order][1:] < stops[order][:-1]):
        raise ValueError('Overlapping GTIs must be resolved first.')
    left = time-pixr*dt
    origin = float(left.min())
    in_gti = np.zeros(len(time), bool)
    for a, b in zip(starts, stops):
        in_gti |= (left >= a-1e-6) & (left+dt <= b+1e-6)
    good = np.isfinite(y) & np.isfinite(err) & (err > 0) & np.isfinite(frac) & (frac > 0) & (frac <= 1)
    usable = good & in_gti
    unit = str(h.get('TUNIT2', '')).strip().replace(' ', '').lower()
    if unit not in {'count/s', 'counts/s', 'cts/s'}:
        raise ValueError(f'Unexpected rate unit: {unit!r}')
    metadata = {
        'quantity': 'count_rate', 'unit': 'count / s', 'epoch': 'relative', 'time_scale': 'relative',
        'time_origin_met_s': origin, 'original_mjdref': mjd, 'original_time_scale': scale,
        'source_file': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'gti_sha256': hashlib.sha256(gti_path.read_bytes()).hexdigest(),
        'energy_band': 'unverified', 'adapter_version': 'xsm-level2-1',
        'input_rows': len(time), 'invalid_rows': int((~good).sum()),
        'gti_excluded_otherwise_valid_rows': int((good & ~in_gti).sum()),
        'gti_intervals': len(starts), 'original_timepixr': pixr,
    }
    return LightCurve(left-origin+dt/2, y, err, np.full(len(time), dt),
                      np.where(usable, frac*dt, 0.), np.where(usable, 0, 1), metadata,
                      ['Energy band is unverified; no GOES class or physical energy flux is inferred.',
                       'Unlabelled real XSM data: detection accuracy is not established.']).prepared()
