"""Permissive candidate generation. Thresholds are not false-alarm guarantees."""
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d, binary_dilation, label
from astropy.stats import bayesian_blocks
import pywt


def robust_noise(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return max(float(1.4826*np.median(np.abs(x-med))), np.finfo(float).eps)


def estimate_background(t, y, err, window_s):
    """Symmetric robust local-linear background, evaluated within one segment.

    Truncated local windows retain a slope at boundaries; no endpoint padding or
    one-sided, accumulating positive mask is used. Long events comparable with
    the smoothing window can still enter this background: inspect full intervals.
    This is an offline smoother, not a causal forecasting background.
    """
    t, y, err = map(lambda a: np.asarray(a, float), (t, y, err))
    dt = float(np.median(np.diff(t)))
    half = max(window_s / 2, 3 * dt)
    neighborhoods = []
    for ti in t:
        left, right = np.searchsorted(t, [ti-half, ti+half], side="left")
        ids = np.arange(left, right)
        x = (t[ids]-ti)/half
        kernel = np.maximum(1-np.abs(x)**3, 0)**3
        neighborhoods.append((ids, x, kernel / err[ids]**2))
    robust = np.ones(len(y))
    base = np.full(len(y), np.median(y))
    poor = False
    for iteration in range(4):
        updated = np.empty(len(y))
        for i, (ids, x, initial) in enumerate(neighborhoods):
            w = initial * robust[ids]
            a, b, c = w.sum(), np.sum(w*x), np.sum(w*x*x)
            d, e = np.sum(w*y[ids]), np.sum(w*x*y[ids])
            determinant = a*c-b*b
            if a <= 0 or determinant <= 1e-12*max(a*c, 1e-300):
                updated[i] = base[i]
                poor = True
            else:
                updated[i] = (c*d-b*e)/determinant
        base = updated
        residual = y-base
        total = robust_noise(np.diff(residual))/np.sqrt(2)
        noise = np.sqrt(err**2 + max(0., total**2-np.median(err**2)))
        u = residual/(4.685*noise)
        robust = np.maximum(1-u*u, 0)**2
        robust[np.abs(u) >= 1] = 0
    return base, noise, bool(poor or (robust > 0).mean() < 0.3)


def proposals(t, y, err, base, noise, cfg):
    dt = float(np.median(np.diff(t)))
    # Continuous but variable-width observations are supported for fits. Wavelets
    # require near-regular centers; skip and flag rather than interpolate them.
    regular = bool(np.max(np.abs(np.diff(t)-dt)) <= 0.05*dt)
    z = (y-base)/noise
    sigma = z >= cfg.sigma_seed
    wave = np.zeros(len(t))
    dom = np.full(len(t), np.nan)
    flags = []
    if regular:
        scales = np.array([x/dt for x in cfg.wavelet_scales_s if 2 <= x/dt <= len(t)/8])
        if len(scales):
            coeff, _ = pywt.cwt(z, scales, "mexh", sampling_period=dt)
            for j, scale in enumerate(scales):
                # Conservative support mask; no padding-generated edge evidence.
                edge = min(len(t)//2, int(np.ceil(4*scale)))
                valid = np.arange(len(t)) >= edge
                valid &= np.arange(len(t)) < len(t)-edge
                norm = robust_noise(coeff[j, valid]) if valid.any() else 1
                score = coeff[j]/norm
                better = valid & (score > wave)
                wave[better] = score[better]
                dom[better] = scale*dt
        else:
            flags.append("wavelet_no_supported_scales")
    else:
        flags.append("wavelet_skipped_irregular_sampling")

    # Bound O(N^2) Bayesian Blocks cost through contiguous inverse-variance bins.
    chunks = np.array_split(np.arange(len(t)), min(len(t), cfg.bb_max_points))
    bt = np.array([np.mean(t[c]) for c in chunks])
    bw = [1/noise[c]**2 for c in chunks]
    bz = np.array([np.average(y[c]-base[c], weights=w) for c, w in zip(chunks, bw)])
    be = np.array([1/np.sqrt(w.sum()) for w in bw])
    edges = bayesian_blocks(bt, x=bz, sigma=be, fitness="measures", p0=cfg.bb_p0)
    bb = np.zeros(len(t), bool)
    for j in range(len(edges)-1) if len(edges) > 2 else []:
        inside = (t >= edges[j]) & ((t <= edges[j+1]) if j == len(edges)-2 else (t < edges[j+1]))
        if inside.any():
            w = 1/noise[inside]**2
            score = np.sum((y[inside]-base[inside])*w)/np.sqrt(w.sum())
            if score >= cfg.bb_sigma and np.mean(z[inside]) > 0.5:
                bb[inside] = True
    if len(chunks) < len(t):
        flags.append("bayesian_blocks_coarsened")
    wave_seed = (wave >= cfg.wavelet_sigma) & (z >= cfg.grow_sigma)
    seed = sigma | wave_seed | bb
    # One subthreshold bin may bridge a candidate, never an invalid/gap bin.
    # A significant BB interval is ONE continuous proposal support, not a
    # license to seed every small pointwise fluctuation inside it.
    active = (z > cfg.grow_sigma) | bb
    bridge = active.copy()
    bridge[1:-1] |= active[:-2] & active[2:]
    regions, n = label(bridge)
    out = []
    for k in range(1, n+1):
        ids = np.flatnonzero(regions == k)
        if not seed[ids].any():
            continue
        lo, hi = int(ids[0]), int(ids[-1])
        peak = lo+int(np.argmax(z[lo:hi+1]))
        out.append({"lo": lo, "hi": hi, "peak": peak,
                    "sigma_evidence": bool(sigma[ids].any()),
                    "wavelet_evidence": bool(wave_seed[ids].any()),
                    "bb_evidence": bool(bb[ids].any())})
    return out, {"z": z, "wavelet": wave, "dominant_scale": dom, "flags": flags}
