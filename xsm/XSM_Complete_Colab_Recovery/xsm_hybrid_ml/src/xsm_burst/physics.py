"""Causal FRED model, bin integration and likelihood-aware profile fitting.

s(t)=A*(1-exp(-x/tau_r))*exp(-x/tau_d), x=t-t0>=0; zero otherwise.
A is NOT net peak. tau_r is NOT the observed rise duration.
"""
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.special import xlogy


def fred(t, amplitude, onset, tau_r, tau_d):
    if amplitude < 0 or tau_r <= 0 or tau_d <= 0:
        raise ValueError("Invalid FRED amplitude/timescales")
    x = np.maximum(np.asarray(t, dtype=float)-onset, 0.0)
    return amplitude * (-np.expm1(-x/tau_r)) * np.exp(-x/tau_d)


def fred_bin_mean(lo, hi, amplitude, onset, tau_r, tau_d):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.any(hi <= lo) or amplitude < 0 or tau_r <= 0 or tau_d <= 0:
        raise ValueError("Invalid bins or FRED parameters")
    a, b = np.maximum(lo-onset, 0), np.maximum(hi-onset, 0)
    effective = tau_r*tau_d/(tau_r+tau_d)
    def integral(tau):
        return tau*np.exp(-a/tau)*(-np.expm1(-(b-a)/tau))
    # Small negative cancellation at onset is clipped at physical zero.
    return amplitude*np.maximum(integral(tau_d)-integral(effective), 0)/(hi-lo)


def fred_parameters(amplitude, onset, tau_r, tau_d, fraction=0.05):
    if not 0 < fraction < 1 or amplitude <= 0:
        raise ValueError("Positive amplitude and fractional boundary required")
    xp = tau_r*np.log1p(tau_d/tau_r)
    net_peak = float(fred(xp, amplitude, 0, tau_r, tau_d))
    def crossings(q):
        f = lambda x: float(fred(x, amplitude, 0, tau_r, tau_d))-q*net_peak
        left = brentq(f, 0, xp, xtol=1e-10)
        upper = xp+max(tau_r, tau_d)*100
        right = brentq(f, xp, upper, xtol=1e-10)
        return left, right
    a, b = crossings(fraction)
    h1, h2 = crossings(0.5)
    return {
        "onset_s": float(onset), "peak_s": float(onset+xp),
        "start_5pct_s": float(onset+a), "end_5pct_s": float(onset+b),
        "rise_duration_s": float(xp-a), "decay_duration_s": float(b-xp),
        "duration_s": float(b-a), "fwhm_s": float(h2-h1),
        "net_peak": net_peak, "amplitude_parameter": float(amplitude),
        "rise_constant_s": float(tau_r), "decay_constant_s": float(tau_d),
        "fluence_full_model": float(amplitude*tau_d*tau_d/(tau_r+tau_d)),
    }


def poisson_deviance_residual(observed, expected):
    expected = np.maximum(np.asarray(expected, float), np.finfo(float).tiny)
    observed = np.asarray(observed, float)
    # xlogy(0, ...) is exactly zero, including zero observed counts.
    dev = 2*(expected-observed+xlogy(observed, observed/expected))
    return np.sign(expected-observed)*np.sqrt(np.maximum(dev, 0))


def fit_fred(lo, hi, y, error, exposure, counts=None, bootstrap=0, seed=42, peak_window=None):
    lo, hi, y, error, exposure = map(lambda x: np.asarray(x, float), (lo, hi, y, error, exposure))
    if len(y) < 12:
        return {"success": False, "flags": ["insufficient_fit_bins"]}
    if not all(a.shape == y.shape for a in (lo, hi, error, exposure)):
        raise ValueError("Fit arrays must have identical shape")
    if not all(np.isfinite(a).all() for a in (lo, hi, y, error, exposure)) or np.any(error <= 0):
        raise ValueError("Fit arrays must be finite with positive errors")
    if np.any(hi <= lo) or np.any(exposure <= 0):
        raise ValueError("Invalid fit bin or exposure")
    if counts is not None:
        counts = np.asarray(counts, float)
        if counts.shape != y.shape or np.any(counts < 0) or not np.isfinite(counts).all():
            raise ValueError("Invalid raw counts")
    tref = float(lo.min())
    l, h = lo-tref, hi-tref
    mid = (l+h)/2
    dt = float(np.median(h-l))
    span = float(h.max())
    scale = max(float(np.ptp(y)), float(np.median(error)), 1e-20)
    ys, es = y/scale, error/scale
    edge = max(3, len(y)//8)
    b0 = float(np.median(np.r_[ys[:edge], ys[-edge:]]))
    amp0 = max(float(ys.max()-b0)*2, float(np.median(es)))
    peak = float(mid[np.argmax(ys)])
    floor_b = 1e-12 if counts is not None else float(ys.min()-3*np.ptp(ys)-10*np.median(es))
    lower = np.array([np.log(amp0*1e-4), -span, np.log(dt/4), np.log(dt/4), floor_b])
    upper = np.array([np.log(amp0*1e4), h.max(), np.log(span*2), np.log(span*10), float(ys.max()+10*np.median(es)+1)])

    # When fitting a detected event, bind the physical peak to its interval.
    # q[1] is then peak time rather than onset; standalone fits remain unchanged.
    if peak_window is not None:
        a, b = map(float, peak_window)
        if not np.isfinite([a, b]).all() or a >= b or a < lo.min() or b > hi.max():
            raise ValueError("Peak window must lie within the fitting interval")
        lower[1], upper[1] = a-tref, b-tref

    def onset_of(q):
        if peak_window is None:
            return q[1]
        return q[1]-np.exp(q[2])*np.log1p(np.exp(q[3]-q[2]))

    def prediction(q):
        return q[4]+fred_bin_mean(l, h, np.exp(q[0]), onset_of(q), np.exp(q[2]), np.exp(q[3]))

    def residual(q, target_y=ys, target_counts=counts):
        model = prediction(q)
        if target_counts is not None:
            return poisson_deviance_residual(target_counts, model*scale*exposure)
        return (model-target_y)/es

    solutions = []
    for tr in (max(dt, span/40), max(dt, span/10)):
        for td in (max(dt, span/8), max(dt, span/2)):
            onset = peak-tr*np.log1p(td/tr)
            q0 = np.array([np.log(amp0), peak if peak_window is not None else onset, np.log(tr), np.log(td), b0])
            q0 = np.clip(q0, lower+1e-8, upper-1e-8)
            r = least_squares(residual, q0, bounds=(lower, upper), x_scale="jac", max_nfev=600)
            if r.success and np.isfinite(r.fun).all():
                solutions.append(r)
    if not solutions:
        return {"success": False, "flags": ["optimizer_failed"]}
    best = min(solutions, key=lambda r: np.sum(r.fun**2))
    q = best.x
    pars = fred_parameters(np.exp(q[0])*scale, onset_of(q)+tref, np.exp(q[2]), np.exp(q[3]))
    fitted = prediction(q)*scale
    raw_residual = y-fitted
    rss = float(np.sum(raw_residual**2))
    tss = float(np.sum((y-y.mean())**2))
    stat = float(np.sum(best.fun**2))
    n, k = len(y), len(q)
    flags = []
    if np.any(np.minimum(q-lower, upper-q) < 1e-4*(upper-lower)):
        flags.append("parameter_near_bound")
    left = pars["start_5pct_s"] < lo.min()
    right = pars["end_5pct_s"] > hi.max()
    if left: flags.append("left_censored")
    if right: flags.append("right_censored")
    std = float(np.std(best.fun))
    autocorr = float(np.corrcoef(best.fun[:-1], best.fun[1:])[0, 1]) if std > 1e-10 else 0.0
    if abs(autocorr) > 0.3: flags.append("correlated_residuals")
    if stat/(n-k) > 3: flags.append("poor_likelihood_fit")
    if min(pars["rise_constant_s"], pars["decay_constant_s"]) < dt:
        flags.append("timescale_unresolved")
    rank = np.linalg.matrix_rank(best.jac)
    if rank < k or np.linalg.cond(best.jac) > 1e8:
        flags.append("ill_conditioned_fit")
    # Compare against a sloping background using the SAME bins and likelihood.
    x = (mid-mid.min())/max(float(np.ptp(mid)), dt)
    design = np.column_stack([1-x, x])
    endpoint0 = np.linalg.lstsq(design/es[:, None], ys/es, rcond=None)[0]
    if counts is None:
        null_prediction = design @ endpoint0
        null_stat = float(np.sum(((null_prediction-ys)/es)**2))
    else:
        null = least_squares(
            lambda v: poisson_deviance_residual(counts, (design@v)*scale*exposure),
            np.maximum(endpoint0, 1e-10), bounds=(1e-12, np.inf), max_nfev=600)
        null_stat = float(np.sum(null.fun**2)) if null.success else np.nan
    delta_bic = null_stat+2*np.log(n) - (stat+k*np.log(n))
    if not np.isfinite(delta_bic) or delta_bic <= 0:
        flags.append("no_improvement_over_linear_background")
    pars["delta_bic_vs_linear_background"] = float(delta_bic)
    pars["parameters_reliable"] = not flags
    pars.update({"success": True, "flags": flags, "baseline": float(q[4]*scale),
                 "total_peak": float(q[4]*scale+pars["net_peak"]),
                 "r2": 1-rss/tss if tss > 1e-20 else np.nan,
                 "rmse": float(np.sqrt(rss/n)), "fit_statistic": stat,
                 "reduced_statistic": stat/(n-k),
                 "statistic_kind": "poisson_deviance" if counts is not None else "chi_square",
                 "aic_relative": stat+2*k, "bic_relative": stat+k*np.log(n),
                 "aicc_relative": stat+2*k+2*k*(k+1)/(n-k-1),
                 "residual_autocorrelation": autocorr,
                 "predicted": fitted.tolist(), "residual": raw_residual.tolist(),
                 "fit_time_s": ((lo+hi)/2).tolist(), "bootstrap_intervals_95": {}})
    # AIC/BIC omit data-only likelihood constants; compare same bins/likelihood only.
    if bootstrap:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            yc = rng.poisson(np.maximum(fitted*exposure, 0)) if counts is not None else None
            yr = rng.normal(fitted/scale, es) if counts is None else yc/exposure/scale
            rr = least_squares(lambda x: residual(x, yr, yc), q, bounds=(lower, upper),
                               max_nfev=400, x_scale="jac")
            if rr.success:
                a = rr.x
                draws.append(fred_parameters(np.exp(a[0])*scale, onset_of(a)+tref, np.exp(a[2]), np.exp(a[3])))
        pars["bootstrap_successes"] = len(draws)
        if len(draws) >= max(20, int(0.8*bootstrap)):
            for key in draws[0]:
                pars["bootstrap_intervals_95"][key] = np.quantile([d[key] for d in draws], [0.025, 0.975]).tolist()
        else:
            pars["flags"].append("insufficient_bootstrap_refits")
        pars["flags"].append("bootstrap_assumes_model_and_independent_noise")
    return pars
