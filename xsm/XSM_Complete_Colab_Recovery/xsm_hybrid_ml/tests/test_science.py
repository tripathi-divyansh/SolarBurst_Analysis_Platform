import json
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from xsm_burst.schema import LightCurve, PipelineConfig
from xsm_burst.physics import fred, fred_bin_mean, fred_parameters, fit_fred, poisson_deviance_residual
from xsm_burst.pipeline import analyze
from xsm_burst.synthetic import synthetic_observation
from xsm_burst.io import load_lightcurve
from xsm_burst.evaluation import match_events


def basic(n=100):
    return LightCurve(np.arange(n)+0.5, np.full(n, 20.), np.ones(n), np.ones(n),
                      np.ones(n), np.zeros(n, int),
                      {"quantity": "count_rate", "unit": "count/s", "epoch": "relative", "time_scale": "relative"})


def test_fred_causal_and_amplitude_not_peak():
    t = np.array([-1e9, -1, 0, 1, 100.])
    y = fred(t, 12, 0, 4, 20)
    assert np.all(y[:3] == 0) and np.isfinite(y).all()
    p = fred_parameters(12, 0, 4, 20)
    assert 0 < p["net_peak"] < 12
    assert p["start_5pct_s"] < p["peak_s"] < p["end_5pct_s"]
    assert p["rise_constant_s"] != p["rise_duration_s"]


@pytest.mark.parametrize("lo,hi", [(-3,-1), (-2,2), (1,2), (20,70), (1000,1001)])
def test_bin_integral_matches_quadrature(lo, hi):
    actual = fred_bin_mean([lo], [hi], 23, 0, 5, 30)[0]
    exact = quad(lambda t: float(fred(t, 23, 0, 5, 30)), lo, hi, points=[0] if lo < 0 < hi else None)[0]/(hi-lo)
    assert actual == pytest.approx(exact, abs=1e-9, rel=1e-8)


def test_fluence_conservation_rebin():
    edges = np.arange(-10., 501.)
    fine = fred_bin_mean(edges[:-1], edges[1:], 12, 0, 4, 20)
    coarse = fred_bin_mean(edges[:-1:10], edges[10::10], 12, 0, 4, 20)
    assert np.sum(fine) == pytest.approx(np.sum(coarse)*10, rel=1e-10)
    assert fine.sum() == pytest.approx(fred_parameters(12, 0, 4, 20)["fluence_full_model"], rel=1e-8)


def test_fit_recovers_noiseless_fred():
    t = np.arange(220.)+0.5
    y = 10+fred_bin_mean(t-0.5, t+0.5, 70, 40, 5, 25)
    fit = fit_fred(t-0.5, t+0.5, y, np.ones(len(t)), np.ones(len(t)))
    assert fit["success"]
    assert fit["onset_s"] == pytest.approx(40, abs=0.01)
    assert fit["rise_constant_s"] == pytest.approx(5, rel=0.01)
    assert fit["decay_constant_s"] == pytest.approx(25, rel=0.01)
    assert fit["r2"] > 0.9999


def test_poisson_zero_counts_and_fit():
    r = poisson_deviance_residual([0, 1, 5], [0.2, 1, 4])
    assert np.isfinite(r).all() and r[1] == 0
    rng = np.random.default_rng(17)
    t = np.arange(220.)+0.5
    rate = 2+fred_bin_mean(t-0.5, t+0.5, 50, 40, 5, 25)
    counts = rng.poisson(rate)
    fit = fit_fred(t-0.5, t+0.5, counts, np.sqrt(np.maximum(counts, 1)), np.ones(len(t)), counts=counts)
    assert fit["success"] and fit["statistic_kind"] == "poisson_deviance"
    assert fit["decay_constant_s"] == pytest.approx(25, rel=0.3)


def test_censored_tail_is_flagged():
    t = np.arange(80.)+0.5
    y = 10+fred_bin_mean(t-0.5, t+0.5, 70, 30, 5, 60)
    fit = fit_fred(t-0.5, t+0.5, y, np.ones(len(t)), np.ones(len(t)))
    assert "right_censored" in fit["flags"]


def test_missing_errors_rejected_for_corrected_rate():
    lc = basic(); lc.error = None
    with pytest.raises(ValueError, match="measurement errors"): lc.prepared()


def test_overlapping_bins_rejected():
    lc = basic(); lc.bin_width[:] = 2
    with pytest.raises(ValueError, match="Overlapping"): lc.prepared()


def test_raw_noninteger_counts_rejected():
    lc = basic(); lc.metadata["quantity"] = "counts"; lc.value[5] = 3.5
    with pytest.raises(ValueError, match="integers"): lc.prepared()


def test_sort_and_duplicate_policy():
    lc = basic()
    for name in ("time", "value", "error", "bin_width", "exposure", "quality"):
        a = getattr(lc, name)
        setattr(lc, name, np.r_[a[::-1], a[0]])
    with pytest.raises(ValueError, match="Duplicate"): lc.prepared()
    p = lc.prepared("identical")
    assert len(p.time) == 100 and np.all(np.diff(p.time) > 0)
    lc.value[-1] = 30
    with pytest.raises(ValueError, match="Conflicting"): lc.prepared("identical")


def test_invalid_samples_and_gaps_split():
    lc = basic()
    lc.value[20] = np.nan; lc.quality[40] = 4; lc.time[60:] += 20
    p = lc.prepared()
    segs = p.segments()
    assert len(segs) == 4
    assert all(20 not in seg and 40 not in seg for seg in segs)
    assert not any(59 in seg and 60 in seg for seg in segs)


def test_empty_quiet_analysis():
    result = analyze(basic())
    assert result["catalog"].empty
    assert result["manifest"]["processed_exposure_s"] == 100


def test_synthetic_candidates_recover_truth_and_do_not_cross_gap():
    lc, truth = synthetic_observation(42)
    lc.quality[600:620] = 1
    out = analyze(lc, group_id="g")
    truth["group_id"] = "g"
    match = match_events(out["catalog"], truth, 35)
    assert match["recall"] == 1
    cat = out["catalog"]
    assert not ((cat.observed_start_s < 600) & (cat.observed_end_s > 620)).any()
    assert out["manifest"]["processed_exposure_s"] == 980
    assert all(fit["success"] or "optimizer_failed" in fit["flags"] or "insufficient_fit_bins" in fit["flags"] or "insufficient_event_bins" in fit["flags"] for fit in out["fits"].values())


@pytest.mark.parametrize("ext", ["csv", "tsv", "txt", "xlsx", "fits"])
def test_format_equivalence(tmp_path, ext):
    frame = pd.DataFrame({"TIME": np.arange(30.)+0.5, "RATE": np.ones(30)*5, "ERROR": np.ones(30)})
    path = tmp_path/f"sample.{ext}"
    if ext == "fits":
        from astropy.table import Table
        Table.from_pandas(frame).write(path)
    elif ext == "xlsx": frame.to_excel(path, index=False)
    else: frame.to_csv(path, index=False, sep="\t" if ext == "tsv" else " " if ext == "txt" else ",")
    mapping = {"time": "TIME", "value": "RATE", "error": "ERROR", "bin_width_s": 1,
               "exposure_s": 1, "quantity": "count_rate", "unit": "count/s", "epoch": "relative",
               "time_scale": "relative", "time_unit_s": 1, "time_position": "center"}
    result = load_lightcurve(path, mapping).prepared()
    np.testing.assert_array_equal(result.time, frame.TIME)
    np.testing.assert_array_equal(result.value, frame.RATE)


def test_missing_epoch_mapping_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing explicit"): load_lightcurve(tmp_path/"missing.csv", {})


def test_placeholder_epoch_rejected():
    lc = basic(); lc.metadata.update(epoch="REPLACE", time_scale="utc")
    with pytest.raises(ValueError, match="verified date"): lc.prepared()


def test_fit_failed_small_window():
    out = fit_fred(np.arange(5.), np.arange(5.)+1, np.ones(5), np.ones(5), np.ones(5))
    assert not out["success"]


def test_duplicate_event_predictions_are_false_positives():
    event = {"group_id": "g", "observed_start_s": 1, "observed_peak_s": 5, "observed_end_s": 10}
    out = match_events(pd.DataFrame([event, event]), pd.DataFrame([event]))
    assert (out["tp"], out["fp"], out["fn"]) == (1, 1, 0)


def test_count_output_signal_unit_is_rate():
    lc = basic(200)
    lc.metadata.update(quantity="counts", unit="count")
    lc.error = None
    lc.value[70:80] += 50
    result = analyze(lc)
    assert result["manifest"]["signal_unit"] == "count / s"
    assert not result["catalog"].empty
    assert (result["catalog"].unit == "count / s").all()


def test_parametric_bootstrap_produces_intervals():
    t = np.arange(180.)+0.5
    y = 10+fred_bin_mean(t-0.5, t+0.5, 60, 30, 4, 20)
    fit = fit_fred(t-0.5, t+0.5, y, np.ones(len(t)), np.ones(len(t)), bootstrap=20)
    assert fit["bootstrap_successes"] == 20
    interval = fit["bootstrap_intervals_95"]["decay_constant_s"]
    assert interval[0] < interval[1]
    assert "bootstrap_assumes_model_and_independent_noise" in fit["flags"]


def test_irregular_wavelet_is_skipped():
    from xsm_burst.detection import proposals
    t = np.cumsum(np.tile([0.7, 1.3], 40))
    y = np.ones(len(t))*5
    candidates, evidence = proposals(t, y, np.ones(len(t)), y.copy(), np.ones(len(t)), PipelineConfig())
    assert "wavelet_skipped_irregular_sampling" in evidence["flags"]


def test_fits_gti_excludes_partial_bins(tmp_path):
    from astropy.io import fits
    cols = [fits.Column(name="TIME", format="D", array=np.arange(30.)+0.5),
            fits.Column(name="RATE", format="D", array=np.ones(30)*5),
            fits.Column(name="ERROR", format="D", array=np.ones(30))]
    gti = fits.BinTableHDU.from_columns([fits.Column(name="START", format="D", array=[3.2]),
                                       fits.Column(name="STOP", format="D", array=[20.])], name="GTI")
    path = tmp_path/"gti.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols), gti]).writeto(path)
    mapping = {"time":"TIME", "value":"RATE", "error":"ERROR", "bin_width_s":1,
               "exposure_s":1, "quantity":"count_rate", "unit":"count/s", "epoch":"relative",
               "time_scale":"relative", "time_unit_s":1, "time_position":"center"}
    lc = load_lightcurve(path, mapping).prepared()
    assert not lc.valid[3] and lc.valid[4] and lc.valid[19] and not lc.valid[20]
