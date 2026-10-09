from pathlib import Path
import hashlib
import numpy as np
import pytest
from xsm_burst.schema import PipelineConfig, LightCurve
from xsm_burst.detection import estimate_background, proposals
from xsm_burst.physics import fit_fred, fred_bin_mean, fred_parameters
from xsm_burst.pipeline import analyze
from xsm_burst.xsm import load_xsm_level2

DATA = Path(__file__).parents[1]/'data/july16'


def test_local_linear_background_preserves_slope_at_boundaries():
    t = np.arange(400.)+.5
    y = 100+0.2*t
    base, noise, poor = estimate_background(t, y, np.ones(len(t)), 301.)
    np.testing.assert_allclose(base, y, atol=1e-8)
    assert not poor
    assert not proposals(t, y, noise, base, noise, PipelineConfig())[0]


def test_bb_does_not_call_constant_residual_offset_an_event():
    t = np.arange(300.)+.5
    y = np.ones(len(t))*.8
    proposals_found, _ = proposals(t, y, np.ones(len(t)), np.zeros(len(t)), np.ones(len(t)), PipelineConfig())
    assert proposals_found == []


def test_bb_broad_weak_interval_not_fragmented_into_individual_spikes():
    t = np.arange(500.)+.5
    y = np.zeros(len(t)); y[200:350] = np.tile([2.,0.,0.],50)
    found, _ = proposals(t, y, np.ones(len(t)), np.zeros(len(t)), np.ones(len(t)), PipelineConfig())
    bb = [p for p in found if p['bb_evidence']]
    assert len(bb) == 1
    assert bb[0]['hi']-bb[0]['lo'] > 100


def test_peak_constraint_targets_detected_event():
    t = np.arange(220.)+.5
    y = 10+fred_bin_mean(t-.5,t+.5,70,40,5,25)
    peak=fred_parameters(70,40,5,25)['peak_s']
    fit=fit_fred(t-.5,t+.5,y,np.ones(len(t)),np.ones(len(t)),peak_window=(peak-1,peak+1))
    assert fit['success'] and peak-1 <= fit['peak_s'] <= peak+1
    assert fit['decay_constant_s'] == pytest.approx(25, rel=.01)
    assert fit['delta_bic_vs_linear_background'] > 0


def test_single_sample_has_no_resolved_physical_parameters():
    t=np.arange(400.)+.5
    y=np.full(len(t),100.); y[200]+=30
    lc=LightCurve(t,y,np.ones(len(t)),np.ones(len(t)),np.ones(len(t)),np.zeros(len(t),int),
                  {'quantity':'count_rate','unit':'count / s','epoch':'relative','time_scale':'relative'})
    out=analyze(lc)
    single=out['catalog'][out['catalog'].quality_flags.str.contains('single_bin_candidate')]
    assert len(single)==1
    fit=out['fits'][single.iloc[0].candidate_id]
    assert not fit['success'] and not fit['parameters_reliable']
    assert single.fit_rise_constant_s.isna().all()


def test_supplied_xsm_adapter_preserves_exposure_gaps_and_hash():
    path=DATA/'ch2_xsm_20260716_v1_level2.lc'
    lc=load_xsm_level2(path)
    assert len(lc.time)==6163
    assert len(lc.segments())==11
    assert lc.exposure[lc.valid].sum()==pytest.approx(6120.199950397015)
    assert lc.metadata['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert lc.metadata['invalid_rows']==1


def test_truncated_fits_fails_with_actionable_error(tmp_path):
    original=DATA/'ch2_xsm_20260716_v1_level2.lc'
    short=tmp_path/'short.lc';short.write_bytes(original.read_bytes()[:-5760])
    with pytest.warns(Warning):
        with pytest.raises(ValueError,match='Truncated FITS'):
            load_xsm_level2(short, DATA/'ch2_xsm_20260716_v1_level2.gti')
