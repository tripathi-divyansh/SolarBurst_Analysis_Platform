import numpy as np
import pandas as pd
import pytest
from xsm_burst.features import FEATURES, FEATURE_VERSION
from xsm_burst.models import FeatureTransform, train_models, BurstModels, validate_training_table, SPLITS
from xsm_burst.evaluation import select_threshold, far_stats


def artificial_feature_fixture():
    """Software fixture, not a physical dataset or evidence of XSM performance."""
    rng = np.random.default_rng(19)
    rows, exposures = [], []
    for split in SPLITS:
        for group in range(4):
            name = f"{split}_{group}"
            exposures.append({"group_id": name, "split": split, "processed_exposure_s": 86400., "fully_reviewed": True})
            for i in range(30):
                label = i % 2
                features = {key: float(rng.normal()) for key in FEATURES}
                features["peak_snr"] = label*5+rng.normal(0, 0.3)
                features["fit_r2"] = np.nan  # All-missing column must preserve feature shape.
                rows.append({**features, "candidate_id": f"{name}_{i}", "group_id": name,
                             "source_hash": name, "pipeline_hash": "fixture_config", "feature_version": FEATURE_VERSION,
                             "quantity": "count_rate", "unit": "count/s", "label": label, "split": split})
    return pd.DataFrame(rows), pd.DataFrame(exposures)


def test_feature_order_and_missingness():
    frame, _ = artificial_feature_fixture()
    tr = FeatureTransform(FEATURES)
    a = tr.fit_transform(frame)
    b = tr.transform(frame[frame.columns[::-1]])
    assert a.shape[1] == len(FEATURES)*2
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="Missing model features"):
        tr.transform(frame.drop(columns=[FEATURES[0]]))


@pytest.mark.parametrize("column", ["group_id", "source_hash"])
def test_cross_partition_leakage_rejected(column):
    frame, _ = artificial_feature_fixture()
    frame.loc[frame.split == "test", column] = frame.loc[0, column]
    with pytest.raises(ValueError, match="Leakage"): validate_training_table(frame)


def test_mixed_feature_configs_rejected():
    frame, _ = artificial_feature_fixture(); frame.loc[0, "pipeline_hash"] = "changed"
    with pytest.raises(ValueError, match="Mixed pipeline_hash"): validate_training_table(frame)


def test_threshold_zero_false_alarm_operating_point():
    t, stats = select_threshold([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9], 86400, 0)
    assert t == 0.7 and stats["false_alarms_per_valid_day"] == 0
    assert stats["far_poisson_upper_95"] > 0


def test_invalid_exposure_rejected():
    with pytest.raises(ValueError, match="exposure"): far_stats(1, 0)


def test_train_calibrate_save_reload(tmp_path):
    frame, exposure = artificial_feature_fixture()
    bundle, report = train_models(frame, exposure, n_estimators=60, domain_status="SOFTWARE_FIXTURE_ONLY")
    assert np.isfinite(report["brier"])
    test = frame[frame.split == "test"]
    before = bundle.score(test)
    assert before.burst_probability.between(0, 1).all()
    path = tmp_path/"model.joblib"
    bundle.save(path)
    with pytest.raises(ValueError, match="trusted"): BurstModels.load(path)
    loaded = BurstModels.load(path, trusted=True)
    after = loaded.score(test)
    np.testing.assert_allclose(before.burst_probability, after.burst_probability, rtol=0, atol=0)
    wrong = test.copy(); wrong["unit"] = "W/m2"
    with pytest.raises(ValueError, match="unit"): loaded.score(wrong)


def test_unreviewed_exposure_rejected():
    frame, exposure = artificial_feature_fixture(); exposure.loc[0, "fully_reviewed"] = False
    with pytest.raises(ValueError, match="fully reviewed"): train_models(frame, exposure, n_estimators=10)
