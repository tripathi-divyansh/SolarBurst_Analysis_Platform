"""
Tests for Random Forest and XGBoost classifier wrappers and persistence.
"""

import os
import numpy as np
import pytest
from solarburst.ml.features import FEATURE_NAMES
from solarburst.ml.models import BurstClassifier, train_candidate_models
from solarburst.ml.persistence import save_classifier, load_classifier, verify_classifier_agreement


def test_train_rf_and_xgb():
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_NAMES)
    
    # Generate synthetic feature sets
    X_train = rng.normal(0, 1, size=(60, n_features))
    y_train = np.array([1]*15 + [0]*45)

    X_val = rng.normal(0, 1, size=(30, n_features))
    y_val = np.array([1]*8 + [0]*22)

    res = train_candidate_models(X_train, y_train, X_val, y_val)

    rf: BurstClassifier = res["rf_wrapper"]
    xgb: BurstClassifier = res["xgb_wrapper"]

    # Verify score outputs
    rf_scores = rf.predict_scores(X_val)
    xgb_scores = xgb.predict_scores(X_val)

    assert len(rf_scores) == len(X_val)
    assert len(xgb_scores) == len(X_val)
    assert np.all(rf_scores >= 0.0) and np.all(rf_scores <= 1.0)
    assert np.all(xgb_scores >= 0.0) and np.all(xgb_scores <= 1.0)


def test_model_persistence_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_NAMES)
    X = rng.normal(0, 1, size=(40, n_features))
    y = np.array([1]*10 + [0]*30)

    clf = BurstClassifier(model_type="random_forest")
    clf.fit(X, y)

    saved = save_classifier(clf, str(tmp_path), "test_rf")
    assert os.path.exists(saved["model_path"])

    loaded = load_classifier(saved["model_path"])
    assert verify_classifier_agreement(loaded, X)

    # Predictions must be identical
    orig_p = clf.predict_scores(X)
    loaded_p = loaded.predict_scores(X)
    np.testing.assert_allclose(orig_p, loaded_p, atol=1e-5)
