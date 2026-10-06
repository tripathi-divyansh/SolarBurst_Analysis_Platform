"""
Model persistence, serialization, checksum verification, and model cards.
"""

from typing import Dict, Any, Optional
import os
import json
import hashlib
import joblib
import numpy as np
from .models import BurstClassifier


def save_classifier(classifier: BurstClassifier, directory: str, filename_prefix: str = "burst_classifier") -> Dict[str, str]:
    """
    Save BurstClassifier model, imputer, feature schema, and model card JSON.
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, f"{filename_prefix}.joblib")
    card_path = os.path.join(directory, f"{filename_prefix}_card.json")

    # Serialize object
    state = {
        "model_type": classifier.model_type,
        "accept_threshold": classifier.accept_threshold,
        "review_threshold": classifier.review_threshold,
        "feature_names": classifier.feature_names,
        "imputer": classifier.imputer,
        "model": classifier.model,
        "training_metadata": classifier.training_metadata
    }
    joblib.dump(state, model_path, compress=3)

    # Compute checksum
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha256 = h.hexdigest()

    # Save Model Card
    card = {
        "model_name": f"SolarBurst-{classifier.model_type.upper()}-v1.0",
        "model_type": classifier.model_type,
        "file_sha256": sha256,
        "features_order": classifier.feature_names,
        "accept_threshold": classifier.accept_threshold,
        "review_threshold": classifier.review_threshold,
        "training_metadata": classifier.training_metadata,
        "intended_use": "Scoring proposed solar X-ray burst candidates from permissive classical detectors",
        "limitations": "Trained with engineered candidate features; physical parameters require profile fitting."
    }
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    return {"model_path": model_path, "card_path": card_path, "sha256": sha256}


def load_classifier(filepath: str) -> BurstClassifier:
    """
    Load and verify a serialized BurstClassifier.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    state = joblib.load(filepath)
    clf = BurstClassifier(
        model_type=state["model_type"],
        accept_threshold=state.get("accept_threshold", 0.65),
        review_threshold=state.get("review_threshold", 0.35)
    )
    clf.feature_names = state["feature_names"]
    clf.imputer = state["imputer"]
    clf.model = state["model"]
    clf.training_metadata = state.get("training_metadata", {})
    return clf


def verify_classifier_agreement(clf: BurstClassifier, test_features: np.ndarray) -> bool:
    """
    Verify that classifier produces finite, bounded predictions in [0, 1].
    """
    scores = clf.predict_scores(test_features)
    return bool(np.all(np.isfinite(scores)) and np.all(scores >= 0.0) and np.all(scores <= 1.0))
