"""
Random Forest baseline and XGBoost challenger candidate classification models.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from .features import FEATURE_NAMES


class BurstClassifier:
    """
    Wrapper for candidate classification models (Random Forest and XGBoost).
    Persists imputer, model, threshold operating points, and feature schema.
    """
    def __init__(
        self,
        model_type: str = "random_forest",
        accept_threshold: float = 0.65,
        review_threshold: float = 0.35
    ):
        self.model_type = model_type
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        self.imputer: Optional[SimpleImputer] = None
        self.model = None
        self.feature_names = list(FEATURE_NAMES)
        self.training_metadata: Dict[str, Any] = {}

    def prepare_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply imputer and concatenate missingness indicators."""
        X_arr = np.asarray(X, dtype=np.float64).copy()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_arr[~np.isfinite(X_arr)] = np.nan

        if fit:
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            imputed = self.imputer.fit_transform(X_arr)
        else:
            if self.imputer is None:
                raise ValueError("Imputer is not fitted yet.")
            imputed = self.imputer.transform(X_arr)

        indicators = np.isnan(X_arr).astype(np.float64)
        return np.column_stack([imputed, indicators])

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Random Forest or XGBoost on candidate feature matrix.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=int)

        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set requires both positive (1) and negative (0) candidate classes.")

        A = self.prepare_features(X_train, fit=True)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            B = self.prepare_features(X_val, fit=False)
            has_val = True
        else:
            B, y_val = None, None
            has_val = False

        ratio = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(A, y_train)
            val_scores = self.model.predict_proba(B)[:, 1] if has_val else None

        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_lambda=5.0,
                scale_pos_weight=ratio,
                tree_method="hist",
                eval_metric="aucpr",
                early_stopping_rounds=30 if has_val else None,
                random_state=42,
                n_jobs=-1
            )
            if has_val:
                self.model.fit(A, y_train, eval_set=[(B, y_val)], verbose=False)
                val_scores = self.model.predict_proba(B)[:, 1]
            else:
                self.model.fit(A, y_train, verbose=False)
                val_scores = None
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.training_metadata = {
            "model_type": self.model_type,
            "n_train": len(y_train),
            "n_val": len(y_val) if y_val is not None else 0,
            "pos_ratio": float(np.mean(y_train)),
            "features_count": len(self.feature_names),
        }

        return {
            "model_type": self.model_type,
            "val_scores": val_scores
        }

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return continuous burst confidence score in [0.0, 1.0]."""
        if self.model is None or self.imputer is None:
            # Fallback heuristic score based on peak SNR and excess
            snr_col = self.feature_names.index("peak_snr")
            snr_vals = X[:, snr_col] if X.ndim == 2 else np.array([X[snr_col]])
            # Sigmoid on SNR
            return 1.0 / (1.0 + np.exp(-0.5 * (snr_vals - 4.0)))

        A = self.prepare_features(X, fit=False)
        return self.model.predict_proba(A)[:, 1]

    def predict_decisions(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Return (scores, decisions) where decision is 'accepted', 'review', or 'rejected'.
        """
        scores = self.predict_scores(X)
        decisions = []
        for s in scores:
            if s >= self.accept_threshold:
                decisions.append("accepted")
            elif s >= self.review_threshold:
                decisions.append("review")
            else:
                decisions.append("rejected")
        return scores, decisions


def train_candidate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    Train both Random Forest baseline and XGBoost challenger for model comparison.
    Satisfies specification section 2.4.
    """
    rf_wrapper = BurstClassifier(model_type="random_forest")
    rf_res = rf_wrapper.fit(X_train, y_train, X_val, y_val)

    xgb_wrapper = BurstClassifier(model_type="xgboost")
    xgb_res = xgb_wrapper.fit(X_train, y_train, X_val, y_val)

    return {
        "rf_wrapper": rf_wrapper,
        "xgb_wrapper": xgb_wrapper,
        "rf_val_scores": rf_res["val_scores"],
        "xgb_val_scores": xgb_res["val_scores"]
    }
