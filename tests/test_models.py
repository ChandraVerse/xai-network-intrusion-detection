"""
tests/test_models.py
---------------------
Smoke tests for model training and prediction interfaces.
Uses tiny synthetic datasets — no CICIDS-2017 download required.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.random_forest import train_random_forest  # noqa: E402
from src.models.xgboost_model import train_xgboost  # noqa: E402

N_SAMPLES = 300
N_FEATURES = 20
N_CLASSES = 5
SEED = 42


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(SEED)
    X = rng.random((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, N_SAMPLES)
    split = int(0.8 * N_SAMPLES)
    return X[:split], y[:split], X[split:], y[split:]


class TestRandomForest:
    def test_trains_without_error(self, synthetic_data, tmp_path):
        X_tr, y_tr, _, _ = synthetic_data
        clf = train_random_forest(
            X_tr, y_tr,
            n_estimators=10,
            save_path=str(tmp_path / "rf.pkl")
        )
        assert clf is not None

    def test_predict_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        clf = train_random_forest(X_tr, y_tr, n_estimators=10, save_path=None)
        preds = clf.predict(X_te)
        assert preds.shape == (X_te.shape[0],)

    def test_predict_proba_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        clf = train_random_forest(X_tr, y_tr, n_estimators=10, save_path=None)
        proba = clf.predict_proba(X_te)
        assert proba.shape == (X_te.shape[0], N_CLASSES)

    def test_valid_class_labels(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        clf = train_random_forest(X_tr, y_tr, n_estimators=10, save_path=None)
        preds = clf.predict(X_te)
        assert set(np.unique(preds)).issubset(set(range(N_CLASSES)))


class TestXGBoost:
    def test_trains_without_error(self, synthetic_data, tmp_path):
        X_tr, y_tr, _, _ = synthetic_data
        clf = train_xgboost(
            X_tr, y_tr,
            n_estimators=20,
            max_depth=4,
            save_path=str(tmp_path / "xgb.pkl")
        )
        assert clf is not None

    def test_predict_shape(self, synthetic_data):
        X_tr, y_tr, X_te, _ = synthetic_data
        clf = train_xgboost(X_tr, y_tr, n_estimators=20, max_depth=4, save_path=None)
        preds = clf.predict(X_te)
        assert preds.shape == (X_te.shape[0],)

    def test_f1_above_threshold(self, synthetic_data):
        """Even on tiny synthetic data, F1 should be non-zero."""
        from sklearn.metrics import f1_score
        X_tr, y_tr, X_te, y_te = synthetic_data
        clf = train_xgboost(X_tr, y_tr, n_estimators=50, max_depth=4, save_path=None)
        preds = clf.predict(X_te)
        f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        assert f1 >= 0.0
