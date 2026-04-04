"""
tests/test_explainability.py
-----------------------------
Smoke tests for the SHAP explainability layer.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.explainability.shap_explainer import explain_tree  # noqa: E402, F401
from src.models.random_forest import train_random_forest  # noqa: E402, F401

N_SAMPLES = 200
N_FEATURES = 15
N_CLASSES = 4
SEED = 7


@pytest.fixture(scope="module")
def fitted_rf():
    rng = np.random.default_rng(SEED)
    X = rng.random((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, N_SAMPLES)
    clf = train_random_forest(X, y, n_estimators=10, save_path=None)
    return clf, X


class TestSHAPTreeExplainer:
    def test_returns_explanation_object(self, fitted_rf):
        import shap
        clf, X = fitted_rf
        expl = explain_tree(clf, X, max_samples=50)
        assert isinstance(expl, shap.Explanation)

    def test_shap_values_shape(self, fitted_rf):
        clf, X = fitted_rf
        expl = explain_tree(clf, X, max_samples=50)
        # Multi-class RF: shape is (samples, features, n_classes)
        # or (samples, features) depending on SHAP version
        assert expl.values is not None
        assert expl.values.shape[0] == 50
        assert expl.values.shape[1] == N_FEATURES

    def test_feature_names_preserved(self, fitted_rf):
        clf, X = fitted_rf
        names = [f"feature_{i}" for i in range(N_FEATURES)]
        expl = explain_tree(clf, X, feature_names=names, max_samples=20)
        assert expl.feature_names == names

    def test_base_values_present(self, fitted_rf):
        clf, X = fitted_rf
        expl = explain_tree(clf, X, max_samples=30)
        assert expl.base_values is not None
