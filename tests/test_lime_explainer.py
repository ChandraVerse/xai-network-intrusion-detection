"""
tests/test_lime_explainer.py
-----------------------------
Smoke + unit tests for the LIME explainability module.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.explainability.lime_explainer import LIMEExplainer, make_keras_predict_fn  # noqa: E402

# Uses conftest fixtures: X_small, y_small, fitted_rf_small (15 features, 4 classes)


class TestLIMEExplainerInit:
    def test_initialises_without_names(self, X_small):
        exp = LIMEExplainer(training_data=X_small)
        assert exp.explainer is not None

    def test_initialises_with_names(self, X_small):
        names = [f"feat_{i}" for i in range(X_small.shape[1])]
        exp = LIMEExplainer(training_data=X_small, feature_names=names)
        assert exp.feature_names == names

    def test_class_names_stored(self, X_small):
        labels = ["BENIGN", "DDoS", "PortScan", "Bot"]
        exp = LIMEExplainer(training_data=X_small, class_names=labels)
        assert exp.class_names == labels


class TestExplainInstance:
    def test_returns_explanation(self, X_small, fitted_rf_small):
        explainer = LIMEExplainer(training_data=X_small)
        result = explainer.explain_instance(
            instance=X_small[0],
            predict_fn=fitted_rf_small.predict_proba,
            num_features=5,
            num_samples=200,
        )
        assert result is not None

    def test_as_list_has_correct_length(self, X_small, fitted_rf_small):
        explainer = LIMEExplainer(training_data=X_small)
        result = explainer.explain_instance(
            instance=X_small[0],
            predict_fn=fitted_rf_small.predict_proba,
            num_features=5,
            num_samples=200,
        )
        feats = result.as_list()
        assert 1 <= len(feats) <= 5

    def test_feature_weights_are_floats(self, X_small, fitted_rf_small):
        explainer = LIMEExplainer(training_data=X_small)
        result = explainer.explain_instance(
            instance=X_small[0],
            predict_fn=fitted_rf_small.predict_proba,
            num_features=5,
            num_samples=200,
        )
        for _, w in result.as_list():
            assert isinstance(w, float)

    def test_different_instances_differ(self, X_small, fitted_rf_small):
        explainer = LIMEExplainer(training_data=X_small)
        e1 = explainer.explain_instance(X_small[0], fitted_rf_small.predict_proba,
                                         num_features=5, num_samples=200)
        e2 = explainer.explain_instance(X_small[1], fitted_rf_small.predict_proba,
                                         num_features=5, num_samples=200)
        assert e1.as_list() != e2.as_list() or X_small[0].tolist() == X_small[1].tolist()


class TestAsDict:
    def test_dict_has_required_keys(self, X_small, fitted_rf_small):
        explainer = LIMEExplainer(training_data=X_small)
        exp = explainer.explain_instance(
            X_small[0], fitted_rf_small.predict_proba,
            num_features=5, num_samples=200,
        )
        result = explainer.as_dict(exp)
        for key in ("label", "label_name", "score", "local_pred", "features"):
            assert key in result

    def test_features_list_has_dicts(self, X_small, fitted_rf_small):
        explainer = LIMEExplainer(training_data=X_small)
        exp = explainer.explain_instance(
            X_small[0], fitted_rf_small.predict_proba,
            num_features=5, num_samples=200,
        )
        result = explainer.as_dict(exp)
        for item in result["features"]:
            assert "feature" in item and "weight" in item


class TestPlotExplanation:
    def test_returns_figure(self, X_small, fitted_rf_small):
        import matplotlib.pyplot as plt
        explainer = LIMEExplainer(training_data=X_small)
        exp = explainer.explain_instance(
            X_small[0], fitted_rf_small.predict_proba,
            num_features=5, num_samples=200,
        )
        fig = explainer.plot_explanation(exp)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMakeKerasPredict:
    def test_wrapper_returns_proba_shape(self, X_small, fitted_rf_small):
        """Smoke-test the Keras wrapper using an sklearn model as stand-in."""
        class _FakeKeras:
            def predict(self, X, verbose=0):
                return fitted_rf_small.predict_proba(
                    X.reshape(X.shape[0], -1)[:, :X_small.shape[1]]
                )

        fake = _FakeKeras()
        fn   = make_keras_predict_fn(fake, time_steps=3)
        out  = fn(X_small[:5])
        assert out.shape[0] == 5
