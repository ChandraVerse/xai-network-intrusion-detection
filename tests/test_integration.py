"""
tests/test_integration.py
--------------------------
End-to-end pipeline integration tests.
Each test exercises a full stage of the pipeline using synthetic
in-memory data so no real artefacts need to be present on disk.

Coverage
--------
  Stage 1  Preprocessing  -> scaled arrays + encoders
  Stage 2  Model training -> RF / XGBoost fit + predict
  Stage 3  SHAP           -> explainer runs, output shape correct
  Stage 4  LIME           -> explainer runs, weights returned
  Stage 5  Metrics        -> compute_metrics contract verified
  Stage 6  Report         -> ReportGenerator produces non-empty JSON
  Stage 7  Full pipeline  -> stages 1-5 chained end-to-end
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# Stage 1 — Preprocessing
# ===========================================================================

class TestPreprocessingPipeline:

    def test_scaler_fit_transform_shape(self, X_full, scaler_full):
        X_scaled = scaler_full.transform(X_full)
        assert X_scaled.shape == X_full.shape

    def test_scaled_values_in_range(self, X_full, scaler_full):
        X_scaled = scaler_full.transform(X_full)
        assert X_scaled.min() >= -1e-6
        assert X_scaled.max() <= 1.0 + 1e-6

    def test_label_encoder_roundtrip(self, class_names):
        le = LabelEncoder()
        le.fit(class_names)
        encoded   = le.transform(class_names)
        decoded   = le.inverse_transform(encoded)
        assert list(decoded) == class_names

    def test_no_nan_after_scaling(self, X_full, scaler_full):
        X_scaled = scaler_full.transform(X_full)
        assert not np.isnan(X_scaled).any(), "NaN found after scaling"

    def test_no_inf_after_scaling(self, X_full, scaler_full):
        X_scaled = scaler_full.transform(X_full)
        assert not np.isinf(X_scaled).any(), "Inf found after scaling"


# ===========================================================================
# Stage 2 — Model Training & Inference
# ===========================================================================

class TestModelPipeline:

    def test_rf_predict_shape(self, fitted_rf_full, X_full, y_full):
        preds = fitted_rf_full.predict(X_full)
        assert preds.shape == y_full.shape

    def test_rf_predict_proba_shape(self, fitted_rf_full, X_full, class_names):
        proba = fitted_rf_full.predict_proba(X_full)
        assert proba.shape == (len(X_full), len(class_names))

    def test_rf_proba_sums_to_one(self, fitted_rf_full, X_full):
        proba = fitted_rf_full.predict_proba(X_full)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_xgboost_train_predict(self, X_full, y_full):
        pytest.importorskip("xgboost")
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42, verbosity=0
        )
        clf.fit(X_full, y_full)
        preds = clf.predict(X_full)
        assert preds.shape == y_full.shape
        assert set(preds).issubset(set(y_full))

    def test_rf_feature_importances_shape(self, fitted_rf_full, X_full):
        imps = fitted_rf_full.feature_importances_
        assert imps.shape == (X_full.shape[1],)
        assert np.isclose(imps.sum(), 1.0, atol=1e-5)

    def test_inference_deterministic(self, fitted_rf_full, X_full):
        p1 = fitted_rf_full.predict(X_full[:10])
        p2 = fitted_rf_full.predict(X_full[:10])
        np.testing.assert_array_equal(p1, p2)


# ===========================================================================
# Stage 3 — SHAP Explainability
# ===========================================================================

class TestSHAPPipeline:

    def test_shap_tree_explainer_values_shape(self, fitted_rf_small, X_small):
        shap = pytest.importorskip("shap")
        explainer = shap.TreeExplainer(fitted_rf_small)
        shap_values = explainer.shap_values(X_small[:10])
        if isinstance(shap_values, list):
            # multi-class: list of [n_samples, n_features] arrays
            assert all(sv.shape == (10, X_small.shape[1]) for sv in shap_values)
        else:
            assert shap_values.shape[0] == 10

    def test_shap_values_finite(self, fitted_rf_small, X_small):
        shap = pytest.importorskip("shap")
        explainer = shap.TreeExplainer(fitted_rf_small)
        shap_values = explainer.shap_values(X_small[:10])
        if isinstance(shap_values, list):
            for sv in shap_values:
                assert np.isfinite(sv).all()
        else:
            assert np.isfinite(shap_values).all()

    def test_shap_global_importance_top_feature_valid(
        self, fitted_rf_small, X_small, feature_names
    ):
        shap = pytest.importorskip("shap")
        explainer  = shap.TreeExplainer(fitted_rf_small)
        shap_vals  = explainer.shap_values(X_small[:20])
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx   = int(np.argmax(mean_abs))
        names_slice = feature_names[:X_small.shape[1]]
        top_name  = names_slice[top_idx]
        assert isinstance(top_name, str) and len(top_name) > 0


# ===========================================================================
# Stage 4 — LIME Explainability
# ===========================================================================

class TestLIMEPipeline:

    def test_lime_explainer_returns_weights(
        self, fitted_rf_small, X_small, y_small, feature_names
    ):
        lime = pytest.importorskip("lime")
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_small)
        class_names_small = [str(c) for c in le.classes_]
        names_slice = feature_names[:X_small.shape[1]]
        explainer = LimeTabularExplainer(
            training_data=X_small,
            feature_names=names_slice,
            class_names=class_names_small,
            mode="classification",
            random_state=42,
        )
        exp = explainer.explain_instance(
            X_small[0], fitted_rf_small.predict_proba,
            num_features=5, num_samples=100
        )
        weights = exp.as_list()
        assert len(weights) > 0
        assert all(isinstance(w, tuple) and len(w) == 2 for w in weights)

    def test_lime_weights_are_finite(
        self, fitted_rf_small, X_small, y_small, feature_names
    ):
        pytest.importorskip("lime")
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_small)
        class_names_small = [str(c) for c in le.classes_]
        names_slice = feature_names[:X_small.shape[1]]
        explainer = LimeTabularExplainer(
            training_data=X_small,
            feature_names=names_slice,
            class_names=class_names_small,
            mode="classification",
            random_state=42,
        )
        exp = explainer.explain_instance(
            X_small[0], fitted_rf_small.predict_proba,
            num_features=5, num_samples=100
        )
        for _, w in exp.as_list():
            assert np.isfinite(w)


# ===========================================================================
# Stage 5 — Metrics
# ===========================================================================

class TestMetricsPipeline:

    def test_compute_metrics_accuracy_range(
        self, fitted_rf_full, X_full, y_full, class_names
    ):
        from src.utils import compute_metrics
        preds = fitted_rf_full.predict(X_full)
        m = compute_metrics(y_full, preds, class_names)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_compute_metrics_f1_range(
        self, fitted_rf_full, X_full, y_full, class_names
    ):
        from src.utils import compute_metrics
        preds = fitted_rf_full.predict(X_full)
        m = compute_metrics(y_full, preds, class_names)
        assert 0.0 <= m["macro_f1"] <= 1.0

    def test_compute_metrics_mean_fpr_present(
        self, fitted_rf_full, X_full, y_full, class_names
    ):
        from src.utils import compute_metrics
        preds = fitted_rf_full.predict(X_full)
        m = compute_metrics(y_full, preds, class_names)
        assert "mean_fpr" in m
        assert 0.0 <= m["mean_fpr"] <= 1.0

    def test_format_metrics_for_dashboard(
        self, fitted_rf_full, X_full, y_full, class_names
    ):
        from src.utils import compute_metrics, format_metrics_for_dashboard
        preds = fitted_rf_full.predict(X_full)
        m = compute_metrics(y_full, preds, class_names)
        dash_m = format_metrics_for_dashboard(m)
        assert isinstance(dash_m, dict)
        assert len(dash_m) > 0


# ===========================================================================
# Stage 6 — Report Generator
# ===========================================================================

class TestReportGeneratorPipeline:

    def test_report_generator_creates_json(
        self, fitted_rf_full, X_full, y_full, class_names
    ):
        from src.utils import compute_metrics, ReportGenerator
        preds = fitted_rf_full.predict(X_full)
        m = compute_metrics(y_full, preds, class_names)
        rg = ReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.json")
            rg.save_json(m, out_path)
            assert os.path.exists(out_path)
            with open(out_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert len(data) > 0


# ===========================================================================
# Stage 7 — Full end-to-end chain
# ===========================================================================

class TestFullPipelineChain:
    """Chain all stages: preprocess -> train -> predict -> explain -> metrics."""

    def test_end_to_end_rf_shap_metrics(self, X_full, y_full, class_names):
        shap = pytest.importorskip("shap")

        # Stage 1: scale
        sc = MinMaxScaler()
        X_scaled = sc.fit_transform(X_full)

        # Stage 2: train
        clf = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42
        )
        clf.fit(X_scaled, y_full)

        # Stage 3: SHAP
        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(X_scaled[:10])
        assert shap_vals is not None

        # Stage 4: predict + metrics
        from src.utils import compute_metrics
        preds = clf.predict(X_scaled)
        m = compute_metrics(y_full, preds, class_names)
        assert m["accuracy"] > 0.0
        assert "confusion_matrix" in m

    def test_end_to_end_rf_lime_metrics(self, X_full, y_full, class_names):
        pytest.importorskip("lime")
        from lime.lime_tabular import LimeTabularExplainer

        # Stage 1: scale
        sc = MinMaxScaler()
        X_scaled = sc.fit_transform(X_full)

        # Stage 2: train
        clf = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42
        )
        clf.fit(X_scaled, y_full)

        # Stage 4: LIME on a single instance
        le = LabelEncoder().fit(y_full)
        cn = [str(c) for c in le.classes_]
        exp_obj = LimeTabularExplainer(
            X_scaled, class_names=cn, mode="classification", random_state=42
        )
        exp = exp_obj.explain_instance(
            X_scaled[0], clf.predict_proba,
            num_features=5, num_samples=100
        )
        assert len(exp.as_list()) > 0

        # Stage 5: metrics
        from src.utils import compute_metrics
        preds = clf.predict(X_scaled)
        m = compute_metrics(y_full, preds, class_names)
        assert m["macro_f1"] >= 0.0
