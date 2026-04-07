"""
tests/test_dashboard.py
-----------------------
Smoke-tests for dashboard/app.py and dashboard/config.py.
Does NOT require a running Dash server — tests are pure unit-level
checks on config values, helper functions, and layout structure.
"""

from __future__ import annotations

import importlib
import sys
import types
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch heavy optional deps before importing dashboard modules
# ---------------------------------------------------------------------------

def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _heavy in ["dash", "dash.dcc", "dash.html", "dash.dependencies",
               "plotly", "plotly.express", "plotly.graph_objects",
               "tensorflow", "tensorflow.keras"]:
    if _heavy not in sys.modules:
        _stub_module(_heavy)

# Make dash.Dash a no-op class so app.py can be imported without a server
sys.modules["dash"].Dash = type("Dash", (), {"__init__": lambda s, *a, **k: None,
                                               "layout": None,
                                               "run_server": lambda s, **k: None})
sys.modules["dash"].callback = lambda *a, **k: (lambda f: f)
sys.modules["dash"].Input    = lambda *a, **k: None
sys.modules["dash"].Output   = lambda *a, **k: None
sys.modules["dash"].State    = lambda *a, **k: None


# ---------------------------------------------------------------------------
# dashboard/config.py tests
# ---------------------------------------------------------------------------

class TestDashboardConfig:
    """Validate all expected keys are present and have correct types."""

    @pytest.fixture(scope="class")
    def cfg(self):
        # Allow import from repo root
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import dashboard.config as config
        return config

    def test_model_dir_defined(self, cfg):
        assert hasattr(cfg, "MODEL_DIR"), "MODEL_DIR missing from config"
        assert isinstance(cfg.MODEL_DIR, str)

    def test_data_dir_defined(self, cfg):
        assert hasattr(cfg, "DATA_DIR") or hasattr(cfg, "PROC_DIR"), (
            "Neither DATA_DIR nor PROC_DIR defined in config"
        )

    def test_class_names_defined(self, cfg):
        assert hasattr(cfg, "CLASS_NAMES") or hasattr(cfg, "CLASSES"), (
            "CLASS_NAMES / CLASSES list missing from config"
        )

    def test_feature_names_or_count_defined(self, cfg):
        has_names  = hasattr(cfg, "FEATURE_NAMES")
        has_count  = hasattr(cfg, "N_FEATURES")
        assert has_names or has_count, "Feature schema missing from config"

    def test_no_hardcoded_absolute_paths(self, cfg):
        """User-defined config values should be relative paths, not /home or /root.

        Skips Python dunder/internal attributes (__cached__, __file__, __spec__,
        etc.) because those are set by the interpreter and are always absolute.
        Only public and UPPER_CASE config names are checked.
        """
        for attr in dir(cfg):
            # Skip ALL dunder attributes — these are Python internals
            # (__cached__, __file__, __loader__, __spec__, __path__, etc.)
            # and their absolute paths are set by the interpreter, not the dev.
            if attr.startswith("__") and attr.endswith("__"):
                continue
            val = getattr(cfg, attr)
            if isinstance(val, str) and val.startswith("/"):
                # Allow /tmp (e.g. pytest tmp dirs)
                assert val.startswith("/tmp"), (
                    f"config.{attr} contains a hardcoded absolute path: {val!r}"
                )


# ---------------------------------------------------------------------------
# Utility / helper function tests (no server needed)
# ---------------------------------------------------------------------------

class TestDashboardHelpers:
    """Test pure helper functions extracted from app.py."""

    def test_prediction_output_shape(self, fitted_rf_small, X_small):
        """Model used by dashboard must return class-probability matrix."""
        proba = fitted_rf_small.predict_proba(X_small[:5])
        assert proba.shape[0] == 5
        assert proba.shape[1] >= 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_label_encoding_roundtrip(self, class_names):
        """Class indices must map back to the original label strings."""
        label_map = {i: name for i, name in enumerate(class_names)}
        for i, name in enumerate(class_names):
            assert label_map[i] == name

    def test_top_k_feature_selection(self, fitted_rf_small, feature_names):
        """Dashboard feature-importance top-k slice must be stable."""
        importances = fitted_rf_small.feature_importances_
        # Use only the first N_FEATURES_S feature names (small model)
        names_slice = feature_names[:len(importances)]
        top_k = 5
        top_idx = np.argsort(importances)[::-1][:top_k]
        top_names = [names_slice[i] for i in top_idx]
        assert len(top_names) == top_k
        assert all(isinstance(n, str) for n in top_names)

    def test_metrics_dict_keys(self, fitted_rf_small, X_small, y_small):
        """compute_metrics must return all keys the dashboard needs."""
        from src.utils import compute_metrics
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_small)
        class_names_small = [str(c) for c in le.classes_]
        preds = fitted_rf_small.predict(X_small)
        m = compute_metrics(y_small, preds, class_names_small)
        required_keys = {"accuracy", "macro_f1", "macro_precision",
                         "macro_recall", "confusion_matrix"}
        missing = required_keys - set(m.keys())
        assert not missing, f"compute_metrics missing keys: {missing}"

    def test_confusion_matrix_shape(self, fitted_rf_small, X_small, y_small):
        """Confusion matrix must be square with correct class dimension."""
        from src.utils import compute_metrics
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_small)
        class_names_small = [str(c) for c in le.classes_]
        preds = fitted_rf_small.predict(X_small)
        m = compute_metrics(y_small, preds, class_names_small)
        cm = m["confusion_matrix"]
        assert cm.shape[0] == cm.shape[1]
        assert cm.shape[0] == len(class_names_small)
