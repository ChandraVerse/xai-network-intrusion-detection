"""
tests/test_preprocessing.py
----------------------------
Unit tests for the preprocessing pipeline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.cleaner import clean_dataframe
from src.preprocessing.scaler import apply_scaler, fit_scaler
from src.preprocessing.smote_balancer import apply_smote


# ─── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small synthetic CICIDS-like DataFrame for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Flow Duration": np.random.exponential(1000, n),
        "Total Fwd Packets": np.random.randint(1, 100, n).astype(float),
        "Flow Bytes/s": np.random.exponential(5000, n),
        "Zero Variance Feature": np.zeros(n),
        "Flow ID": [f"flow_{i}" for i in range(n)],
        " Label": np.random.choice(["BENIGN", "DDoS", "PortScan"], n),
    })
    # Inject some Inf and NaN
    df.loc[0, "Flow Duration"] = np.inf
    df.loc[1, "Total Fwd Packets"] = np.nan
    return df


@pytest.fixture
def clean_df(sample_df):
    return clean_dataframe(sample_df.copy())


# ─── cleaner.py tests ────────────────────────────────────────────

class TestCleaner:
    def test_no_inf(self, clean_df):
        """No Inf values should remain after cleaning."""
        numeric = clean_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Inf values found after cleaning"

    def test_no_nan(self, clean_df):
        """No NaN values should remain after cleaning."""
        numeric = clean_df.select_dtypes(include=[np.number])
        assert not numeric.isnull().any().any(), "NaN values found after cleaning"

    def test_zero_variance_dropped(self, clean_df):
        """Zero-variance numeric columns must be removed."""
        assert "Zero Variance Feature" not in clean_df.columns

    def test_flow_id_dropped(self, clean_df):
        """Non-feature identifier columns must be removed."""
        assert "Flow ID" not in clean_df.columns

    def test_label_preserved(self, clean_df):
        """Label column should survive cleaning."""
        label_col = [c for c in clean_df.columns if "Label" in c or "label" in c]
        assert len(label_col) >= 1, "Label column missing after cleaning"

    def test_output_shape(self, sample_df, clean_df):
        """Cleaned DataFrame should have fewer columns and same row count."""
        assert clean_df.shape[0] == sample_df.shape[0]
        assert clean_df.shape[1] < sample_df.shape[1]


# ─── scaler.py tests ─────────────────────────────────────────────

class TestScaler:
    def test_scale_range(self, clean_df):
        """All scaled values must be in [0, 1]."""
        label_col = [c for c in clean_df.columns if "Label" in c or "label" in c][0]
        X = clean_df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        scaler = fit_scaler(X)
        X_scaled = apply_scaler(X, scaler)
        assert X_scaled.min() >= 0.0 - 1e-9
        assert X_scaled.max() <= 1.0 + 1e-9

    def test_scaler_shape(self, clean_df):
        """Scaled matrix shape must match input."""
        label_col = [c for c in clean_df.columns if "Label" in c or "label" in c][0]
        X = clean_df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        scaler = fit_scaler(X)
        X_scaled = apply_scaler(X, scaler)
        assert X_scaled.shape == X.shape


# ─── smote_balancer.py tests ───────────────────────────────────────────

class TestSMOTE:
    def test_smote_increases_minority(self):
        """SMOTE should increase samples in minority classes."""
        np.random.seed(42)
        # Imbalanced: class 0 = 400, class 1 = 40, class 2 = 20
        X = np.random.rand(460, 5)
        y = np.array([0] * 400 + [1] * 40 + [2] * 20)

        X_res, y_res = apply_smote(X, y, strategy="not majority")
        counts_after = dict(zip(*np.unique(y_res, return_counts=True)))

        assert counts_after[1] >= 40, "Class 1 not oversampled"
        assert counts_after[2] >= 20, "Class 2 not oversampled"

    def test_smote_preserves_features(self):
        """SMOTE should not change number of features."""
        np.random.seed(0)
        X = np.random.rand(300, 10)
        y = np.array([0] * 250 + [1] * 50)
        X_res, y_res = apply_smote(X, y)
        assert X_res.shape[1] == 10
