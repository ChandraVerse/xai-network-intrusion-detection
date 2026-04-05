"""
tests/conftest.py
------------------
Shared pytest fixtures for all test modules.

Available fixtures
------------------
``feature_names``    list[str]  -- 78-column CICIDS-2017 feature list
``class_names``      list[str]  -- 14 CICIDS-2017 attack class labels
``X_small``          np.ndarray -- (200, 15) float32 feature matrix
``y_small``          np.ndarray -- (200,)    int32 labels (4 classes)
``X_full``           np.ndarray -- (280, 78) scaled float32 matrix (14 classes)
``y_full``           np.ndarray -- (280,)    int32 labels (14 classes)
``fitted_rf_small``  sklearn RF -- trained on X_small/y_small
``fitted_rf_full``   sklearn RF -- trained on X_full/y_full
``scaler_full``      MinMaxScaler -- fitted on X_full training split
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_samples import CLASSES, FEATURE_NAMES, gen_class_samples  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SMALL      = 200
N_FEATURES_S = 15
N_CLASSES_S  = 4
SEED         = 42
N_FULL_PER_CLS = 20   # 20 samples per class  × 14 classes = 280 rows


# ---------------------------------------------------------------------------
# Static schema fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feature_names() -> list[str]:
    return FEATURE_NAMES


@pytest.fixture(scope="session")
def class_names() -> list[str]:
    return CLASSES


# ---------------------------------------------------------------------------
# Small (fast) dataset — 15 features, 4 classes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def X_small() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.random((N_SMALL, N_FEATURES_S)).astype(np.float32)


@pytest.fixture(scope="session")
def y_small(X_small) -> np.ndarray:  # noqa: F811
    rng = np.random.default_rng(SEED + 1)
    return rng.integers(0, N_CLASSES_S, N_SMALL).astype(np.int32)


@pytest.fixture(scope="session")
def fitted_rf_small(X_small, y_small) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=10, max_depth=5, random_state=SEED, n_jobs=1
    )
    clf.fit(X_small, y_small)
    return clf


# ---------------------------------------------------------------------------
# Full (78-feature, 14-class) dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def X_full() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    rows: list = []
    for cls in CLASSES:
        rows += gen_class_samples(cls, N_FULL_PER_CLS, rng)
    return np.array(rows, dtype=np.float32)


@pytest.fixture(scope="session")
def y_full() -> np.ndarray:
    labels: list[int] = []
    for i, _ in enumerate(CLASSES):
        labels += [i] * N_FULL_PER_CLS
    return np.array(labels, dtype=np.int32)


@pytest.fixture(scope="session")
def scaler_full(X_full) -> MinMaxScaler:  # noqa: F811
    sc = MinMaxScaler()
    sc.fit(X_full)
    return sc


@pytest.fixture(scope="session")
def fitted_rf_full(X_full, y_full) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=20, max_depth=8, random_state=SEED,
        class_weight="balanced", n_jobs=-1,
    )
    clf.fit(X_full, y_full)
    return clf
