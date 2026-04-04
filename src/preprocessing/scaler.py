"""MinMaxScaler wrapper for XAI-NIDS preprocessing.

Fits MinMaxScaler on training data only (no data leakage) and
serialises the fitted scaler to disk so it can be reloaded during
inference and Streamlit dashboard operation.

Usage:
    from src.preprocessing.scaler import fit_scaler, apply_scaler, load_scaler

    scaler = fit_scaler(X_train, save_path="data/processed/scaler.pkl")
    X_train_scaled = apply_scaler(X_train, scaler)
    X_test_scaled  = apply_scaler(X_test, scaler)

Note:
    Only call fit_scaler() on the TRAINING set.
    Always use apply_scaler() (transform only) on the test/val sets.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def fit_scaler(
    X_train: np.ndarray,
    save_path: str | Path | None = None,
) -> MinMaxScaler:
    """Fit a MinMaxScaler on X_train and optionally save to disk.

    Args:
        X_train: 2D float array, shape (n_samples, n_features).
        save_path: Optional path to persist the fitted scaler (joblib).

    Returns:
        Fitted MinMaxScaler instance.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    log.info(
        "MinMaxScaler fitted  |  features=%d  |  data_min_[:5]=%s  data_max_[:5]=%s",
        X_train.shape[1],
        scaler.data_min_[:5],
        scaler.data_max_[:5],
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        log.info("Scaler saved -> %s", save_path)
    return scaler


def apply_scaler(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Apply a pre-fitted scaler to X (transform only, no fitting).

    Args:
        X: 2D float array, same number of features as training data.
        scaler: A fitted MinMaxScaler instance.

    Returns:
        Scaled array, same shape as X, dtype float32.
    """
    scaled = scaler.transform(X)
    return scaled.astype(np.float32)


def load_scaler(path: str | Path) -> MinMaxScaler:
    """Load a serialised MinMaxScaler from disk.

    Args:
        path: Path to the joblib-serialised scaler file.

    Returns:
        Loaded MinMaxScaler instance ready for transform().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at: {path}")
    scaler = joblib.load(path)
    log.info("Scaler loaded from %s", path)
    return scaler
