"""SMOTE oversampler for XAI-NIDS minority class balancing.

Applied to the TRAINING SET ONLY after train/test split and after
MinMaxScaling -- applying SMOTE before scaling would distort the
scaler's learned feature range.

Usage:
    from src.preprocessing.smote_balancer import apply_smote

    X_train_bal, y_train_bal = apply_smote(
        X_train_scaled, y_train,
        strategy="not majority",
        random_state=42,
    )

Requires: imbalanced-learn >= 0.10  (pip install imbalanced-learn)
"""

import logging

import numpy as np

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "imbalanced-learn is required: pip install imbalanced-learn>=0.10"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str | dict = "not majority",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to the training set.

    Args:
        X_train: 2D float array (MinMax-scaled), shape (n_samples, n_features).
        y_train: Integer label array, shape (n_samples,).
        strategy: SMOTE resampling strategy. Default 'not majority' balances
            all minority classes up to the majority class count.
        random_state: RNG seed for reproducibility.
        k_neighbors: Number of nearest neighbours for SMOTE synthesis.

    Returns:
        Tuple (X_balanced, y_balanced) with shapes (n_balanced, n_features)
        and (n_balanced,) respectively.
    """
    from collections import Counter
    before = Counter(y_train)
    log.info("Class distribution BEFORE SMOTE: %s", dict(before))

    sm = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
        n_jobs=-1,
    )
    X_bal, y_bal = sm.fit_resample(X_train, y_train)

    after = Counter(y_bal)
    log.info("Class distribution AFTER SMOTE:  %s", dict(after))
    log.info(
        "Samples added: %d  (%.1f%% increase)",
        len(y_bal) - len(y_train),
        100.0 * (len(y_bal) - len(y_train)) / len(y_train),
    )
    return X_bal.astype(np.float32), y_bal.astype(np.int32)
