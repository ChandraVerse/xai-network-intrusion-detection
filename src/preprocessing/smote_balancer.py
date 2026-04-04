"""
smote_balancer.py
-----------------
SMOTE over-sampling for minority attack classes.
Applied to the TRAINING set only — never touch the test set.

Usage:
    from src.preprocessing.smote_balancer import apply_smote
"""
import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "not majority",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple:
    """
    Apply SMOTE to the training set.

    Parameters
    ----------
    X_train      : scaled feature matrix (numpy array)
    y_train      : integer-encoded label vector
    strategy     : SMOTE sampling_strategy ('not majority' by default)
    random_state : reproducibility seed
    k_neighbors  : number of nearest neighbours for synthetic sample generation

    Returns
    -------
    X_resampled, y_resampled as numpy arrays
    """
    print(f"  [smote] Before SMOTE — X: {X_train.shape}, class counts: "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    smote = SMOTE(
        sampling_strategy=strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  [smote] After SMOTE  — X: {X_res.shape}, class counts: "
          f"{dict(zip(*np.unique(y_res, return_counts=True)))}")

    return X_res, y_res
