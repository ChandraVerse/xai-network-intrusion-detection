"""
scaler.py
---------
MinMaxScaler wrapper — fit on train set only to prevent data leakage.

Usage:
    from src.preprocessing.scaler import fit_scaler, apply_scaler
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(X_train: pd.DataFrame, save_path: str = None) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training feature matrix.
    Optionally saves the fitted scaler to save_path.

    Returns the fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"  [scaler] Saved MinMaxScaler to {save_path}")

    return scaler


def apply_scaler(X: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """
    Apply a pre-fitted scaler to a feature matrix.
    Returns a numpy array of scaled values.
    """
    return scaler.transform(X)


def load_scaler(path: str) -> MinMaxScaler:
    """Load a serialised MinMaxScaler from disk."""
    scaler = joblib.load(path)
    print(f"  [scaler] Loaded scaler from {path}")
    return scaler
