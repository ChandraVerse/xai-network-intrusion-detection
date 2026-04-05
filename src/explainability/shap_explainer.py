"""Unified SHAP explainer wrapper for Random Forest, XGBoost, and LSTM.

Selects TreeExplainer (RF/XGBoost) or DeepExplainer (LSTM) based on model type.
Returns SHAP values and expected base value for downstream visualisation.

Public API
----------
explain_tree(model, X, feature_names, max_samples)  -> shap.Explanation
explain(model, X, model_type, background_data, ...)  -> (shap_values, expected_value)
explain_single(model, x, model_type, predicted_class, ...) -> (shap_vals_1d, base_val)
build_explainer(model, model_type, background_data)  -> shap.Explainer

Test contracts (test_explainability.py)
---------------------------------------
- explain_tree returns shap.Explanation
- .values.shape == (max_samples, n_features)  [2-D; multi-class averaged]
- .feature_names preserved when passed
- .base_values is not None
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ModelType = Literal["tree", "deep"]


# ---------------------------------------------------------------------------
# explain_tree  — primary function used by tests/test_explainability.py
# ---------------------------------------------------------------------------

def explain_tree(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 100,
    check_additivity: bool = False,
) -> shap.Explanation:
    """Compute SHAP values for a tree model and return a shap.Explanation.

    Shape contract
    --------------
    .values        -> np.ndarray, shape (n_samples, n_features)  [always 2-D]
    .base_values   -> np.ndarray, shape (n_samples,)
    .feature_names -> feature_names argument (or None)

    For multi-class RF the per-class arrays are mean-abs collapsed to 2-D so
    the contract is identical regardless of the number of classes.

    Args:
        model:            Fitted sklearn RandomForest or XGBoost model.
        X:                Feature matrix, shape (n_samples, n_features).
        feature_names:    Optional list of feature names (length == n_features).
        max_samples:      Subsample X to at most this many rows for speed.
        check_additivity: Passed to TreeExplainer.shap_values.

    Returns:
        shap.Explanation with .values, .base_values, and .feature_names set.
    """
    # Subsample deterministically
    n = min(max_samples, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sub = np.asarray(X[idx], dtype=np.float32)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sub, check_additivity=check_additivity)
    expected = explainer.expected_value

    # ------------------------------------------------------------------ #
    # Normalise to shape (n_samples, n_features)                          #
    # ------------------------------------------------------------------ #
    if isinstance(raw, list):
        # Multi-class sklearn RF: list of (n_samples, n_features) arrays,
        # one per class.  Reduce to mean absolute attribution.
        stacked = np.stack(raw, axis=0)          # (n_classes, n_samples, n_features)
        values_2d = np.mean(np.abs(stacked), axis=0)  # (n_samples, n_features)
        base_val = (
            float(np.mean(expected))
            if hasattr(expected, "__len__")
            else float(expected)
        )
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # Some SHAP versions return (n_samples, n_features, n_classes)
        values_2d = np.mean(np.abs(raw), axis=-1)     # (n_samples, n_features)
        base_val = (
            float(np.mean(expected))
            if hasattr(expected, "__len__")
            else float(expected)
        )
    else:
        # Binary classifier or regression: already (n_samples, n_features)
        values_2d = raw
        base_val = float(expected)

    # base_values must be a 1-D array of length n_samples for shap.Explanation
    base_values_arr = np.full(n, base_val, dtype=np.float64)

    explanation = shap.Explanation(
        values=values_2d,
        base_values=base_values_arr,
        data=X_sub,
        feature_names=feature_names,
    )
    return explanation


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

def build_explainer(
    model,
    model_type: ModelType,
    background_data: np.ndarray | None = None,
    n_background: int = 100,
) -> shap.Explainer:
    """Return the appropriate SHAP explainer for the given model type."""
    if model_type == "tree":
        log.info("Using SHAP TreeExplainer")
        return shap.TreeExplainer(model)
    elif model_type == "deep":
        if background_data is None:
            raise ValueError(
                "background_data is required for DeepExplainer (LSTM)"
            )
        bg = shap.sample(background_data, min(n_background, len(background_data)))
        log.info("Using SHAP DeepExplainer with %d background samples", len(bg))
        return shap.DeepExplainer(model, bg)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Use 'tree' or 'deep'."
        )


def explain(
    model,
    X: np.ndarray,
    model_type: ModelType,
    background_data: np.ndarray | None = None,
    check_additivity: bool = False,
) -> tuple[np.ndarray, np.ndarray | float]:
    """Compute raw SHAP values for the given input matrix X."""
    explainer = build_explainer(model, model_type, background_data)
    log.info("Computing SHAP values for %d samples...", len(X))
    if model_type == "tree":
        shap_values = explainer.shap_values(
            X, check_additivity=check_additivity
        )
    else:
        shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    log.info("SHAP values computed. Shape: %s", np.array(shap_values).shape)
    return shap_values, expected_value


def explain_single(
    model,
    x: np.ndarray,
    model_type: ModelType,
    predicted_class: int,
    background_data: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Explain a single flow (for SOC per-alert triage)."""
    x = np.atleast_2d(x)
    shap_values, expected_value = explain(
        model, x, model_type, background_data
    )
    if isinstance(shap_values, list):
        shap_vals_1d = shap_values[predicted_class][0]
        base_val = (
            expected_value[predicted_class]
            if hasattr(expected_value, "__len__")
            else float(expected_value)
        )
    else:
        shap_vals_1d = shap_values[0]
        base_val = float(expected_value)
    return shap_vals_1d, base_val


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SHAP values for XAI-NIDS")
    p.add_argument("--model", required=True, help="Path to .pkl model file")
    p.add_argument("--type",  default="tree", choices=["tree", "deep"])
    p.add_argument("--data",  required=True, help="CSV of flows to explain")
    p.add_argument("--bg",    default=None,  help="Background CSV for DeepExplainer")
    p.add_argument("--label", default="label_encoded")
    p.add_argument("--out",   default="models/shap_values.npy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Loading model from %s", args.model)
    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    feature_cols = [c for c in df.columns if c != args.label]
    X = df[feature_cols].values.astype(np.float32)
    bg = None
    if args.bg:
        bg_df = pd.read_csv(args.bg)
        bg = bg_df[feature_cols].values.astype(np.float32)
    shap_values, expected_value = explain(model, X, args.type, bg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), np.array(shap_values))
    log.info("SHAP values saved -> %s", out_path)


if __name__ == "__main__":
    main()
