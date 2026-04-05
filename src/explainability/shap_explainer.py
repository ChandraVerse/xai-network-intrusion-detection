"""Unified SHAP explainer wrapper for Random Forest, XGBoost, and LSTM.

Selects TreeExplainer (RF/XGBoost) or DeepExplainer (LSTM) based on model type.
Returns SHAP values and expected base value for downstream visualisation.

Public API
----------
explain_tree(model, X, feature_names, max_samples)  -> shap.Explanation
explain(model, X, model_type, background_data, ...)  -> (shap_values, expected_value)
explain_single(model, x, model_type, predicted_class, ...) -> (shap_vals_1d, base_val)
build_explainer(model, model_type, background_data)  -> shap.Explainer

Usage (CLI)::
    python src/explainability/shap_explainer.py \\\
        --model  models/random_forest.pkl \\\
        --type   tree \\\
        --data   data/samples/sample_100.csv \\\
        --out    models/rf_shap_values.npy
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
    """Compute SHAP values for a tree model and return a shap.Explanation object.

    Args:
        model:           Fitted sklearn RandomForest or XGBoost model.
        X:               Feature matrix, shape (n_samples, n_features).
        feature_names:   Optional list of feature names (length == n_features).
        max_samples:     Subsample X to at most this many rows for speed.
        check_additivity: Passed to TreeExplainer.shap_values (tree models only).

    Returns:
        shap.Explanation with .values, .base_values, and .feature_names set.
    """
    # Subsample for speed
    n = min(max_samples, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sub = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub, check_additivity=check_additivity)
    expected_value = explainer.expected_value

    # Normalise to a single array (multi-class RF returns list of arrays)
    if isinstance(shap_values, list):
        # Stack -> (n_classes, n_samples, n_features); take mean over classes
        # so .values has shape (n_samples, n_features) — simplest for tests
        values_arr = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
        base_val = (
            float(np.mean(expected_value))
            if hasattr(expected_value, "__len__")
            else float(expected_value)
        )
    else:
        values_arr = shap_values
        base_val = float(expected_value)

    # Build shap.Explanation
    explanation = shap.Explanation(
        values=values_arr,
        base_values=np.full(n, base_val),
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
            raise ValueError("background_data is required for DeepExplainer (LSTM)")
        bg = shap.sample(background_data, min(n_background, len(background_data)))
        log.info("Using SHAP DeepExplainer with %d background samples", len(bg))
        return shap.DeepExplainer(model, bg)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'tree' or 'deep'.")


def explain(
    model,
    X: np.ndarray,
    model_type: ModelType,
    background_data: np.ndarray | None = None,
    check_additivity: bool = False,
) -> tuple[np.ndarray, np.ndarray | float]:
    """Compute SHAP values for the given input matrix X."""
    explainer = build_explainer(model, model_type, background_data)
    log.info("Computing SHAP values for %d samples...", len(X))
    if model_type == "tree":
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)
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
    shap_values, expected_value = explain(model, x, model_type, background_data)
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
    p.add_argument("--type", default="tree", choices=["tree", "deep"])
    p.add_argument("--data", required=True, help="CSV of flows to explain")
    p.add_argument("--bg", default=None, help="Background CSV for DeepExplainer")
    p.add_argument("--label", default="label_encoded")
    p.add_argument("--out", default="models/shap_values.npy")
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
