"""Unified SHAP explainer wrapper for Random Forest, XGBoost, and LSTM.

Selects TreeExplainer (RF/XGBoost) or DeepExplainer (LSTM) based on model type.
Returns SHAP values and expected base value for downstream visualisation.

Usage (CLI):
    python src/explainability/shap_explainer.py \\
        --model  models/random_forest.pkl \\
        --type   tree \\
        --data   data/samples/sample_100.csv \\
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
# Explainer factory
# ---------------------------------------------------------------------------

def build_explainer(
    model,
    model_type: ModelType,
    background_data: np.ndarray | None = None,
    n_background: int = 100,
) -> shap.Explainer:
    """Return the appropriate SHAP explainer for the given model type.

    Args:
        model:           Trained sklearn, XGBoost, or Keras model.
        model_type:      'tree' for RF/XGBoost, 'deep' for LSTM.
        background_data: Required for DeepExplainer — sample of training data.
        n_background:    Number of background samples for DeepExplainer.

    Returns:
        Configured SHAP explainer instance.
    """
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
    """Compute SHAP values for the given input matrix X.

    Args:
        model:            Trained model.
        X:                Feature matrix, shape (n_samples, n_features).
        model_type:       'tree' or 'deep'.
        background_data:  Training data sample for DeepExplainer.
        check_additivity: If True, verify SHAP values sum to prediction (TreeExplainer only).

    Returns:
        shap_values:     Array of shape (n_classes, n_samples, n_features) for tree models,
                         or (n_samples, n_features) for a single predicted class.
        expected_value:  Base value(s) — model output when no features are present.
    """
    explainer = build_explainer(model, model_type, background_data)

    log.info("Computing SHAP values for %d samples...", len(X))
    if model_type == "tree":
        shap_values = explainer.shap_values(X, check_additivity=check_additivity)
    else:
        shap_values = explainer.shap_values(X)

    expected_value = explainer.expected_value
    log.info(
        "SHAP values computed. Shape: %s",
        np.array(shap_values).shape,
    )
    return shap_values, expected_value


def explain_single(
    model,
    x: np.ndarray,
    model_type: ModelType,
    predicted_class: int,
    background_data: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Explain a single flow (for SOC per-alert triage).

    Args:
        model:           Trained model.
        x:               Single sample, shape (n_features,) or (1, n_features).
        model_type:      'tree' or 'deep'.
        predicted_class: The class index to extract SHAP values for.
        background_data: Required for DeepExplainer.

    Returns:
        shap_vals_1d:   SHAP values for predicted_class, shape (n_features,)
        base_val:       Scalar expected value for predicted_class.
    """
    x = np.atleast_2d(x)
    shap_values, expected_value = explain(
        model, x, model_type, background_data
    )

    # TreeExplainer returns list[np.ndarray] of length n_classes
    if isinstance(shap_values, list):
        shap_vals_1d = shap_values[predicted_class][0]  # shape (n_features,)
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
    log.info("SHAP values saved → %s", out_path)


if __name__ == "__main__":
    main()
