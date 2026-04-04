"""
shap_explainer.py
-----------------
SHAP TreeExplainer (RF / XGBoost) and DeepExplainer (LSTM) wrappers.

Usage:
    python src/explainability/shap_explainer.py \
        --model models/random_forest.pkl \
        --data  data/samples/sample_100.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import shap


def explain_tree(
    model,
    X: np.ndarray,
    feature_names: list = None,
    max_samples: int = 500,
) -> shap.Explanation:
    """
    Compute SHAP values for a tree-based model (RF or XGBoost)
    using TreeExplainer.

    Parameters
    ----------
    model         : fitted RandomForestClassifier or XGBClassifier
    X             : feature matrix (numpy array or DataFrame)
    feature_names : list of feature names for plot labels
    max_samples   : number of samples to explain (subset for speed)

    Returns shap.Explanation object.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values

    X_sample = X[:max_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    if feature_names and shap_values.feature_names is None:
        shap_values.feature_names = feature_names

    print(f"  [SHAP] TreeExplainer — computed SHAP values for "
          f"{X_sample.shape[0]} samples, {X_sample.shape[1]} features.")
    return shap_values


def explain_deep(
    model,
    X_3d: np.ndarray,
    X_background: np.ndarray,
    feature_names: list = None,
    max_samples: int = 200,
) -> np.ndarray:
    """
    Compute SHAP values for a Keras LSTM model using DeepExplainer.

    Parameters
    ----------
    model         : compiled/trained Keras Sequential LSTM model
    X_3d          : 3D input tensor (samples, time_steps, features)
    X_background  : small background dataset for SHAP baseline (3D)
    feature_names : feature names (unused in DeepExplainer output,
                    kept for interface consistency)
    max_samples   : number of samples to explain

    Returns numpy array of SHAP values shape (samples, time_steps, features).
    """
    background = X_background[:100]
    X_explain  = X_3d[:max_samples]

    explainer  = shap.DeepExplainer(model, background)
    shap_vals  = explainer.shap_values(X_explain)

    print(f"  [SHAP] DeepExplainer — computed SHAP values for "
          f"{X_explain.shape[0]} LSTM samples.")
    return shap_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP TreeExplainer on saved model")
    parser.add_argument("--model", required=True, help="Path to .pkl model")
    parser.add_argument("--data",  required=True, help="Path to feature CSV")
    parser.add_argument("--label", default="Label")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    df    = pd.read_csv(args.data)
    X     = df.drop(columns=[args.label], errors="ignore").values
    names = df.drop(columns=[args.label], errors="ignore").columns.tolist()

    model      = joblib.load(args.model)
    shap_vals  = explain_tree(model, X, feature_names=names,
                              max_samples=args.max_samples)
    print("  [SHAP] Top 10 mean |SHAP| values:")
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)   # average over classes
    top10 = np.argsort(mean_abs)[::-1][:10]
    for i, idx in enumerate(top10, 1):
        print(f"    {i:2d}. {names[idx]:<45s}  {mean_abs[idx]:.4f}")
