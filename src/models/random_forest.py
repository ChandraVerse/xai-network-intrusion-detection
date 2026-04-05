"""Random Forest classifier for network intrusion detection.

Trains a RandomForestClassifier on the SMOTE-balanced CICIDS-2017
dataset and exports SHAP feature importances.

Public API (used by tests and dashboard):
    train_random_forest(X, y, n_estimators, max_depth, save_path) -> clf

Usage (CLI):
    python src/models/random_forest.py \\
        --data data/processed/train_balanced.csv \\
        --test data/processed/test.csv \\
        --out  models/

Outputs:
    models/random_forest.pkl        Serialised sklearn model
    models/rf_metrics.json          Accuracy, Macro F1, FPR, timing
    models/rf_shap_summary.png      SHAP beeswarm summary plot
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_ESTIMATORS = 200
MAX_DEPTH = None       # fully grown trees
MIN_SAMPLES_LEAF = 2
RANDOM_STATE = 42
SHAP_SAMPLE_N = 1000   # subsample for SHAP (full dataset is slow)


# ---------------------------------------------------------------------------
# Public API wrapper  (called by tests + dashboard)
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int | None = MAX_DEPTH,
    save_path: str | None = None,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier and optionally save it.

    Parameters
    ----------
    X_train      : feature matrix
    y_train      : integer class labels
    n_estimators : number of trees (default 200)
    max_depth    : max tree depth; None = fully grown
    save_path    : if given, joblib-dumps the model to this path
    random_state : RNG seed for reproducibility

    Returns
    -------
    Fitted RandomForestClassifier
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    log.info(
        "Fitting RandomForest  n_estimators=%d  random_state=%d",
        n_estimators, random_state,
    )
    clf.fit(X_train, y_train)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path, compress=3)
        log.info("Model saved -> %s", save_path)
    return clf


# ---------------------------------------------------------------------------
# Internal pipeline helpers (used by main / CLI)
# ---------------------------------------------------------------------------

def load_split(
    train_path: str | Path,
    test_path: str | Path,
    label_col: str = "label_encoded",
) -> tuple:
    """Load train/test CSVs and return numpy arrays."""
    log.info("Loading training data from %s", train_path)
    train = pd.read_csv(train_path)
    log.info("Loading test data from %s", test_path)
    test = pd.read_csv(test_path)

    feature_cols = [c for c in train.columns if c != label_col]
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[label_col].values.astype(int)
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[label_col].values.astype(int)

    log.info(
        "Train: %s  Test: %s  Features: %d",
        X_train.shape, X_test.shape, X_train.shape[1],
    )
    return X_train, y_train, X_test, y_test, feature_cols


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Fit with CLI-default hyperparameters (used by main())."""
    return train_random_forest(
        X_train, y_train,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        save_path=None,
    )


def evaluate(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Run inference and return metric dict."""
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    inference_time = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    n_classes = len(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)  # noqa: F841
    tn = cm.sum() - (fp + fn + np.diag(cm))
    fpr_per_class = fp / (fp + tn + 1e-9)
    mean_fpr = float(np.mean(fpr_per_class))

    metrics = {
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "mean_fpr": round(mean_fpr, 6),
        "inference_time_s": round(inference_time, 4),
        "n_test_samples": int(len(y_test)),
    }
    log.info("Accuracy=%.4f  Macro-F1=%.4f  Mean-FPR=%.4f", acc, macro_f1, mean_fpr)
    log.info("\n%s", classification_report(y_test, y_pred, zero_division=0))
    return metrics


def compute_shap(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    feature_names: list,
    out_dir: Path,
    sample_n: int = SHAP_SAMPLE_N,
) -> np.ndarray:
    """Compute SHAP values and save a beeswarm summary plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("Computing SHAP values on %d samples", sample_n)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(sample_n, len(X_test)), replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_abs = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        mean_abs = np.abs(shap_values)

    log.info("Generating SHAP beeswarm summary plot")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values if isinstance(shap_values, np.ndarray) else shap_values[1],
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plot_path = out_dir / "rf_shap_summary.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP plot saved -> %s", plot_path)
    return mean_abs


def save_model(clf: RandomForestClassifier, out_dir: Path) -> Path:
    """Serialise model with joblib."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "random_forest.pkl"
    joblib.dump(clf, model_path, compress=3)
    log.info("Model saved -> %s", model_path)
    return model_path


def main(args: argparse.Namespace) -> None:
    """End-to-end train + evaluate + explain + save pipeline."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_cols = load_split(args.data, args.test)
    clf = train(X_train, y_train)
    metrics = evaluate(clf, X_test, y_test)
    compute_shap(clf, X_test, feature_cols, out_dir)
    save_model(clf, out_dir)

    metrics_path = out_dir / "rf_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Metrics saved -> %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XAI-NIDS Random Forest classifier")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--test", required=True, help="Path to test CSV")
    parser.add_argument("--out", default="models/", help="Output directory")
    main(parser.parse_args())
