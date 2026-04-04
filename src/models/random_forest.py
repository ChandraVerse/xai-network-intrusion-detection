"""Random Forest classifier for network intrusion detection.

Trains a RandomForestClassifier on the SMOTE-balanced CICIDS-2017 dataset,
evaluates it on the held-out test set, and serialises the model + metrics.

Usage (CLI):
    python src/models/random_forest.py \\
        --data data/processed/train_balanced.csv \\
        --test data/processed/test.csv \\
        --out  models/

Outputs:
    models/random_forest.pkl       Serialised model (joblib, compress=3)
    models/rf_metrics.json         Accuracy, Macro F1, FPR, inference time
    models/feature_importance_rf.json  Gini feature importance
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_split(
    train_path: str | Path,
    test_path: str | Path,
    label_col: str = "label_encoded",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train / test CSV splits and return feature matrices + label vectors."""
    log.info("Loading training data from %s", train_path)
    train = pd.read_csv(train_path)
    log.info("Loading test data from %s", test_path)
    test = pd.read_csv(test_path)

    feature_cols = [c for c in train.columns if c != label_col]
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[label_col].values
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[label_col].values

    log.info(
        "Train: %s  |  Test: %s  |  Features: %d",
        X_train.shape,
        X_test.shape,
        X_train.shape[1],
    )
    return X_train, y_train, X_test, y_test


def train(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train the Random Forest classifier."""
    log.info("Training RandomForestClassifier  params=%s", RF_PARAMS)
    model = RandomForestClassifier(**RF_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f s", elapsed)
    return model


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model; return metrics dict."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # False Positive Rate (macro-averaged across all classes)
    cm = confusion_matrix(y_test, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr_per_class = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
    mean_fpr = float(np.mean(fpr_per_class))

    metrics = {
        "model": "RandomForest",
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "false_positive_rate": round(mean_fpr, 6),
        "inference_ms_per_flow": round(inference_ms, 4),
        "n_estimators": RF_PARAMS["n_estimators"],
        "n_test_samples": int(len(y_test)),
    }

    log.info("Accuracy : %.4f", acc)
    log.info("Macro F1 : %.4f", macro_f1)
    log.info("Mean FPR : %.4f", mean_fpr)
    log.info("Inference: %.4f ms/flow", inference_ms)
    log.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    return metrics


def save(
    model: RandomForestClassifier,
    metrics: dict,
    feature_cols: list[str],
    out_dir: str | Path,
) -> None:
    """Serialise model, metrics, and feature importances to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = out / "random_forest.pkl"
    joblib.dump(model, model_path, compress=3)
    log.info("Model saved → %s", model_path)

    # Metrics
    metrics_path = out / "rf_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # Feature importances
    importances = dict(
        zip(feature_cols, model.feature_importances_.tolist())
    )
    importances_sorted = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )
    fi_path = out / "feature_importance_rf.json"
    with open(fi_path, "w") as f:
        json.dump(importances_sorted, f, indent=2)
    log.info("Feature importances saved → %s", fi_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Random Forest for XAI-NIDS")
    p.add_argument("--data",  default="data/processed/train_balanced.csv", help="Path to balanced training CSV")
    p.add_argument("--test",  default="data/processed/test.csv",           help="Path to test CSV")
    p.add_argument("--out",   default="models/",                            help="Output directory for model artifacts")
    p.add_argument("--label", default="label_encoded",                      help="Name of the target column")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X_train, y_train, X_test, y_test = load_split(
        args.data, args.test, label_col=args.label
    )
    feature_cols = pd.read_csv(args.data, nrows=0).columns.tolist()
    feature_cols = [c for c in feature_cols if c != args.label]

    model = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save(model, metrics, feature_cols, args.out)
    log.info("Done.")


if __name__ == "__main__":
    main()
