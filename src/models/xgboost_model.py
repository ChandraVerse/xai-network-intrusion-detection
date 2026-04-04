"""XGBoost classifier for network intrusion detection.

Trains an XGBClassifier on the SMOTE-balanced CICIDS-2017 dataset with early
stopping, evaluates on the held-out test set, and serialises the model + metrics.

Usage (CLI):
    python src/models/xgboost_model.py \\
        --data data/processed/train_balanced.csv \\
        --test data/processed/test.csv \\
        --out  models/

Outputs:
    models/xgboost_model.pkl   Serialised model (joblib, compress=3)
    models/xgb_metrics.json    Accuracy, Macro F1, FPR, inference time
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "tree_method": "hist",  # fast histogram method; use 'gpu_hist' for GPU
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

EARLY_STOPPING_ROUNDS = 20  # stop if mlogloss doesn't improve for 20 rounds


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_split(
    train_path: str | Path,
    test_path: str | Path,
    label_col: str = "label_encoded",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load CSV splits, return arrays and feature column names."""
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
        "Train: %s  |  Test: %s  |  Features: %d  |  Classes: %d",
        X_train.shape, X_test.shape,
        X_train.shape[1], len(np.unique(y_train)),
    )
    return X_train, y_train, X_test, y_test, feature_cols


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> XGBClassifier:
    """Train XGBoost with early stopping on a 10% validation split."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=42
    )

    params = {**XGB_PARAMS, "num_class": n_classes, "objective": "multi:softprob"}
    model = XGBClassifier(**params)

    log.info("Training XGBClassifier  params=%s", XGB_PARAMS)
    t0 = time.perf_counter()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f s  |  best_iteration=%d",
             elapsed, model.best_iteration)
    return model


def evaluate(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate model; return metrics dict."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr_per_class = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
    mean_fpr = float(np.mean(fpr_per_class))

    metrics = {
        "model": "XGBoost",
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "false_positive_rate": round(mean_fpr, 6),
        "inference_ms_per_flow": round(inference_ms, 4),
        "best_iteration": int(model.best_iteration),
        "n_test_samples": int(len(y_test)),
    }

    log.info("Accuracy : %.4f", acc)
    log.info("Macro F1 : %.4f", macro_f1)
    log.info("Mean FPR : %.4f", mean_fpr)
    log.info("Inference: %.4f ms/flow", inference_ms)
    log.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    return metrics


def save(
    model: XGBClassifier,
    metrics: dict,
    out_dir: str | Path,
) -> None:
    """Serialise model and metrics to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "xgboost_model.pkl"
    joblib.dump(model, model_path, compress=3)
    log.info("Model saved → %s", model_path)

    metrics_path = out / "xgb_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost for XAI-NIDS")
    p.add_argument("--data",  default="data/processed/train_balanced.csv")
    p.add_argument("--test",  default="data/processed/test.csv")
    p.add_argument("--out",   default="models/")
    p.add_argument("--label", default="label_encoded")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X_train, y_train, X_test, y_test, feature_cols = load_split(
        args.data, args.test, label_col=args.label
    )
    n_classes = len(np.unique(y_train))
    model = train(X_train, y_train, n_classes)
    metrics = evaluate(model, X_test, y_test)
    save(model, metrics, args.out)
    log.info("Done.")


if __name__ == "__main__":
    main()
