"""Evaluation metric utilities for XAI-NIDS.

Computes Detection Rate, False Alarm Rate, Precision, Recall,
Macro F1, and ROC-AUC for each model and attack class.

Usage (CLI):
    python src/utils/metrics.py \\
        --model  models/random_forest.pkl \\
        --test   data/processed/test.csv \\
        --out    models/rf_metrics.json
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true:       Ground-truth integer labels.
        y_pred:       Predicted integer labels.
        y_prob:       Predicted probability matrix, shape (n_samples, n_classes).
        class_names:  Human-readable class names (optional).

    Returns:
        Dictionary with accuracy, macro F1, per-class DR/FAR/FPR, macro ROC-AUC.
    """
    n_classes = y_prob.shape[1]
    classes = list(range(n_classes))

    # ----- Overall metrics -----
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ----- Confusion-matrix-derived per-class metrics -----
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)

    dr_per_class = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)  # Detection Rate / Recall
    fpr_per_class = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)  # False Positive Rate
    prec_per_class = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)

    mean_dr = float(np.mean(dr_per_class))
    mean_fpr = float(np.mean(fpr_per_class))
    mean_prec = float(np.mean(prec_per_class))

    # ----- ROC-AUC (macro OvR) -----
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1:  # binary case
        y_bin = np.hstack([1 - y_bin, y_bin])
    try:
        macro_auc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        macro_auc = None
        log.warning("Could not compute ROC-AUC (likely missing class in test set)")

    # ----- Per-class breakdown -----
    per_class = []
    for i, cls in enumerate(classes):
        name = class_names[i] if class_names and i < len(class_names) else str(cls)
        per_class.append({
            "class": name,
            "detection_rate":    round(float(dr_per_class[i]),  4),
            "false_alarm_rate":  round(float(fpr_per_class[i]), 4),
            "precision":         round(float(prec_per_class[i]), 4),
            "support":           int((y_true == i).sum()),
        })

    result = {
        "accuracy":          round(float(acc), 6),
        "macro_f1":          round(float(macro_f1), 6),
        "mean_detection_rate":   round(mean_dr,  6),
        "mean_false_alarm_rate": round(mean_fpr, 6),
        "mean_precision":        round(mean_prec, 6),
        "macro_roc_auc":     round(float(macro_auc), 6) if macro_auc else None,
        "per_class":         per_class,
    }

    log.info("Accuracy       : %.4f", acc)
    log.info("Macro F1       : %.4f", macro_f1)
    log.info("Mean DR        : %.4f", mean_dr)
    log.info("Mean FAR       : %.4f", mean_fpr)
    if macro_auc:
        log.info("Macro ROC-AUC  : %.4f", macro_auc)

    return result


def compare_models(models_dir: str | Path, test_path: str | Path, label_col: str) -> dict:
    """Load all three serialised models and compare metrics on the same test set."""
    models_dir = Path(models_dir)
    test = pd.read_csv(test_path)
    feature_cols = [c for c in test.columns if c != label_col]
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[label_col].values.astype(int)

    results = {}
    for name, filename in [("random_forest", "random_forest.pkl"), ("xgboost", "xgboost_model.pkl")]:
        path = models_dir / filename
        if not path.exists():
            log.warning("%s not found — skipping", path)
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        results[name] = metrics
        log.info("--- %s ---", name.upper())

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate XAI-NIDS model metrics")
    p.add_argument("--model",  required=True, help="Path to .pkl model file")
    p.add_argument("--test",   default="data/processed/test.csv")
    p.add_argument("--out",    default="models/metrics.json")
    p.add_argument("--label",  default="label_encoded")
    p.add_argument("--labels", default=None,
                   help="Path to label_map.json for human-readable class names")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = joblib.load(args.model)
    test = pd.read_csv(args.test)
    feature_cols = [c for c in test.columns if c != args.label]
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[args.label].values.astype(int)

    class_names = None
    if args.labels:
        with open(args.labels) as f:
            label_map = json.load(f)
        class_names = [label_map[str(i)] for i in range(len(label_map))]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob, class_names)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", out_path)


if __name__ == "__main__":
    main()
