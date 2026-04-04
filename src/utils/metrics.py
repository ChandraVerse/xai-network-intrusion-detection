"""Evaluation metric utilities for XAI-NIDS.

Computes Detection Rate, False Alarm Rate, Precision, Recall, F1,
and a per-class metric breakdown table. Designed for multi-class
network intrusion classification (14 CICIDS-2017 classes).

Usage:
    from src.utils.metrics import compute_metrics, print_metrics_table

    metrics = compute_metrics(y_true, y_pred, label_names)
    print_metrics_table(metrics)
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core compute function
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict:
    """Compute a comprehensive metric dictionary for multi-class classification.

    Args:
        y_true: Ground-truth integer labels, shape (n_samples,).
        y_pred: Predicted integer labels, shape (n_samples,).
        label_names: Optional list of class name strings for the report.

    Returns:
        dict with keys: accuracy, macro_f1, macro_precision, macro_recall,
        mean_fpr, mean_dr, per_class (DataFrame), confusion_matrix (ndarray).
    """
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    class_labels = list(range(n_classes))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    # Per-class metrics
    dr = tp / (tp + fn + 1e-9)           # Detection Rate == Recall
    fpr = fp / (fp + tn + 1e-9)          # False Positive Rate
    prec = tp / (tp + fp + 1e-9)         # Precision
    f1 = 2 * prec * dr / (prec + dr + 1e-9)

    if label_names is None:
        label_names = [str(i) for i in class_labels]

    per_class_df = pd.DataFrame(
        {
            "class": label_names[:n_classes],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "DR (Recall)": np.round(dr, 6),
            "FPR": np.round(fpr, 6),
            "Precision": np.round(prec, 6),
            "F1": np.round(f1, 6),
        }
    )

    metrics = {
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "macro_precision": round(float(macro_prec), 6),
        "macro_recall": round(float(macro_rec), 6),
        "mean_fpr": round(float(np.mean(fpr)), 6),
        "mean_dr": round(float(np.mean(dr)), 6),
        "n_samples": int(len(y_true)),
        "n_classes": n_classes,
        "per_class": per_class_df,
        "confusion_matrix": cm,
    }
    return metrics


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict, title: str = "") -> None:
    """Pretty-print the per-class metric table to stdout."""
    if title:
        print(f"\n{'=' * 70}")
        print(f" {title}")
        print(f"{'=' * 70}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
    print(f"  Macro Prec.  : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall : {metrics['macro_recall']:.4f}")
    print(f"  Mean FPR     : {metrics['mean_fpr']:.4f}")
    print(f"  Mean DR      : {metrics['mean_dr']:.4f}")
    print(f"  Samples      : {metrics['n_samples']}")
    print()
    print(metrics["per_class"].to_string(index=False))
    print()


def classification_report_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> str:
    """Return sklearn classification_report as a string."""
    return classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
    )


def format_metrics_for_dashboard(metrics: dict) -> dict:
    """Return a flat dict of scalar metrics suitable for Streamlit display."""
    return {
        "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
        "Macro F1": f"{metrics['macro_f1']:.4f}",
        "Macro Precision": f"{metrics['macro_precision']:.4f}",
        "Macro Recall": f"{metrics['macro_recall']:.4f}",
        "Mean FPR": f"{metrics['mean_fpr']:.4f}",
        "Mean DR": f"{metrics['mean_dr']:.4f}",
        "Samples Evaluated": str(metrics["n_samples"]),
        "Classes": str(metrics["n_classes"]),
    }
