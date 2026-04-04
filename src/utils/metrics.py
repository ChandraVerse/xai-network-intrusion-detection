"""
metrics.py
----------
Detection Rate, False Alarm Rate, Precision, Recall, Macro F1,
and inference latency — the metrics that matter in a real SOC deployment.

Usage:
    python src/utils/metrics.py \
        --models models/ \
        --test   data/processed/test.csv
"""
import argparse
import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Compute a comprehensive SOC-relevant metrics dictionary.

    Parameters
    ----------
    y_true : ground-truth integer labels
    y_pred : predicted integer labels
    y_prob : predicted probability matrix (samples × classes) — optional

    Returns dict with keys:
        accuracy, macro_f1, precision, recall,
        detection_rate, false_alarm_rate, roc_auc (if y_prob provided)
    """
    cm   = confusion_matrix(y_true, y_pred)
    tn   = cm[0, 0]
    fp   = cm[0, 1:].sum()
    fn   = cm[1:, 0].sum()
    tp   = cm[1:, 1:].sum()

    dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Detection Rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Alarm Rate

    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "macro_f1":         f1_score(y_true, y_pred, average="macro",     zero_division=0),
        "precision":        precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":           recall_score(y_true, y_pred, average="macro",    zero_division=0),
        "detection_rate":   dr,
        "false_alarm_rate": far,
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def print_metrics_table(results: dict) -> None:
    """Pretty-print a dict of {model_name: metrics_dict}."""
    header = f"{'Model':<20}  {'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}  {'DR':>7}  {'FAR':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<20}  "
            f"{m['accuracy']*100:>6.2f}%  "
            f"{m['macro_f1']:>7.4f}  "
            f"{m['precision']:>7.4f}  "
            f"{m['recall']:>7.4f}  "
            f"{m['detection_rate']*100:>6.2f}%  "
            f"{m['false_alarm_rate']*100:>6.3f}%"
        )
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved models on test set")
    parser.add_argument("--models", required=True, help="Directory containing .pkl models")
    parser.add_argument("--test",   required=True, help="Path to test.csv")
    parser.add_argument("--label",  default="Label")
    args = parser.parse_args()

    df    = pd.read_csv(args.test)
    X     = df.drop(columns=[args.label]).values
    y     = df[args.label].values

    model_files = [f for f in os.listdir(args.models) if f.endswith(".pkl")]
    results = {}

    for mf in model_files:
        path  = os.path.join(args.models, mf)
        model = joblib.load(path)
        t0    = time.time()
        y_pred = model.predict(X)
        elapsed = (time.time() - t0) / len(X) * 1000  # ms per sample
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        m = compute_metrics(y, y_pred, y_prob)
        m["inference_ms"] = elapsed
        results[mf.replace(".pkl", "")] = m
        print(f"  [{mf}] avg inference: {elapsed:.3f} ms/sample")

    print_metrics_table(results)
