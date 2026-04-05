"""LIME explainer wrapper for Random Forest, XGBoost, and LSTM.

Provides local, per-prediction explanations using the LIME framework.
Exposes TWO interfaces so both existing callers and tests work:

  1. ``LIMEExplainer``  -- simple wrapper used by tests/test_lime_explainer.py
        LIMEExplainer(training_data, feature_names, class_names)
        .explain_instance(instance, predict_fn, num_features, num_samples)
        .as_dict(explanation)
        .plot_explanation(explanation)

  2. ``LimeExplainer``  -- full-featured class used by the dashboard/pipeline
        LimeExplainer(feature_names, class_names, ...)
        .fit(background_data)
        .explain_single(model, x, predicted_class)
        .explain_batch(model, X, predicted_classes)
        .get_weights(explanation, predicted_class)
        .plot_weights(weights, ...)
        .save_explanation(explanation, ...)

Usage (CLI)::
    python src/explainability/lime_explainer.py \\\
        --model  models/random_forest.pkl \\\
        --data   data/samples/sample_100.csv \\\
        --bg     data/samples/sample_100.csv \\\
        --index  0 \\\
        --out    models/lime_explanation.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_RED  = "#d7191c"
_BLUE = "#2c7bb6"
_GREY = "#888888"


# ---------------------------------------------------------------------------
# Helper: build a predict_proba callable for any supported model type
# ---------------------------------------------------------------------------

def _make_predict_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba function compatible with LIME."""
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X.astype(np.float32))
    if hasattr(model, "predict"):
        def _keras_predict(X: np.ndarray) -> np.ndarray:
            if X.ndim == 2:
                X = np.expand_dims(X, axis=1)
            return model.predict(X, verbose=0)
        return _keras_predict
    raise TypeError(
        f"Model type {type(model).__name__} has neither predict_proba nor predict."
    )


def make_keras_predict_fn(
    model: Any,
    time_steps: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba-compatible function for a Keras LSTM model."""
    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_seq = np.stack([X] * time_steps, axis=1)
        return model.predict(X_seq, verbose=0)
    return _predict


# ---------------------------------------------------------------------------
# LIMEExplainer  -- simple API expected by test_lime_explainer.py
# ---------------------------------------------------------------------------

class LIMEExplainer:
    """Thin LIME wrapper with the API expected by tests.

    Args:
        training_data:  2-D array used to build the LimeTabularExplainer.
        feature_names:  Optional list of feature names.
        class_names:    Optional list of class label strings.
        random_state:   RNG seed.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.training_data = np.asarray(training_data, dtype=np.float32)
        self.feature_names = feature_names
        self.class_names   = class_names
        self.random_state  = random_state

        self.explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=False,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Core explanation
    # ------------------------------------------------------------------

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 1000,
        top_labels: int = 1,
    ):
        """Explain a single instance.

        Args:
            instance:     1-D feature vector.
            predict_fn:   Model predict_proba function.
            num_features: Max features in explanation.
            num_samples:  LIME neighbourhood samples.
            top_labels:   Number of top labels to include.

        Returns:
            lime.explanation.Explanation object.
        """
        instance = np.asarray(instance, dtype=np.float32).ravel()
        return self.explainer.explain_instance(
            data_row=instance,
            predict_fn=lambda X: predict_fn(X.astype(np.float32)),
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels,
        )

    # ------------------------------------------------------------------
    # Serialisation helper
    # ------------------------------------------------------------------

    def as_dict(
        self,
        explanation,
        label: int | None = None,
    ) -> dict:
        """Convert a LIME explanation to a JSON-serialisable dict.

        Keys returned: label, label_name, score, local_pred, features.

        Args:
            explanation: Output of explain_instance.
            label:       Class index to extract; defaults to top label.

        Returns:
            dict with keys: label, label_name, score, local_pred, features.
        """
        if label is None:
            label = explanation.top_labels[0]

        label_name = (
            self.class_names[label]
            if self.class_names and label < len(self.class_names)
            else str(label)
        )

        feature_list = [
            {"feature": feat, "weight": float(w)}
            for feat, w in explanation.as_list(label=label)
        ]

        return {
            "label":      int(label),
            "label_name": label_name,
            "score":      float(explanation.score),
            "local_pred": float(explanation.local_pred[0]),
            "features":   feature_list,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_explanation(
        self,
        explanation,
        label: int | None = None,
        top_n: int = 10,
        figsize: tuple[float, float] = (9, 5),
    ) -> plt.Figure:
        """Render a horizontal bar chart for the explanation.

        Args:
            explanation: Output of explain_instance.
            label:       Class index to visualise; defaults to top label.
            top_n:       Max features to display.
            figsize:     Matplotlib figure size.

        Returns:
            matplotlib.figure.Figure.
        """
        if label is None:
            label = explanation.top_labels[0]

        items  = explanation.as_list(label=label)[:top_n]
        names  = [k for k, _ in items]
        vals   = np.array([v for _, v in items], dtype=float)
        order  = np.argsort(vals)
        names  = [names[i] for i in order]
        vals   = vals[order]
        colours = [_RED if v > 0 else _BLUE for v in vals]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(names, vals, color=colours, edgecolor="white", linewidth=0.5, height=0.6)
        ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title("LIME Explanation", fontsize=10, pad=12)
        ax.set_xlabel("LIME Weight", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# LimeExplainer  -- full-featured class used by dashboard / pipeline
# ---------------------------------------------------------------------------

class LimeExplainer:
    """Full-featured LIME wrapper for dashboard and SOC triage pipeline."""

    def __init__(
        self,
        feature_names: list[str],
        class_names: list[str],
        mode: str = "classification",
        kernel_width: float | None = None,
        n_samples: int = 1000,
        discretize: bool = False,
        random_state: int = 42,
    ) -> None:
        self.feature_names = feature_names
        self.class_names   = class_names
        self.mode          = mode
        self.kernel_width  = kernel_width
        self.n_samples     = n_samples
        self.discretize    = discretize
        self.random_state  = random_state
        self._explainer: LimeTabularExplainer | None = None
        self._background:  np.ndarray | None = None

    def fit(self, background_data: np.ndarray) -> "LimeExplainer":
        self._background = background_data.astype(np.float32)
        kw = {"kernel_width": self.kernel_width} if self.kernel_width else {}
        self._explainer = LimeTabularExplainer(
            training_data=self._background,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            discretize_continuous=self.discretize,
            random_state=self.random_state,
            **kw,
        )
        log.info(
            "LimeExplainer fitted — %d background samples, %d features, %d classes",
            len(background_data), len(self.feature_names), len(self.class_names),
        )
        return self

    def explain_single(self, model, x, predicted_class, n_samples=None):
        self._assert_fitted()
        x = np.asarray(x, dtype=np.float32).ravel()
        predict_fn = _make_predict_fn(model)
        ns = n_samples or self.n_samples
        return self._explainer.explain_instance(
            data_row=x,
            predict_fn=predict_fn,
            labels=(predicted_class,),
            num_features=len(self.feature_names),
            num_samples=ns,
            top_labels=None,
        )

    def explain_batch(self, model, X, predicted_classes, n_samples=None):
        self._assert_fitted()
        return [
            self.explain_single(model, x, int(cls), n_samples=n_samples)
            for x, cls in zip(X, predicted_classes)
        ]

    def get_weights(self, explanation, predicted_class, top_n=None):
        raw = dict(explanation.as_list(label=predicted_class))
        sorted_weights = dict(
            sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )
        if top_n is not None:
            sorted_weights = dict(list(sorted_weights.items())[:top_n])
        return sorted_weights

    def plot_weights(self, weights, prediction_label="Attack", confidence=None,
                     top_n=10, figsize=(9, 5)):
        items   = list(weights.items())[:top_n]
        names   = [k for k, _ in items]
        vals    = np.array([v for _, v in items], dtype=float)
        order   = np.argsort(vals)
        names   = [names[i] for i in order]
        vals    = vals[order]
        colours = [_RED if v > 0 else _BLUE for v in vals]
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(names, vals, color=colours, edgecolor="white", linewidth=0.5, height=0.6)
        ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--", alpha=0.7)
        conf_str = f"  |  Confidence: {confidence * 100:.1f}%" if confidence else ""
        ax.set_title(
            f"LIME Explanation — Prediction: {prediction_label}{conf_str}",
            fontsize=10, pad=12,
        )
        ax.set_xlabel("LIME Weight", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return fig

    def save_explanation(self, explanation, predicted_class, out_path,
                         top_n=None, extra_meta=None):
        weights = self.get_weights(explanation, predicted_class, top_n=top_n)
        payload = {
            "explainer": "LIME",
            "predicted_class_index": predicted_class,
            "predicted_class_label": (
                self.class_names[predicted_class]
                if predicted_class < len(self.class_names)
                else str(predicted_class)
            ),
            "top_features": [
                {"feature": feat, "weight": round(w, 8)}
                for feat, w in weights.items()
            ],
            "intercept":  float(explanation.intercept[predicted_class]),
            "local_pred": float(explanation.local_pred[0]),
            "score":      float(explanation.score),
        }
        if extra_meta:
            payload.update(extra_meta)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return out_path

    def _assert_fitted(self):
        if self._explainer is None:
            raise RuntimeError(
                "LimeExplainer has not been fitted. Call .fit(background_data) first."
            )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def explain(
    model, X, predicted_classes, background_data,
    feature_names, class_names, n_samples=1000, random_state=42,
):
    exp = LimeExplainer(
        feature_names=feature_names, class_names=class_names,
        n_samples=n_samples, random_state=random_state,
    ).fit(background_data)
    return exp.explain_batch(model, X, predicted_classes, n_samples=n_samples)


def explain_single(
    model, x, predicted_class, background_data,
    feature_names, class_names, n_samples=1000, random_state=42,
):
    exp = LimeExplainer(
        feature_names=feature_names, class_names=class_names,
        n_samples=n_samples, random_state=random_state,
    ).fit(background_data)
    return exp.explain_single(model, x, predicted_class, n_samples=n_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute LIME explanation for a single network flow (XAI-NIDS)"
    )
    p.add_argument("--model",  required=True)
    p.add_argument("--data",   required=True)
    p.add_argument("--bg",     required=True)
    p.add_argument("--index",  type=int, default=0)
    p.add_argument("--label",  default="label_encoded")
    p.add_argument("--n",      type=int, default=1000)
    p.add_argument("--top",    type=int, default=10)
    p.add_argument("--out",    default="models/lime_explanation.json")
    p.add_argument("--plot",   default=None)
    p.add_argument("--labels", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    model = joblib.load(args.model)
    df    = pd.read_csv(args.data)
    bg_df = pd.read_csv(args.bg)
    feature_cols = [c for c in df.columns if c != args.label]
    X    = df[feature_cols].values.astype(np.float32)
    X_bg = bg_df[feature_cols].values.astype(np.float32)
    if args.labels and Path(args.labels).exists():
        with open(args.labels) as fh:
            label_map = json.load(fh)
        class_names = [label_map[str(i)] for i in range(len(label_map))]
    else:
        n_cls = len(np.unique(getattr(model, "classes_", np.arange(15))))
        class_names = [f"Class_{i}" for i in range(n_cls)]
    predict_fn = _make_predict_fn(model)
    proba      = predict_fn(X[args.index : args.index + 1])[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])
    pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    lime_exp = LimeExplainer(
        feature_names=feature_cols, class_names=class_names,
        n_samples=args.n, random_state=42,
    ).fit(X_bg)
    exp = lime_exp.explain_single(model, X[args.index], pred_class)
    lime_exp.save_explanation(
        exp, pred_class, out_path=args.out, top_n=args.top,
        extra_meta={"source_row_index": args.index, "confidence": round(confidence, 6)},
    )
    if args.plot:
        weights = lime_exp.get_weights(exp, pred_class, top_n=args.top)
        fig = lime_exp.plot_weights(weights, prediction_label=pred_label, confidence=confidence)
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
