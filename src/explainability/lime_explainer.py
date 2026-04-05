"""LIME explainer wrapper for Random Forest, XGBoost, and LSTM.

Provides local, per-prediction explanations using the LIME
(Local Interpretable Model-agnostic Explanations) framework.
Complements the SHAP explainer for cross-validated XAI coverage.

Key difference vs SHAP:
  - SHAP  : global + local, model-aware (TreeExplainer / DeepExplainer)
  - LIME  : local only, fully model-agnostic — works on any black-box

Usage (as a module)::

    from src.explainability.lime_explainer import LimeExplainer

    lime = LimeExplainer(
        feature_names=feature_cols,
        class_names=class_names,
        random_state=42,
    )
    lime.fit(X_train_background)          # store background for sampling
    exp = lime.explain_single(model, x_single, predicted_class=2)
    weights = lime.get_weights(exp, predicted_class=2)
    fig = lime.plot_weights(weights, prediction_label="DDoS", confidence=0.97)
    lime.save_explanation(exp, predicted_class=2, out_path="models/lime_alert.json")

Usage (CLI)::

    python src/explainability/lime_explainer.py \\
        --model  models/random_forest.pkl \\
        --data   data/samples/sample_100rows.csv \\
        --bg     data/samples/sample_100rows.csv \\
        --index  0 \\
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
matplotlib.use("Agg")  # non-interactive backend — safe for Streamlit + Docker
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

# Colour palette — consistent with waterfall.py
_RED  = "#d7191c"   # positive weight  -> pushes toward predicted class
_BLUE = "#2c7bb6"   # negative weight  -> pushes away from predicted class
_GREY = "#888888"


# ---------------------------------------------------------------------------
# Helper: build a predict_proba callable for any supported model type
# ---------------------------------------------------------------------------

def _make_predict_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba function compatible with LIME.

    Supports:
      - sklearn-style models with ``.predict_proba()``
      - Keras / TensorFlow models with ``.predict()``
      - Any callable that accepts a 2-D array and returns class probabilities

    Args:
        model: Trained model object.

    Returns:
        A callable ``fn(X) -> np.ndarray`` of shape (n_samples, n_classes).
    """
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        # Keras / TF — LSTM expects (samples, time_steps, features)
        def _keras_predict(X: np.ndarray) -> np.ndarray:
            if X.ndim == 2:
                X = np.expand_dims(X, axis=1)   # (n, 1, features)
            return model.predict(X, verbose=0)
        log.debug("Using Keras predict wrapper")
        return _keras_predict

    if hasattr(model, "predict_proba"):
        log.debug("Using predict_proba wrapper")
        return lambda X: model.predict_proba(X.astype(np.float32))

    raise TypeError(
        f"Model type {type(model).__name__} has neither predict_proba "
        "nor predict.  Wrap it manually before passing to LimeExplainer."
    )


def make_keras_predict_fn(
    model: Any,
    time_steps: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba-compatible function for a Keras LSTM model.

    LIME passes 2-D arrays (n_perturbed, n_features) to the predict function.
    This wrapper reshapes each row into (n, time_steps, n_features) — the
    format expected by the LSTM — by repeating features across time steps.

    Args:
        model:      Compiled Keras model with ``predict`` method.
        time_steps: Number of time-steps the LSTM was trained on (default 5).

    Returns:
        Callable ``fn(X) -> np.ndarray`` returning probabilities
        of shape (n_samples, n_classes).
    """
    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_seq = np.stack([X] * time_steps, axis=1)   # (n, T, features)
        return model.predict(X_seq, verbose=0)
    return _predict


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class LimeExplainer:
    """Thin wrapper around ``lime.lime_tabular.LimeTabularExplainer``.

    Provides a consistent API mirroring ``src.explainability.shap_explainer``
    so both SHAP and LIME can be swapped in the dashboard and SOC triage
    pipeline without changing calling code.

    Args:
        feature_names:  List of feature column names (length == n_features).
        class_names:    List of class label strings (length == n_classes).
        mode:           ``'classification'`` (default) or ``'regression'``.
        kernel_width:   LIME kernel width for locality weighting.
                        ``None`` -> LIME default (sqrt(n_features) * 0.75).
        n_samples:      Number of perturbed neighbourhood samples per explanation.
        discretize:     Whether to discretize continuous features for LIME.
        random_state:   Seed for full reproducibility.
    """

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

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit(self, background_data: np.ndarray) -> "LimeExplainer":
        """Initialise the underlying LimeTabularExplainer.

        Must be called before ``explain_single`` or ``explain_batch``.

        Args:
            background_data: Training data matrix, shape (n_samples, n_features).
                             Used by LIME to learn feature statistics for
                             neighbourhood sampling.

        Returns:
            self  (fluent interface)
        """
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

    # ------------------------------------------------------------------
    # Explanation API
    # ------------------------------------------------------------------

    def explain_single(
        self,
        model: Any,
        x: np.ndarray,
        predicted_class: int,
        n_samples: int | None = None,
    ):
        """Explain a single network flow (for SOC per-alert triage).

        Args:
            model:           Trained sklearn / XGBoost / Keras model.
            x:               1-D feature vector, shape (n_features,).
            predicted_class: Class index to explain (usually the model's prediction).
            n_samples:       Override default n_samples for this call.

        Returns:
            ``lime.explanation.Explanation`` object.
        """
        self._assert_fitted()
        x = np.asarray(x, dtype=np.float32).ravel()
        predict_fn = _make_predict_fn(model)
        ns = n_samples or self.n_samples

        log.info(
            "LIME explaining single sample — predicted_class=%d  n_samples=%d",
            predicted_class, ns,
        )
        return self._explainer.explain_instance(
            data_row=x,
            predict_fn=predict_fn,
            labels=(predicted_class,),
            num_features=len(self.feature_names),
            num_samples=ns,
            top_labels=None,
        )

    def explain_batch(
        self,
        model: Any,
        X: np.ndarray,
        predicted_classes: np.ndarray,
        n_samples: int | None = None,
    ) -> list:
        """Explain a batch of flows (e.g., all alerts in a detection window).

        Warning: LIME is inherently per-sample; this is a sequential loop.
        For large batches consider using SHAP (faster for tree models).

        Args:
            model:             Trained model.
            X:                 Feature matrix, shape (n_flows, n_features).
            predicted_classes: 1-D array of predicted class indices.
            n_samples:         Override default n_samples per explanation.

        Returns:
            List of ``lime.explanation.Explanation`` objects, one per flow.
        """
        self._assert_fitted()
        explanations = []
        for i, (x, cls) in enumerate(zip(X, predicted_classes)):
            log.info("LIME batch: explaining sample %d / %d", i + 1, len(X))
            exp = self.explain_single(model, x, int(cls), n_samples=n_samples)
            explanations.append(exp)
        return explanations

    # ------------------------------------------------------------------
    # Weight extraction
    # ------------------------------------------------------------------

    def get_weights(
        self,
        explanation,
        predicted_class: int,
        top_n: int | None = None,
    ) -> dict[str, float]:
        """Extract feature weights from a LIME explanation as a sorted dict.

        Args:
            explanation:     Output of ``explain_single`` / ``explain_batch``.
            predicted_class: Class index used during explanation.
            top_n:           Return only the top-N features by absolute weight.
                             ``None`` -> return all features.

        Returns:
            Dict of ``{feature_name: weight}`` sorted by descending |weight|.
        """
        raw = dict(explanation.as_list(label=predicted_class))
        sorted_weights = dict(
            sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )
        if top_n is not None:
            sorted_weights = dict(list(sorted_weights.items())[:top_n])
        return sorted_weights

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_weights(
        self,
        weights: dict[str, float],
        prediction_label: str = "Attack",
        confidence: float | None = None,
        top_n: int = 10,
        figsize: tuple[float, float] = (9, 5),
    ) -> plt.Figure:
        """Render a horizontal bar chart of LIME feature weights.

        Visual style is consistent with ``waterfall.plot_waterfall`` so both
        charts can appear side-by-side in the dashboard.

        Args:
            weights:          ``{feature_name: weight}`` dict from ``get_weights``.
            prediction_label: Human-readable predicted class name.
            confidence:       Model confidence (0-1) shown in the title.
            top_n:            Maximum number of features to display.
            figsize:          Matplotlib figure size.

        Returns:
            ``matplotlib.figure.Figure`` object.
        """
        items   = list(weights.items())[:top_n]
        names   = [k for k, _ in items]
        vals    = np.array([v for _, v in items], dtype=float)

        # Sort ascending for waterfall-style ordering
        order   = np.argsort(vals)
        names   = [names[i] for i in order]
        vals    = vals[order]
        colours = [_RED if v > 0 else _BLUE for v in vals]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(
            names, vals,
            color=colours,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )

        for bar, val in zip(bars, vals):
            label_x = val + 0.003 if val >= 0 else val - 0.003
            ha = "left" if val >= 0 else "right"
            ax.text(
                label_x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center", ha=ha,
                fontsize=8, color="#333333",
            )

        ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--", alpha=0.7)

        conf_str = f"  |  Confidence: {confidence * 100:.1f}%" if confidence is not None else ""
        ax.set_title(
            f"LIME Explanation — Prediction: {prediction_label}{conf_str}\n"
            f"Top {len(items)} contributing features",
            fontsize=10, pad=12,
        )
        ax.set_xlabel("LIME Weight (local linear approximation)", fontsize=9)

        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=_RED,  label="-> Pushes toward predicted class"),
                Patch(facecolor=_BLUE, label="<- Pushes away from predicted class"),
            ],
            loc="lower right", fontsize=8,
        )

        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        plt.tight_layout()

        log.info("LIME weight chart rendered for prediction='%s'", prediction_label)
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_explanation(
        self,
        explanation,
        predicted_class: int,
        out_path: str | Path,
        top_n: int | None = None,
        extra_meta: dict | None = None,
    ) -> Path:
        """Persist a LIME explanation to a JSON file.

        The output schema mirrors the SHAP waterfall JSON format so both
        explanation types can be rendered uniformly in the dashboard.

        Args:
            explanation:     Output of ``explain_single``.
            predicted_class: Class index that was explained.
            out_path:        Destination ``.json`` path.
            top_n:           Limit saved features to top-N by |weight|.
            extra_meta:      Optional dict merged into the JSON root
                             (e.g., ``{"flow_id": "...", "timestamp": "..."}``)

        Returns:
            Resolved ``Path`` of the saved file.
        """
        weights = self.get_weights(explanation, predicted_class, top_n=top_n)
        payload: dict = {
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
        log.info("LIME explanation saved -> %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if self._explainer is None:
            raise RuntimeError(
                "LimeExplainer has not been fitted.  "
                "Call .fit(background_data) before explaining."
            )


# ---------------------------------------------------------------------------
# Module-level convenience functions  (mirrors shap_explainer public API)
# ---------------------------------------------------------------------------

def explain(
    model: Any,
    X: np.ndarray,
    predicted_classes: np.ndarray,
    background_data: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    n_samples: int = 1000,
    random_state: int = 42,
) -> list:
    """Explain a batch of predictions (module-level convenience function).

    Creates and fits a ``LimeExplainer`` internally.  For repeated calls
    on the same dataset use the class directly to avoid re-fitting.

    Args:
        model:             Trained sklearn / XGBoost / Keras model.
        X:                 Feature matrix, shape (n_rows, n_features).
        predicted_classes: Predicted class indices, shape (n_rows,).
        background_data:   Training data for neighbourhood sampling.
        feature_names:     Feature column names.
        class_names:       Class label strings.
        n_samples:         LIME neighbourhood size per explanation.
        random_state:      RNG seed.

    Returns:
        List of ``lime.explanation.Explanation`` objects.
    """
    exp = LimeExplainer(
        feature_names=feature_names,
        class_names=class_names,
        n_samples=n_samples,
        random_state=random_state,
    ).fit(background_data)
    return exp.explain_batch(model, X, predicted_classes, n_samples=n_samples)


def explain_single(
    model: Any,
    x: np.ndarray,
    predicted_class: int,
    background_data: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    n_samples: int = 1000,
    random_state: int = 42,
):
    """Explain a single flow (module-level convenience function).

    Args:
        model:           Trained sklearn / XGBoost / Keras model.
        x:               1-D feature vector, shape (n_features,).
        predicted_class: Class index to explain.
        background_data: Training data for neighbourhood sampling.
        feature_names:   Feature column names.
        class_names:     Class label strings.
        n_samples:       LIME neighbourhood size.
        random_state:    RNG seed.

    Returns:
        ``lime.explanation.Explanation`` object.
    """
    exp = LimeExplainer(
        feature_names=feature_names,
        class_names=class_names,
        n_samples=n_samples,
        random_state=random_state,
    ).fit(background_data)
    return exp.explain_single(model, x, predicted_class, n_samples=n_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute LIME explanation for a single network flow (XAI-NIDS)"
    )
    p.add_argument("--model",  required=True, help="Path to .pkl model file")
    p.add_argument("--data",   required=True, help="CSV of flows to explain")
    p.add_argument("--bg",     required=True, help="Background CSV for neighbourhood sampling")
    p.add_argument("--index",  type=int, default=0,    help="Row index in --data to explain")
    p.add_argument("--label",  default="label_encoded", help="Label column name")
    p.add_argument("--n",      type=int, default=1000,  help="LIME n_samples")
    p.add_argument("--top",    type=int, default=10,    help="Top-N features to save")
    p.add_argument("--out",    default="models/lime_explanation.json", help="Output JSON path")
    p.add_argument("--plot",   default=None, help="Optional: save weight chart to this PNG path")
    p.add_argument("--labels", default=None, help="JSON file mapping int -> class name")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("Loading model from %s", args.model)
    model = joblib.load(args.model)

    df  = pd.read_csv(args.data)
    bg_df = pd.read_csv(args.bg)
    feature_cols = [c for c in df.columns if c != args.label]

    X    = df[feature_cols].values.astype(np.float32)
    X_bg = bg_df[feature_cols].values.astype(np.float32)

    # Load class names from label_map.json if provided
    if args.labels and Path(args.labels).exists():
        with open(args.labels) as fh:
            label_map = json.load(fh)
        class_names = [label_map[str(i)] for i in range(len(label_map))]
    else:
        n_cls = len(np.unique(getattr(model, "classes_", np.arange(15))))
        class_names = [f"Class_{i}" for i in range(n_cls)]

    # Predict to identify which class to explain
    predict_fn  = _make_predict_fn(model)
    proba       = predict_fn(X[args.index : args.index + 1])[0]
    pred_class  = int(np.argmax(proba))
    confidence  = float(proba[pred_class])
    pred_label  = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    log.info("Row %d -> predicted=%s  confidence=%.3f", args.index, pred_label, confidence)

    lime_exp = LimeExplainer(
        feature_names=feature_cols,
        class_names=class_names,
        n_samples=args.n,
        random_state=42,
    ).fit(X_bg)

    exp = lime_exp.explain_single(model, X[args.index], pred_class)

    lime_exp.save_explanation(
        exp, pred_class,
        out_path=args.out,
        top_n=args.top,
        extra_meta={"source_row_index": args.index, "confidence": round(confidence, 6)},
    )

    if args.plot:
        weights = lime_exp.get_weights(exp, pred_class, top_n=args.top)
        fig = lime_exp.plot_weights(weights, prediction_label=pred_label, confidence=confidence)
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Weight chart saved -> %s", plot_path)


if __name__ == "__main__":
    main()
