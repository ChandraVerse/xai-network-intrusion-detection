"""Local Interpretable Model-agnostic Explanations (LIME) for XAI-NIDS.

Wraps ``lime.lime_tabular.LimeTabularExplainer`` to provide consistent
per-flow explanations that complement the SHAP module.  Works with any
sklearn-compatible predict_proba interface (Random Forest, XGBoost) and
with Keras/TF models via a thin probability wrapper.

Key differences from SHAP in this project
------------------------------------------
* SHAP  → global feature importance + per-flow waterfall (TreeExplainer).
* LIME  → purely local: perturbs a *single* flow around its neighbourhood
          and fits a sparse linear model to explain that one prediction.
  Use LIME when you need a human-readable "top-N rules" explanation for
  a SOC analyst reviewing a single alert.

Usage (library):
    from src.explainability.lime_explainer import LIMEExplainer

    explainer = LIMEExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )
    exp = explainer.explain_instance(
        instance=X_test[0],
        predict_fn=rf_model.predict_proba,
        num_features=10,
    )
    exp.as_pyplot_figure()

Usage (CLI):
    python src/explainability/lime_explainer.py \\
        --model   models/random_forest.pkl \\
        --data    data/samples/sample_100.csv \\
        --index   0 \\
        --out     reports/lime_explanation_0.html
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Sequence

import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive — safe for Streamlit + Docker
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from lime import lime_tabular

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

class LIMEExplainer:
    """Thin wrapper around LimeTabularExplainer for XAI-NIDS.

    Attributes:
        explainer: The underlying ``LimeTabularExplainer`` instance.
        feature_names: Feature column names passed at construction.
        class_names: Attack / BENIGN class labels.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        mode: str = "classification",
        kernel_width: float | None = None,
        discretize_continuous: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialise the LIME tabular explainer.

        Args:
            training_data:        2-D array of training features (float32),
                                  shape (n_samples, n_features).  Used to
                                  infer feature statistics for perturbation.
            feature_names:        Optional list of feature name strings.
            class_names:          Optional list of class label strings.
            mode:                 'classification' (default) or 'regression'.
            kernel_width:         Controls LIME's neighbourhood size.
                                  Defaults to ``sqrt(n_features) * 0.75``.
            discretize_continuous: If True, LIME bins continuous features
                                  before perturbing (recommended for tabular).
            random_state:         Seed for reproducible perturbations.
        """
        if kernel_width is None:
            kernel_width = float(np.sqrt(training_data.shape[1]) * 0.75)

        self.feature_names = list(feature_names) if feature_names else None
        self.class_names   = list(class_names)   if class_names   else None

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data.astype(np.float64),
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            kernel_width=kernel_width,
            discretize_continuous=discretize_continuous,
            random_state=random_state,
            verbose=False,
        )
        log.info(
            "LIMEExplainer initialised — %d training samples, %d features",
            training_data.shape[0],
            training_data.shape[1],
        )

    # -----------------------------------------------------------------------
    # Main explain method
    # -----------------------------------------------------------------------

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 10,
        num_samples: int = 5000,
        top_labels: int = 1,
    ) -> lime_tabular.explanation.Explanation:
        """Compute a LIME explanation for a single network flow.

        Args:
            instance:     1-D feature vector for the flow, shape (n_features,).
            predict_fn:   Function mapping (n, n_features) -> (n, n_classes)
                          probabilities.  For sklearn models pass
                          ``model.predict_proba``; for Keras wrap the model
                          (see ``make_keras_predict_fn``).
            num_features: Maximum number of top features to include in the
                          local linear model.
            num_samples:  Number of neighbourhood samples used to fit the
                          local surrogate.  Higher → more stable, slower.
            top_labels:   Number of top predicted classes to explain.

        Returns:
            ``lime.explanation.Explanation`` object.  Call ``.as_list()`` for
            a ranked list of ``(feature_condition, weight)`` tuples, or
            ``.as_pyplot_figure()`` for an immediate bar chart.
        """
        instance = np.asarray(instance, dtype=np.float64).flatten()
        if instance.shape[0] != (
            len(self.feature_names) if self.feature_names else instance.shape[0]
        ):
            log.warning(
                "Instance length (%d) may not match feature_names length (%s)",
                instance.shape[0],
                len(self.feature_names) if self.feature_names else "unknown",
            )

        log.info(
            "Explaining instance with %d neighbourhood samples, top %d features",
            num_samples,
            num_features,
        )
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=top_labels,
        )
        return exp

    # -----------------------------------------------------------------------
    # Batch convenience
    # -----------------------------------------------------------------------

    def explain_batch(
        self,
        instances: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 10,
        num_samples: int = 3000,
    ) -> list[lime_tabular.explanation.Explanation]:
        """Explain multiple flows.  Returns a list of Explanation objects.

        Warning: LIME is inherently per-sample; this is a sequential loop.
        For large batches consider using SHAP (faster for tree models).

        Args:
            instances:    2-D array, shape (n_flows, n_features).
            predict_fn:   Probability function as in ``explain_instance``.
            num_features: Top features per explanation.
            num_samples:  Neighbourhood samples per explanation.

        Returns:
            List of ``Explanation`` objects, one per flow.
        """
        explanations = []
        for i, row in enumerate(instances):
            exp = self.explain_instance(
                row, predict_fn, num_features=num_features, num_samples=num_samples
            )
            explanations.append(exp)
            if (i + 1) % 10 == 0:
                log.info("Explained %d / %d instances", i + 1, len(instances))
        return explanations

    # -----------------------------------------------------------------------
    # Visualisation helpers
    # -----------------------------------------------------------------------

    def plot_explanation(
        self,
        exp: lime_tabular.explanation.Explanation,
        label: int = 0,
        title: str | None = None,
        figsize: tuple[float, float] = (9, 5),
    ) -> plt.Figure:
        """Render a horizontal bar chart of LIME feature weights.

        Args:
            exp:     Explanation object from ``explain_instance``.
            label:   Class index whose explanation to plot (default: top label).
            title:   Optional figure title.
            figsize: Matplotlib figure size.

        Returns:
            ``matplotlib.figure.Figure`` — caller responsible for
            saving / displaying.
        """
        label = label if label in exp.available_labels() else exp.available_labels()[0]
        features = exp.as_list(label=label)

        names  = [f[0] for f in features]
        values = [f[1] for f in features]
        colours = ["#d7191c" if v > 0 else "#2c7bb6" for v in values]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(names[::-1], values[::-1], color=colours[::-1],
                       edgecolor="white", linewidth=0.5, height=0.6)

        for bar, val in zip(bars, values[::-1]):
            label_x = val + 0.003 if val >= 0 else val - 0.003
            ha = "left" if val >= 0 else "right"
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=8, color="#333333")

        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("LIME Weight (contribution to predicted class)", fontsize=9)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d7191c", label="→ Pushes toward attack"),
            Patch(facecolor="#2c7bb6", label="← Pushes toward benign"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        _title = title or "LIME Local Explanation"
        if self.class_names and label < len(self.class_names):
            _title += f"  |  Class: {self.class_names[label]}"
        ax.set_title(_title, fontsize=10, pad=12)
        plt.tight_layout()
        log.info("LIME bar chart rendered for label=%d", label)
        return fig

    # -----------------------------------------------------------------------
    # JSON / serialisable export
    # -----------------------------------------------------------------------

    def as_dict(
        self,
        exp: lime_tabular.explanation.Explanation,
        label: int | None = None,
    ) -> dict:
        """Return the explanation as a plain dict for logging / API response.

        Args:
            exp:   Explanation object from ``explain_instance``.
            label: Class index.  Defaults to the top predicted class.

        Returns:
            Dict with keys ``label``, ``label_name``, ``score``,
            ``features`` (list of ``{feature, weight}`` dicts).
        """
        label = label if (label is not None and label in exp.available_labels()) \
                      else exp.available_labels()[0]
        label_name = (
            self.class_names[label]
            if self.class_names and label < len(self.class_names)
            else str(label)
        )
        features = [
            {"feature": f, "weight": round(w, 6)}
            for f, w in exp.as_list(label=label)
        ]
        return {
            "label": int(label),
            "label_name": label_name,
            "score": round(float(exp.score), 6),
            "local_pred": round(float(exp.local_pred[label]), 6),
            "features": features,
        }


# ---------------------------------------------------------------------------
# Keras/TF predict wrapper
# ---------------------------------------------------------------------------

def make_keras_predict_fn(
    model,
    time_steps: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_proba-compatible function for a Keras LSTM model.

    LIME passes 2-D arrays (n_perturbed, n_features) to the predict function.
    This wrapper reshapes each row into (1, time_steps, n_features) — the
    format expected by the LSTM — by repeating features across time steps.

    Args:
        model:      Compiled Keras model with ``predict`` method.
        time_steps: Number of time-steps the LSTM was trained on (default 5).

    Returns:
        Callable ``fn(X: np.ndarray) -> np.ndarray`` returning probabilities
        of shape (n_samples, n_classes).
    """
    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # Repeat features across time_steps: (n, features) -> (n, T, features)
        X_seq = np.stack([X] * time_steps, axis=1)
        return model.predict(X_seq, verbose=0)

    return _predict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a LIME explanation for one network flow."
    )
    p.add_argument("--model",    required=True, help="Path to .pkl model file")
    p.add_argument("--data",     required=True, help="CSV of flows")
    p.add_argument("--bg",       default=None,  help="Background CSV (uses --data if omitted)")
    p.add_argument("--label",    default="Label", help="Label column name in CSV")
    p.add_argument("--index",    type=int, default=0, help="Row index to explain")
    p.add_argument("--features", type=int, default=10, help="Top-N features to show")
    p.add_argument("--samples",  type=int, default=5000, help="LIME neighbourhood samples")
    p.add_argument("--out",      default=None, help="Save plot to this path (.png/.html)")
    p.add_argument("--classes",  default=None, nargs="+", help="Class names")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("Loading model from %s", args.model)
    model = joblib.load(args.model)

    df = pd.read_csv(args.data)
    feature_cols = [c for c in df.columns if c != args.label]
    X = df[feature_cols].values.astype(np.float32)

    bg_path = args.bg or args.data
    bg_df = pd.read_csv(bg_path)
    X_bg = bg_df[feature_cols].values.astype(np.float32)

    explainer = LIMEExplainer(
        training_data=X_bg,
        feature_names=feature_cols,
        class_names=args.classes,
    )

    instance = X[args.index]
    exp = explainer.explain_instance(
        instance=instance,
        predict_fn=model.predict_proba,
        num_features=args.features,
        num_samples=args.samples,
    )

    # Print ranked features to stdout
    print(f"\nLIME explanation for row {args.index}:")
    for feat, weight in exp.as_list():
        sign = "+" if weight > 0 else ""
        print(f"  {sign}{weight:+.4f}  {feat}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".html":
            exp.save_to_file(str(out_path))
            log.info("HTML explanation saved → %s", out_path)
        else:
            fig = explainer.plot_explanation(exp)
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Plot saved → %s", out_path)


if __name__ == "__main__":
    main()
