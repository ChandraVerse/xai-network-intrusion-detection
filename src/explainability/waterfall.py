"""Per-alert SHAP waterfall chart renderer.

Produces a ranked waterfall chart showing the top-N features that
contributed most to a single predicted flow.

Usage:
    from src.explainability.waterfall import plot_waterfall

    fig = plot_waterfall(
        shap_vals=shap_vals_1d,
        base_value=base_val,
        feature_names=feature_cols,
        prediction_label="DDoS",
        confidence=0.973,
        top_n=10,
    )
    fig.savefig("waterfall_alert.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for Streamlit + Docker)  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# Colour palette consistent with SHAP defaults
_RED = "#d7191c"    # positive contribution -> pushes toward attack
_BLUE = "#2c7bb6"   # negative contribution -> pushes toward benign
_GREY = "#888888"


def plot_waterfall(
    shap_vals: np.ndarray,
    base_value: float,
    feature_names: list[str],
    prediction_label: str = "Attack",
    confidence: float | None = None,
    top_n: int = 10,
    figsize: tuple[float, float] = (9, 5),
) -> plt.Figure:
    """Render a SHAP waterfall chart for a single prediction.

    Args:
        shap_vals: 1D array of SHAP values, shape (n_features,).
        base_value: Model expected value (SHAP base).
        feature_names: List of feature name strings, len == n_features.
        prediction_label: Human-readable predicted class name.
        confidence: Model confidence (0-1) to show in title.
        top_n: Number of top features to display.
        figsize: Matplotlib figure size.

    Returns:
        matplotlib Figure object (caller is responsible for saving/displaying).
    """
    if len(shap_vals) != len(feature_names):
        raise ValueError(
            f"shap_vals length ({len(shap_vals)}) != "
            f"feature_names length ({len(feature_names)})"
        )

    # Select top-N features by absolute SHAP value
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_vals = shap_vals[indices]
    top_names = [feature_names[i] for i in indices]

    # Sort by value for visual waterfall ordering
    sort_order = np.argsort(top_vals)
    sorted_vals = top_vals[sort_order]
    sorted_names = [top_names[i] for i in sort_order]

    colours = [_RED if v > 0 else _BLUE for v in sorted_vals]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        sorted_names,
        sorted_vals,
        color=colours,
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, sorted_vals):
        label_x = val + 0.003 if val >= 0 else val - 0.003
        ha = "left" if val >= 0 else "right"
        ax.text(
            label_x, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha=ha,
            fontsize=8, color="#333333",
        )

    # Base value reference line
    ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--", alpha=0.7)

    # Title
    conf_str = f"  |  Confidence: {confidence * 100:.1f}%" if confidence is not None else ""
    ax.set_title(
        f"SHAP Waterfall -- Prediction: {prediction_label}{conf_str}\n"
        f"Base value: {base_value:.4f}  |  Top {top_n} contributing features",
        fontsize=10, pad=12,
    )
    ax.set_xlabel("SHAP Value (contribution to prediction)", fontsize=9)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_RED, label="-> Pushes toward attack"),
        Patch(facecolor=_BLUE, label="<- Pushes toward benign"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()

    log.info("Waterfall chart rendered for prediction='%s'", prediction_label)
    return fig
