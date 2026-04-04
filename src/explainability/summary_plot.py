"""Global SHAP summary visualisations (beeswarm, bar, dependence plots).

Used for model validation and global interpretability — not per-alert triage.

Usage:
    from src.explainability.summary_plot import (
        plot_beeswarm, plot_bar, plot_dependence
    )

    fig = plot_beeswarm(shap_values, X_test, feature_names)
    fig.savefig("docs/eda_plots/shap_beeswarm.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def plot_beeswarm(
    shap_values,                  # list[np.ndarray] (n_classes, n_samples, n_features)
    X_test: np.ndarray,
    feature_names: list[str],
    class_idx: int = 0,
    max_display: int = 20,
    figsize: tuple[float, float] = (10, 7),
) -> plt.Figure:
    """Global beeswarm summary plot — shows distribution of SHAP values across test set.

    Args:
        shap_values:   SHAP values from TreeExplainer (list of per-class arrays).
        X_test:        Test feature matrix, shape (n_samples, n_features).
        feature_names: Feature names list.
        class_idx:     Which class to plot (default 0 = BENIGN, change per attack).
        max_display:   Number of top features to show.
        figsize:       Matplotlib figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)

    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(
        sv,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    ax.set_title(
        f"SHAP Beeswarm Summary — Class index {class_idx}  |  Top {max_display} features",
        fontsize=11, pad=12,
    )
    plt.tight_layout()
    log.info("Beeswarm plot rendered (class_idx=%d, features=%d)", class_idx, max_display)
    return fig


def plot_bar(
    shap_values,
    feature_names: list[str],
    max_display: int = 20,
    figsize: tuple[float, float] = (9, 6),
) -> plt.Figure:
    """Bar chart of mean absolute SHAP values per feature (global importance)."""
    # Mean absolute SHAP across classes and samples
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    indices = np.argsort(mean_abs)[::-1][:max_display]
    sorted_vals = mean_abs[indices]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(sorted_names[::-1], sorted_vals[::-1], color="#01696f", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title(f"Global Feature Importance — Top {max_display} features", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    log.info("Bar importance plot rendered")
    return fig


def plot_dependence(
    feature: str,
    shap_values,
    X_test: np.ndarray,
    feature_names: list[str],
    interaction_feature: str | None = None,
    class_idx: int = 0,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Dependence plot: feature value vs SHAP(feature), coloured by interaction feature.

    Args:
        feature:              Name of the feature to plot on x-axis.
        shap_values:          SHAP values array.
        X_test:               Test feature matrix.
        feature_names:        Feature name list.
        interaction_feature:  Optional feature for dot colouring (auto if None).
        class_idx:            Class to use for SHAP values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)

    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    shap.dependence_plot(
        feature,
        sv,
        X_test,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    ax.set_title(
        f"SHAP Dependence: '{feature}' (class idx {class_idx})",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    log.info("Dependence plot rendered for feature='%s'", feature)
    return fig
