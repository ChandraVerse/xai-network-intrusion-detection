"""
summary_plot.py
---------------
Global SHAP beeswarm and dependence plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_beeswarm(
    shap_explanation: shap.Explanation,
    max_display: int = 20,
    save_path: str = None,
    title: str = "SHAP Global Feature Importance (Beeswarm)",
) -> None:
    """
    Plot a SHAP beeswarm summary across all samples.

    Parameters
    ----------
    shap_explanation : shap.Explanation from TreeExplainer
    max_display      : number of top features to show
    save_path        : if provided, saves the figure
    title            : chart title
    """
    # Flatten multi-class SHAP values by taking mean absolute across classes
    exp = shap_explanation
    if exp.values.ndim == 3:
        vals = exp.values.mean(axis=-1)      # (samples, features)
        exp  = shap.Explanation(
            values=vals,
            base_values=exp.base_values.mean(axis=-1),
            data=exp.data,
            feature_names=exp.feature_names,
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(exp, max_display=max_display, show=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [beeswarm] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_dependence(
    shap_explanation: shap.Explanation,
    feature: str,
    interaction_feature: str = "auto",
    save_path: str = None,
) -> None:
    """
    Plot SHAP dependence plot for a single feature.

    Parameters
    ----------
    shap_explanation    : shap.Explanation from TreeExplainer
    feature             : name of the primary feature
    interaction_feature : feature to colour by ('auto' lets SHAP choose)
    save_path           : if provided, saves the figure
    """
    exp = shap_explanation
    feature_names = exp.feature_names or []
    feat_idx = feature_names.index(feature) if feature in feature_names else 0

    vals = exp.values
    if vals.ndim == 3:
        vals = vals.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx, vals, exp.data,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        ax=ax, show=False,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [dependence] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
