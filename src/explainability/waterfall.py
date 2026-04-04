"""
waterfall.py
------------
Generate per-alert SHAP waterfall charts.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_waterfall(
    shap_explanation: shap.Explanation,
    sample_index: int = 0,
    max_display: int = 10,
    save_path: str = None,
    title: str = "SHAP Waterfall — Feature Contributions",
) -> None:
    """
    Plot a SHAP waterfall chart for a single prediction.

    Parameters
    ----------
    shap_explanation : shap.Explanation object from TreeExplainer
    sample_index     : which sample in the batch to visualise
    max_display      : number of top features to display
    save_path        : if provided, saves the figure to this path
    title            : chart title string
    """
    # Handle multi-class output — pick the predicted class
    exp = shap_explanation[sample_index]
    if exp.values.ndim == 2:
        # Multi-class: pick class with highest absolute sum
        class_idx = int(np.abs(exp.values).sum(axis=0).argmax())
        values     = exp.values[:, class_idx]
        base_value = exp.base_values[class_idx]
        exp_single = shap.Explanation(
            values=values,
            base_values=base_value,
            data=exp.data,
            feature_names=exp.feature_names,
        )
    else:
        exp_single = exp

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(exp_single, max_display=max_display, show=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [waterfall] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
