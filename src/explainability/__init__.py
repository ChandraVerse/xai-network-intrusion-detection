"""XAI explainability modules for XAI-NIDS.

Exports both SHAP and LIME explainers so downstream code
can import from a single namespace::

    from src.explainability import shap_explainer, lime_explainer
    # or the class directly:
    from src.explainability.lime_explainer import LimeExplainer
    from src.explainability.shap_explainer import explain as shap_explain
    from src.explainability.lime_explainer import explain as lime_explain
"""

from src.explainability import shap_explainer  # noqa: F401
from src.explainability import lime_explainer   # noqa: F401
from src.explainability.shap_explainer import (
    build_explainer,
    explain as shap_explain,
    explain_single as shap_explain_single,
)
from src.explainability.lime_explainer import (
    LimeExplainer,
    explain as lime_explain,
    explain_single as lime_explain_single,
    make_keras_predict_fn,
)
from src.explainability.waterfall import plot_waterfall
from src.explainability.summary_plot import plot_summary

__all__ = [
    # SHAP
    "shap_explainer",
    "build_explainer",
    "shap_explain",
    "shap_explain_single",
    "plot_waterfall",
    "plot_summary",
    # LIME
    "lime_explainer",
    "LimeExplainer",
    "lime_explain",
    "lime_explain_single",
    "make_keras_predict_fn",
]
