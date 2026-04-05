# Screenshots

This directory contains dashboard screenshots and output visualisations for the XAI-NIDS project.

## Expected Contents

Once the dashboard has been run locally (`streamlit run dashboard/app.py`), add screenshots here:

| Filename | Description |
|---|---|
| `dashboard_overview.png` | Full dashboard landing page |
| `shap_waterfall_alert.png` | SHAP waterfall for a single DoS alert |
| `lime_explanation_alert.png` | LIME rule-based explanation for a PortScan alert |
| `model_comparison_panel.png` | Side-by-side RF vs XGBoost vs LSTM metrics |
| `confusion_matrix_rf.png` | Random Forest confusion matrix heatmap |
| `roc_curves_all.png` | ROC-AUC curves for all three models |

## How to Generate

```bash
# 1. Run the dashboard
streamlit run dashboard/app.py

# 2. Report figures are auto-saved during notebook 05 run:
#    reports/figures/confusion_matrices.png
#    reports/figures/roc_curves.png
#    reports/figures/shap_vs_lime.png
#    reports/figures/latency_benchmark.png

# 3. Copy relevant figures here for documentation
cp reports/figures/*.png docs/screenshots/
```

## README Badge

Add this to the root `README.md` to show the dashboard preview:

```markdown
![Dashboard Overview](docs/screenshots/dashboard_overview.png)
```
