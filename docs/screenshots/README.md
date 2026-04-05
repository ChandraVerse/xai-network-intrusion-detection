# Dashboard Screenshots

These screenshots show the four main pages of the XAI-NIDS Streamlit dashboard.
Generated from the live dashboard running against CICIDS-2017 data.

## screenshot1_streamlit_dashboard.png
**Live Detection page** — Upload a CSV of network flows, get instant per-flow
predictions from the ensemble model, see the SHAP waterfall for each alert,
and monitor real-time KPI cards (total flows, alerts, benign ratio, top threat).

## screenshot2_model_comparison.png
**Model Comparison page** — Side-by-side ROC curves, per-class F1 bar chart,
and normalised confusion matrix for Random Forest vs XGBoost vs LSTM.
Metrics table shows accuracy, macro F1/precision/recall, FPR, AUC-ROC,
and per-flow inference latency.

## screenshot3_shap_summary.png
**Global SHAP page** — Beeswarm summary plot and mean |SHAP| bar chart for
the top-15 most important CICFlowMeter features (Random Forest, full test set).
Colour encodes feature value magnitude (blue = low, red = high).

## screenshot4_dataset_distribution.png
**Dataset Overview** — CICIDS-2017 class distribution donut, flow duration
histogram by class, bytes/s vs packets/s scatter, SMOTE before/after balance
bar chart, and top-10 feature correlation heatmap.

## Regenerate
Run the Streamlit dashboard (`streamlit run dashboard/app.py`) and capture
screenshots, or run:
```bash
python docs/screenshots/generate_screenshots.py
```
