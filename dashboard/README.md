# XAI-NIDS Dashboard

A real-time Streamlit dashboard for the XAI Network Intrusion Detection System.

## Quick Start

```bash
# 1. Train models and generate processed data (run once)
python scripts/bootstrap_artifacts.py

# 2. (Optional) generate sample CSVs for the simulator
python scripts/generate_samples.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

## Requirements

All dependencies are in `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `streamlit>=1.32` | Dashboard framework |
| `plotly>=5.18` | Interactive charts |
| `shap>=0.44` | SHAP explainability |
| `lime>=0.2` | LIME explainability |
| `scikit-learn>=1.3` | RF model + preprocessing |
| `xgboost>=2.0` | XGBoost model |
| `tensorflow-cpu>=2.13` | LSTM model (optional) |

## Configuration

All paths and default settings are in `dashboard/config.py`. Override with environment variables:

```bash
export XAI_NIDS_ROOT=/path/to/repo          # override auto-detected root
export XAI_NIDS_LOG_LEVEL=DEBUG              # verbose logging
streamlit run dashboard/app.py
```

## File Structure

```
dashboard/
├── app.py          Main Streamlit application
├── config.py       Centralised paths & settings
└── README.md       This file
```

## Expected Artifacts

Before launching, ensure these files exist (created by `bootstrap_artifacts.py`):

```
models/
  random_forest.pkl
  xgboost_model.pkl
  lstm_model.h5
data/processed/
  X_test.npy  y_test.npy
  scaler.pkl  label_encoder.pkl
  feature_names.json  label_map.json
```
