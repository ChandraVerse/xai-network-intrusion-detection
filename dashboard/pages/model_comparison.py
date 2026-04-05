"""
dashboard/pages/model_comparison.py
-------------------------------------
Streamlit multipage stub — Model Comparison page.

Displays side-by-side performance metrics for RF, XGBoost, and LSTM
loaded from models/*_metrics.json artefacts.
"""
import json
import os

import streamlit as st

st.set_page_config(
    page_title="Model Comparison | XAI-NIDS", page_icon="📊", layout="wide"
)
st.title("📊 Model Comparison")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT, "models")

METRIC_FILES = {
    "Random Forest": "rf_metrics.json",
    "XGBoost":       "xgb_metrics.json",
    "LSTM":          "lstm_metrics.json",
}

records = []
for model_name, fname in METRIC_FILES.items():
    fpath = os.path.join(MODELS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        data["Model"] = model_name
        records.append(data)

if records:
    import pandas as pd
    df = pd.DataFrame(records).set_index("Model")
    st.dataframe(df, use_container_width=True)
else:
    st.warning(
        "No metrics files found in `models/`. "
        "Run notebooks 01–04 to generate model artefacts.",
        icon="⚠️",
    )
