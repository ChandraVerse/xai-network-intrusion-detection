"""
dashboard/pages/global_shap.py
--------------------------------
Streamlit multipage stub — Global SHAP Summary page.

Loads pre-computed SHAP summary data from data/shap/ and renders
a global feature importance bar chart.
"""
import json
import os

import streamlit as st

st.set_page_config(
    page_title="Global SHAP | XAI-NIDS", page_icon="🔍", layout="wide"
)
st.title("🔍 Global SHAP Feature Importance")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHAP_DIR = os.path.join(ROOT, "data", "shap")

consensus_path = os.path.join(SHAP_DIR, "consensus_features.json")
if os.path.exists(consensus_path):
    with open(consensus_path, encoding="utf-8") as fh:
        shap_meta = json.load(fh)
    top_features = shap_meta.get("top_features", [])
    if top_features:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(
            {"Feature": top_features, "Rank": range(1, len(top_features) + 1)}
        )
        fig = px.bar(
            df, x="Rank", y="Feature", orientation="h",
            title="Top Features by Consensus SHAP Rank",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No top_features key found in consensus_features.json.")
else:
    st.warning(
        "SHAP consensus file not found at `data/shap/consensus_features.json`. "
        "Run notebook 04 to generate SHAP artefacts.",
        icon="⚠️",
    )
