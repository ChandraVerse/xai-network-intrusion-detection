"""
dashboard/pages/live_detection.py
----------------------------------
Streamlit multipage stub — Live Detection page.

This page is rendered by Streamlit's native multi-page app support
(files placed in dashboard/pages/ are auto-discovered).
Full implementation is embedded in dashboard/app.py; this stub
ensures the pages/ directory is importable and flake8-clean.
"""
import streamlit as st

st.set_page_config(page_title="Live Detection | XAI-NIDS", page_icon="🛡️", layout="wide")
st.title("🛡️ Live Detection")
st.info(
    "Live detection simulation is available on the main dashboard. "
    "Run: `streamlit run dashboard/app.py`",
    icon="ℹ️",
)
