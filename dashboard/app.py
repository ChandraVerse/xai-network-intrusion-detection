"""
Phase 5 — XAI Network Intrusion Detection Dashboard
====================================================
Author : Chandra Sekhar Chakraborty
Run    : streamlit run dashboard/app.py
Prereqs: Run notebooks 01-04 first so the following exist:
           models/random_forest.pkl
           models/xgboost_model.pkl
           models/lstm_model.keras
           models/minmax_scaler.pkl
           models/label_encoder.pkl
           data/processed/preprocessing_meta.json
           data/shap/consensus_features.json
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODELS_DIR    = os.path.join(ROOT, "models")
DATA_PROC_DIR = os.path.join(ROOT, "data", "processed")
SHAP_DATA_DIR = os.path.join(ROOT, "data", "shap")

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI-NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root { --bg: #0d1117; --surface: #161b22; --surface2: #1c2128;
          --border: #30363d; --text: #c9d1d9; --muted: #8b949e;
          --teal: #39d353; --red: #f85149; --blue: #58a6ff;
          --orange: #d29922; --purple: #bc8cff; }
  .stApp { background: var(--bg); color: var(--text); }
  section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
  .metric-card { background: var(--surface); border: 1px solid var(--border);
                 border-radius: 8px; padding: 16px 20px; text-align: center; }
  .metric-val  { font-size: 2rem; font-weight: 700; font-family: monospace; }
  .metric-lbl  { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }
  .alert-row   { background: var(--surface2); border-left: 3px solid var(--red);
                 border-radius: 4px; padding: 8px 12px; margin: 4px 0; font-size: 0.82rem; font-family: monospace; }
  .alert-benign{ border-left-color: var(--teal); }
  .section-hdr { font-size: 0.7rem; font-weight: 600; letter-spacing: .12em;
                 text-transform: uppercase; color: var(--muted); margin: 12px 0 6px; }
  div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS  (cached so they load once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    # Metadata
    with open(os.path.join(DATA_PROC_DIR, "preprocessing_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(SHAP_DATA_DIR, "consensus_features.json")) as f:
        shap_meta = json.load(f)

    feature_cols  = meta["feature_cols"]
    n_classes     = meta["n_classes"]
    label_map     = meta["label_map"]
    class_names   = [label_map[str(i)] for i in range(n_classes)]
    top_features  = shap_meta["top_features"]

    # Models
    rf   = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    xgb  = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    try:
        import tensorflow as tf
        lstm = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
    except Exception:
        lstm = None

    # Scaler + encoder
    scaler = joblib.load(os.path.join(MODELS_DIR, "minmax_scaler.pkl"))
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # SHAP explainers (tree models only — DeepExplainer is too slow for live use)
    rf_explainer  = shap.TreeExplainer(rf)
    xgb_explainer = shap.TreeExplainer(xgb)

    return {
        "feature_cols":  feature_cols,
        "n_classes":     n_classes,
        "class_names":   class_names,
        "top_features":  top_features,
        "rf":   rf,   "xgb":  xgb,  "lstm": lstm,
        "scaler": scaler, "le": le,
        "rf_explainer":  rf_explainer,
        "xgb_explainer": xgb_explainer,
    }

# ══════════════════════════════════════════════════════════════════════════════
# TRAFFIC SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_test_pool():
    """Load the test CSV as a simulation pool."""
    path = os.path.join(DATA_PROC_DIR, "test.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

def simulate_flow(pool_df, feature_cols, rng):
    """Draw a random flow from the test pool and add slight noise."""
    row = pool_df.sample(1, random_state=int(rng.integers(0, 999999))).iloc[0]
    X = row[feature_cols].values.astype(np.float32)
    # Add tiny noise so repeated flows look different
    X = np.clip(X + rng.normal(0, 0.005, size=X.shape), 0, 1)
    true_label = int(row["label"]) if "label" in row else -1
    return X, true_label

def predict_and_explain(X, model_name, artifacts):
    """Run inference + SHAP for one flow. Returns (pred_class, confidence, shap_vals_top20)."""
    fc      = artifacts["feature_cols"]
    top_f   = artifacts["top_features"]
    top_idx = [fc.index(f) for f in top_f if f in fc]
    X_row   = X.reshape(1, -1)

    if model_name == "Random Forest":
        proba  = artifacts["rf"].predict_proba(X_row)[0]
        pred   = int(np.argmax(proba))
        conf   = float(proba[pred])
        sv_raw = artifacts["rf_explainer"].shap_values(X_row)   # list[n_cls] of (1,F)
        sv     = np.stack([v[0] for v in sv_raw], axis=-1)      # (F, C)
        sv_top = sv[top_idx, pred]                               # (top_n,)
    elif model_name == "XGBoost":
        proba  = artifacts["xgb"].predict_proba(X_row)[0]
        pred   = int(np.argmax(proba))
        conf   = float(proba[pred])
        sv_raw = artifacts["xgb_explainer"].shap_values(X_row)
        if isinstance(sv_raw, list):
            sv  = np.stack([v[0] for v in sv_raw], axis=-1)
        else:
            sv  = sv_raw[0, :, np.newaxis]
        sv_top = sv[top_idx, min(pred, sv.shape[1]-1)]
    else:  # LSTM fallback to XGBoost SHAP
        proba  = artifacts["xgb"].predict_proba(X_row)[0]
        pred   = int(np.argmax(proba))
        conf   = float(proba[pred])
        sv_raw = artifacts["xgb_explainer"].shap_values(X_row)
        if isinstance(sv_raw, list):
            sv  = np.stack([v[0] for v in sv_raw], axis=-1)
        else:
            sv  = sv_raw[0, :, np.newaxis]
        sv_top = sv[top_idx, min(pred, sv.shape[1]-1)]

    return pred, conf, sv_top, [fc[i] for i in top_idx]

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
ATTACK_COLORS = {
    "BENIGN":          "#39d353",
    "DDoS":            "#f85149",
    "PortScan":        "#58a6ff",
    "Bot":             "#bc8cff",
    "Infiltration":    "#ffa657",
    "Web Attack":      "#d29922",
    "Brute Force":     "#ff7b72",
    "DoS Hulk":        "#f0883e",
    "DoS GoldenEye":   "#e3b341",
    "DoS Slowloris":   "#3fb950",
    "DoS Slowhttptest":"#56d364",
    "SSH-Patator":     "#79c0ff",
    "FTP-Patator":     "#a5d6ff",
    "Heartbleed":      "#ff6e96",
}
DEFAULT_COLOR = "#8b949e"

DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="monospace", size=11),
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
)

def waterfall_chart(sv_top, feat_names, pred_class_name, base_val=0.0):
    n = min(15, len(sv_top))
    order = np.argsort(np.abs(sv_top))[::-1][:n]
    vals  = sv_top[order]
    names = [feat_names[i][:28] for i in order]
    colors = ["#f85149" if v > 0 else "#58a6ff" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"SHAP Waterfall — <b>{pred_class_name}</b>", font=dict(size=12, color="#58a6ff")),
        height=420,
        yaxis=dict(autorange="reversed", gridcolor="#30363d"),
        xaxis_title="SHAP value",
        bargap=0.3,
    )
    return fig

def timeline_chart(history_df):
    if history_df.empty:
        return go.Figure(layout={**DARK_LAYOUT, "height": 220})
    fig = go.Figure()
    for cls in history_df["class_name"].unique():
        sub = history_df[history_df["class_name"] == cls]
        fig.add_trace(go.Scatter(
            x=sub["ts"], y=[cls]*len(sub),
            mode="markers",
            marker=dict(
                color=ATTACK_COLORS.get(cls, DEFAULT_COLOR),
                size=10, symbol="circle",
                line=dict(width=1, color="#0d1117")
            ),
            name=cls,
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Attack Timeline", font=dict(size=12, color="#58a6ff")),
        height=260,
        showlegend=False,
        yaxis=dict(gridcolor="#21262d"),
        xaxis_title="Time",
    )
    return fig

def donut_chart(counts_df):
    if counts_df.empty:
        return go.Figure(layout={**DARK_LAYOUT, "height": 280})
    fig = go.Figure(go.Pie(
        labels=counts_df["class"], values=counts_df["count"],
        hole=0.55,
        marker=dict(colors=[ATTACK_COLORS.get(c, DEFAULT_COLOR) for c in counts_df["class"]],
                    line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=9),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Traffic Distribution", font=dict(size=12, color="#58a6ff")),
        height=300,
        legend=dict(font=dict(size=8), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=4, r=4, t=36, b=4),
    )
    return fig

def top_features_bar(sv_top, feat_names):
    order = np.argsort(np.abs(sv_top))[::-1][:12]
    vals  = np.abs(sv_top[order])
    names = [feat_names[i][:22] for i in order]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals, colorscale="teal",
            line=dict(width=0)
        ),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside", textfont=dict(size=8),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Top Feature Importances (|SHAP|)", font=dict(size=11, color="#58a6ff")),
        height=340,
        yaxis=dict(autorange="reversed"),
        xaxis_title="|SHAP value|",
        bargap=0.25,
    )
    return fig

def confidence_gauge(conf, pred_class_name):
    color = ATTACK_COLORS.get(pred_class_name, DEFAULT_COLOR)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number=dict(suffix="%", font=dict(size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#8b949e", tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#161b22",
            bordercolor="#30363d",
            steps=[
                dict(range=[0,  50], color="#1c2128"),
                dict(range=[50, 80], color="#21262d"),
                dict(range=[80,100], color="#21262d"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=conf*100),
        ),
        title=dict(text="Confidence", font=dict(size=11, color="#8b949e")),
    ))
    fig.update_layout(
        **DARK_LAYOUT, height=200,
        margin=dict(l=16, r=16, t=32, b=8),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = dict(
        running=False,
        history=[],         # list of dicts: ts, pred, conf, class_name, true_label
        sv_top=None,
        sv_feat_names=None,
        last_pred=None,
        last_conf=None,
        last_class=None,
        total=0, attacks=0, tp=0, fp=0,
        rng=np.random.default_rng(42),
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ XAI-NIDS")
    st.markdown("<div class='section-hdr'>Model</div>", unsafe_allow_html=True)
    model_choice = st.selectbox("Active Model", ["Random Forest", "XGBoost", "LSTM"],
                                 label_visibility="collapsed")

    st.markdown("<div class='section-hdr'>Simulation Speed</div>", unsafe_allow_html=True)
    speed = st.slider("Flows per batch", 1, 10, 3, label_visibility="collapsed")
    delay = st.slider("Delay (seconds)", 0.2, 3.0, 0.8, step=0.1, label_visibility="collapsed")

    st.markdown("<div class='section-hdr'>Filters</div>", unsafe_allow_html=True)
    show_benign = st.checkbox("Show BENIGN traffic", value=False)
    conf_threshold = st.slider("Min confidence to alert", 0.0, 1.0, 0.5, step=0.05)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        start_btn = st.button("▶ Start", use_container_width=True, type="primary")
    with col_b:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)

    st.divider()
    reset_btn = st.button("🔄 Reset All", use_container_width=True)

    st.markdown("<div class='section-hdr'>About</div>", unsafe_allow_html=True)
    st.caption("XAI-Based Network Intrusion Detection\n\nDataset: CICIDS-2017\nModels: RF · XGBoost · LSTM\nXAI: SHAP TreeExplainer")

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if reset_btn:
    for k in ["history","sv_top","sv_feat_names","last_pred","last_conf",
              "last_class","total","attacks","tp","fp"]:
        st.session_state[k] = [] if k == "history" else None if "last" in k or "sv" in k else 0
    st.session_state.running = False
    st.session_state.rng = np.random.default_rng(42)
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
try:
    artifacts = load_artifacts()
    pool_df   = load_test_pool()
    data_ok   = pool_df is not None
except Exception as e:
    st.error(f"⚠️ Could not load artifacts: {e}\n\nRun notebooks 01–04 first.", icon="🚨")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛡️ XAI Network Intrusion Detection System")
st.markdown(
    f"<span style='color:#8b949e;font-size:0.82rem;font-family:monospace'>"
    f"Model: <b style='color:#58a6ff'>{model_choice}</b> &nbsp;|&nbsp; "
    f"Dataset: CICIDS-2017 &nbsp;|&nbsp; "
    f"Status: <b style='color:{'#39d353' if st.session_state.running else '#f85149'}'>"
    f"{'● LIVE' if st.session_state.running else '○ STOPPED'}</b>"
    f"</span>",
    unsafe_allow_html=True
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
total   = st.session_state.total
attacks = st.session_state.attacks
tp      = st.session_state.tp
fp      = st.session_state.fp

dr  = tp  / max(attacks, 1) * 100
far = fp  / max(total - attacks, 1) * 100

k1.metric("Total Flows",    f"{total:,}")
k2.metric("Attacks Detected", f"{attacks:,}", delta=f"+{attacks}" if attacks > 0 else None)
k3.metric("Detection Rate",   f"{dr:.1f}%")
k4.metric("False Alarm Rate", f"{far:.2f}%")
k5.metric("Benign Flows",     f"{total - attacks:,}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([3, 2], gap="medium")

with left_col:
    # Timeline
    history_df = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()
    st.plotly_chart(timeline_chart(history_df), use_container_width=True, config={"displayModeBar": False})

    # SHAP Waterfall
    st.markdown("<div class='section-hdr'>Real-Time SHAP Explanation — Last Alert</div>", unsafe_allow_html=True)
    if st.session_state.sv_top is not None:
        wf_fig = waterfall_chart(
            st.session_state.sv_top,
            st.session_state.sv_feat_names,
            st.session_state.last_class or "Unknown",
        )
        st.plotly_chart(wf_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("▶ Start the simulation to see real-time SHAP explanations.", icon="ℹ️")

with right_col:
    # Confidence gauge
    if st.session_state.last_conf is not None:
        st.plotly_chart(
            confidence_gauge(st.session_state.last_conf, st.session_state.last_class or ""),
            use_container_width=True, config={"displayModeBar": False}
        )

    # Donut
    if not history_df.empty:
        counts = history_df["class_name"].value_counts().reset_index()
        counts.columns = ["class", "count"]
        st.plotly_chart(donut_chart(counts), use_container_width=True, config={"displayModeBar": False})

    # Top features bar
    if st.session_state.sv_top is not None:
        st.plotly_chart(
            top_features_bar(st.session_state.sv_top, st.session_state.sv_feat_names),
            use_container_width=True, config={"displayModeBar": False}
        )

# ── Alert Feed ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>Alert Feed</div>", unsafe_allow_html=True)
feed_placeholder = st.empty()

def render_feed(history, show_benign, conf_threshold, class_names):
    rows = []
    for h in reversed(history[-50:]):
        if not show_benign and h["class_name"] == "BENIGN":
            continue
        if h["conf"] < conf_threshold:
            continue
        color_cls = "alert-benign" if h["class_name"] == "BENIGN" else "alert-row"
        true_lbl  = class_names[h["true_label"]] if 0 <= h.get("true_label", -1) < len(class_names) else "?"
        rows.append(
            f"<div class='{color_cls}'>"
            f"<b>{h['ts']}</b> &nbsp; "
            f"<span style='color:{ATTACK_COLORS.get(h['class_name'], DEFAULT_COLOR)}'><b>{h['class_name']}</b></span>"
            f" &nbsp; conf={h['conf']:.3f} &nbsp; true={true_lbl}"
            f"</div>"
        )
    feed_placeholder.markdown("\n".join(rows) if rows else "<p style='color:#8b949e'>No alerts yet.</p>",
                               unsafe_allow_html=True)

render_feed(st.session_state.history, show_benign, conf_threshold, artifacts["class_names"])

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running and data_ok:
    for _ in range(speed):
        X_flow, true_label = simulate_flow(pool_df, artifacts["feature_cols"], st.session_state.rng)

        try:
            pred, conf, sv_top, feat_names = predict_and_explain(X_flow, model_choice, artifacts)
        except Exception:
            continue

        class_name = artifacts["class_names"][pred] if pred < len(artifacts["class_names"]) else "Unknown"
        ts = pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3]

        # Update session state
        st.session_state.history.append(dict(ts=ts, pred=pred, conf=conf,
                                              class_name=class_name, true_label=true_label))
        st.session_state.total   += 1
        is_attack = (class_name != "BENIGN")
        if is_attack:
            st.session_state.attacks += 1
            if true_label != -1 and true_label == pred:
                st.session_state.tp += 1
            elif true_label != -1 and true_label == 0:   # false positive
                st.session_state.fp += 1

        # Always update SHAP for last alert (or benign if show_benign)
        if is_attack or show_benign:
            st.session_state.sv_top        = sv_top
            st.session_state.sv_feat_names = feat_names
            st.session_state.last_pred     = pred
            st.session_state.last_conf     = conf
            st.session_state.last_class    = class_name

    time.sleep(delay)
    st.rerun()
elif st.session_state.running and not data_ok:
    st.warning("Test CSV not found. Run notebooks 01–02 first to generate `data/processed/test.csv`.")
    st.session_state.running = False
