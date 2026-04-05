# XAI-NIDS Architecture

## Overview

This project implements an Explainable AI (XAI) pipeline for Network Intrusion Detection using the CICIDS-2017 dataset. The system classifies network flows into 14 classes (1 benign + 13 attack types) and provides human-readable explanations via SHAP and LIME.

## Data Flow

```
Raw PCAP / CICIDS-2017 CSV
          │
          ▼
┌─────────────────────┐
│  notebooks/01_eda   │  Exploratory Data Analysis
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  notebooks/02_pre   │  Preprocessing: cleaning, scaling, encoding
│  scripts/           │  → data/processed/*.npy + *.pkl
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  notebooks/03_model_training            │
│  scripts/bootstrap_artifacts.py         │
│                                         │
│   Random Forest ──► models/rf.pkl       │
│   XGBoost       ──► models/xgb.pkl      │
│   LSTM          ──► models/lstm.h5      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  notebooks/04_xai_shap                  │
│  src/explainability/shap_explainer.py   │
│  src/explainability/lime_explainer.py   │
│                                         │
│  SHAP TreeExplainer  (RF, XGBoost)      │
│  SHAP DeepExplainer  (LSTM)             │
│  LIME TabularExplainer (all models)     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  notebooks/05_model_comparison          │
│                                         │
│  Side-by-side metrics, ROC curves,      │
│  SHAP vs LIME agreement, latency        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  dashboard/app.py (Streamlit)           │
│                                         │
│  Real-time simulation + SHAP waterfall  │
│  per-alert explanations for SOC use     │
└─────────────────────────────────────────┘
```

## Module Map

| Path | Responsibility |
|---|---|
| `src/models/` | Model training wrappers (RF, XGBoost, LSTM) |
| `src/explainability/shap_explainer.py` | SHAP global + local explanations |
| `src/explainability/lime_explainer.py` | LIME local instance explanations |
| `src/explainability/waterfall.py` | SHAP waterfall chart renderer |
| `src/explainability/summary_plot.py` | SHAP beeswarm / bar summary plots |
| `src/utils/metrics.py` | Evaluation: accuracy, F1, FPR, DR |
| `src/utils/logger.py` | Centralised logging factory |
| `src/utils/report_generator.py` | PDF/HTML report generation |
| `src/utils/pcap_converter.py` | PCAP → feature vector converter |
| `scripts/generate_samples.py` | Synthetic sample CSV generator |
| `scripts/bootstrap_artifacts.py` | One-shot model training & artifact generation |
| `dashboard/app.py` | Streamlit real-time dashboard |
| `dashboard/config.py` | Dashboard path & settings config |

## XAI Decision Matrix

| Scenario | Recommended XAI |
|---|---|
| Global feature importance ranking | SHAP (TreeExplainer, beeswarm) |
| Per-alert SOC triage | SHAP waterfall OR LIME local |
| Non-tree model (LSTM) explanation | SHAP DeepExplainer or LIME |
| Rule-based explanation for audit | LIME (produces human-readable conditions) |
| Fast real-time dashboard | SHAP TreeExplainer (pre-computed) |
