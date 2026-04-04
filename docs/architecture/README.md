# Architecture Documentation

This folder contains the full architecture documentation for the **XAI-Based Network Intrusion Detection System**.

## Files

| File | Description |
|------|-------------|
| [`system_architecture.svg`](system_architecture.svg) | Full-colour layered system diagram (SVG, scalable) |
| [`dataflow.md`](dataflow.md) | End-to-end data flow with Mermaid diagrams |
| [`components.md`](components.md) | Detailed component reference for every module |
| [`decisions.md`](decisions.md) | Architecture Decision Records (ADRs) — why each design choice was made |

---

## System Overview

The system is structured as five sequential layers:

```
INPUT LAYER
    ↓  raw PCAP / CSV
PREPROCESSING PIPELINE
    ↓  cleaned · scaled · SMOTE-balanced
ML DETECTION ENGINE  (RF + XGBoost + LSTM)
    ↓  attack label + confidence score
XAI EXPLAINABILITY LAYER  (SHAP)
    ↓  explanation object (waterfall / beeswarm)
STREAMLIT DASHBOARD
    ↓  Docker-containerised
DEPLOYMENT  (docker-compose)
```

Each layer is independently testable and replaceable — the ML engine can be swapped for a new model without touching the dashboard, and the SHAP layer can be updated independently of the preprocessing pipeline.

---

## Quick Reference — Module Locations

| Layer | Source Code | Notebook |
|-------|------------|----------|
| Preprocessing | `src/preprocessing/` | `notebooks/02_preprocessing.ipynb` |
| Random Forest | `src/models/random_forest.py` | `notebooks/03_model_training.ipynb` |
| XGBoost | `src/models/xgboost_model.py` | `notebooks/03_model_training.ipynb` |
| LSTM | `src/models/lstm_model.py` | `notebooks/03_model_training.ipynb` |
| SHAP | `src/explainability/shap_explainer.py` | `notebooks/04_xai_shap.ipynb` |
| Dashboard | `dashboard/app.py` | — |
| Docker | `Dockerfile`, `docker-compose.yml` | — |
