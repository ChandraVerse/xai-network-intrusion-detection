# XAI-Based Network Intrusion Detection System

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost%20%7C%20LSTM-orange)
![XAI](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-CICIDS--2017-red)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![CI](https://github.com/ChandraVerse/xai-network-intrusion-detection/actions/workflows/ci.yml/badge.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

**A production-grade, explainable AI-powered Network Intrusion Detection System that detects
DDoS, brute force, web attacks, and infiltration — and tells you *why* it flagged each one.**

[🚀 Quick Start](#quick-start) · [📖 How It Works](#how-the-ai-works) · [💡 Benefits](#why-use-this-system) · [📊 Demo](#project-screenshots) · [🤝 Contributing](#contributing)

</div>

---

## What Is This Project?

This is an **Explainable AI (XAI)-based Network Intrusion Detection System (NIDS)** — a machine learning system that:

1. **Monitors** network traffic flows (packet metadata, byte counts, timing patterns)
2. **Classifies** each flow as either benign or one of 14 known attack types
3. **Explains** every single detection in plain, ranked English — powered by SHAP

Most AI-based security tools are black boxes. They raise an alert, but the analyst has no idea which network features triggered it. This project solves that problem entirely. Every alert comes with a ranked list of the exact features that caused the detection, visualized as a waterfall chart inside a live Streamlit dashboard.

> *"A model that detects threats but cannot explain them is a black box — and black boxes have no place in a SOC. Explainability is not a feature; it is a prerequisite for analyst trust."*

---

## Table of Contents

- [What Is This Project?](#what-is-this-project)
- [How the AI Works](#how-the-ai-works)
- [Why Use This System](#why-use-this-system)
- [Who This Is For](#who-this-is-for)
- [Detected Attack Types](#detected-attack-types)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML Models](#ml-models)
- [XAI Layer — SHAP](#xai-layer--shap)
- [Web Dashboard](#web-dashboard)
- [How to Use This Project](#how-to-use-this-project)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Phases](#project-phases)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Research Paper](#research-paper)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Glossary](#glossary)
- [Project Screenshots](#project-screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## How the AI Works

Here is the end-to-end journey of a single network flow through this system:

```
  Your Network Traffic
         │
         ▼
  CICFlowMeter extracts 78 features from raw packets
  (duration, byte counts, flag counts, inter-arrival times…)
         │
         ▼
  Preprocessing Pipeline
  (clean → scale → balance via SMOTE)
         │
         ▼
  ┌──────────────────────────────────────┐
  │  Three ML Models vote on each flow   │
  │  Random Forest  │ XGBoost │ LSTM     │
  └──────────────────────────────────────┘
         │
         ▼
  Prediction: "DDoS" — Confidence: 97.3%
         │
         ▼
  SHAP Explainer computes feature contributions
         │
         ▼
  Dashboard shows:
  ✔ Alert severity badge
  ✔ Waterfall chart of TOP 10 contributing features
  ✔ Human-readable explanation
```

---

## Why Use This System

| Challenge | What This Project Does |
|-----------|------------------------|
| 🔴 Alert fatigue — too many false positives | False positive rate < 0.3% on CICIDS-2017 |
| 🔴 Black-box AI — no explanation for alerts | SHAP waterfall chart for every single detection |
| 🔴 Slow triage — analysts manually investigate | Ranked feature list cuts investigation time drastically |
| 🔴 No benchmarking — can't compare models | Three models trained side-by-side with full metrics |
| 🔴 Deployment gap — notebooks stay notebooks | Full Streamlit dashboard + Docker container |

---

## Detected Attack Types

| # | Attack Class | MITRE ATT&CK | Description |
|---|-------------|-------------|-------------|
| 1 | BENIGN | — | Normal traffic |
| 2 | DDoS | T1498 | Volumetric flood |
| 3 | DoS Hulk | T1499 | HTTP flood |
| 4 | DoS GoldenEye | T1499 | Keep-alive DoS |
| 5 | DoS Slowloris | T1499 | Slow HTTP header |
| 6 | DoS Slowhttptest | T1499 | Slow HTTP body |
| 7 | FTP-Patator | T1110 | FTP brute force |
| 8 | SSH-Patator | T1110 | SSH brute force |
| 9 | PortScan | T1046 | Port sweep |
| 10 | Web Attack — Brute Force | T1110 | HTTP login BF |
| 11 | Web Attack — XSS | T1059.007 | XSS injection |
| 12 | Web Attack — SQLi | T1190 | SQL injection |
| 13 | Infiltration | T1078 | Lateral movement |
| 14 | Bot | T1071 | C2 botnet comms |

---

## Architecture

See [`docs/architecture/architecture_diagram.png`](docs/architecture/architecture_diagram.png) for the full system diagram.

```
PCAP / CSV  →  Preprocessing  →  RF + XGBoost + LSTM  →  SHAP  →  Streamlit Dashboard
```

---

## Dataset

**CICIDS-2017** — 2.8M+ labeled flows, 78 CICFlowMeter features, 14 classes.  
Source: [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## ML Models

| Model | Accuracy | Macro F1 | FPR | Inference |
|-------|----------|----------|-----|-----------|
| Random Forest | 99.94% | 0.997 | 0.28% | ~0.4 ms |
| XGBoost | 99.91% | 0.994 | 0.21% | ~0.2 ms |
| LSTM | 99.76% | 0.981 | 0.51% | ~1.8 ms |

> Results on CICIDS-2017 test split (20% holdout, SMOTE on train only).

---

## Quick Start

```bash
git clone https://github.com/ChandraVerse/xai-network-intrusion-detection.git
cd xai-network-intrusion-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Docker

```bash
docker-compose up --build
# Dashboard at http://localhost:8501
```

---

## Project Screenshots

### 1 · Streamlit Dashboard — Live Detection
![Dashboard](docs/screenshots/screenshot1_streamlit_dashboard.png)

### 2 · Model Comparison — ROC & Metrics
![Model Comparison](docs/screenshots/screenshot2_model_comparison.png)

### 3 · Global SHAP Feature Importance
![SHAP Summary](docs/screenshots/screenshot3_shap_summary.png)

### 4 · CICIDS-2017 Dataset Distribution
![Dataset](docs/screenshots/screenshot4_dataset_distribution.png)

---

## Project Structure

```
xai-network-intrusion-detection/
├── data/samples/          # Synthetic sample CSVs (generate_sample_data.py)
├── notebooks/             # 01_eda → 02_preprocessing → 03_training → 04_shap
├── src/                   # preprocessing/, models/, explainability/, utils/
├── dashboard/             # app.py + pages/ (live_detection, model_comparison, global_shap)
├── models/                # random_forest.pkl, xgboost_model.pkl, lstm_model.tar.gz
├── docs/
│   ├── architecture/      # architecture_diagram.png
│   ├── model_cards/       # RF, XGBoost, LSTM model cards
│   └── screenshots/       # Dashboard screenshots
├── scripts/               # run_pipeline.sh, run_tests.sh, generate_sample_data.py
├── tests/                 # unit + integration tests
├── paper/                 # xai_ids_paper.pdf (IEEE format)
├── Dockerfile
└── docker-compose.yml
```

---

## Model Cards

Detailed model cards with bias, performance, and intended use are in [`docs/model_cards/`](docs/model_cards/):
- [Random Forest](docs/model_cards/random_forest_card.md)
- [XGBoost](docs/model_cards/xgboost_card.md)
- [LSTM](docs/model_cards/lstm_card.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Fork → branch → PR.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Chandra Sekhar Chakraborty**  
Cybersecurity Analyst | SOC Analyst Aspirant | Graduating 2026  
📍 West Bengal, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-ChandraVerse-181717?logo=github)](https://github.com/ChandraVerse)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-4CAF50)](https://chandraverse.github.io/chandraverse-portfolio/)

---

<div align="center">

*Built with 🛡️ for the defensive security community — April 2026*

</div>
