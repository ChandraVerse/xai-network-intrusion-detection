# XAI-Based Network Intrusion Detection System

<div align="center">

[![CI](https://github.com/ChandraVerse/xai-network-intrusion-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ChandraVerse/xai-network-intrusion-detection/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-yellow?logo=python)
![ML](https://img.shields.io/badge/Models-RF%20%7C%20XGBoost%20%7C%20LSTM-orange)
![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-CICIDS--2017-red)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker)
![Tests](https://img.shields.io/badge/Tests-42%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-tracked-informational)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**A production-grade, explainable AI-powered Network Intrusion Detection System.**  
Detects DDoS, brute force, web attacks, and infiltration — and tells you *exactly why* it flagged each one.

[🚀 Quick Start](#quick-start) · [🏗️ Architecture](#architecture) · [📊 Models](#ml-models) · [💡 XAI Layer](#xai-layer--shap--lime) · [🐳 Docker](#docker-deployment) · [🤝 Contributing](#contributing)

</div>

---

## What Is This?

**XAI-NIDS** is a machine learning pipeline that classifies network traffic flows into 15 categories (1 benign + 14 attack types) trained on the **CICIDS-2017** dataset. What sets it apart from standard NIDS implementations is the **full explainability layer** — every single prediction is backed by SHAP and LIME feature attributions surfaced in a live Streamlit dashboard.

Most AI-based security tools raise an alert with no context. Analysts have no idea which network features triggered it. This project solves that gap — every alert comes with a ranked, human-readable explanation so SOC analysts can triage faster and with confidence.

> *"A model that detects threats but cannot explain them is a black box — and black boxes have no place in a SOC. Explainability is not a feature; it is a prerequisite for analyst trust."*

---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Detected Attack Types](#detected-attack-types)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML Models](#ml-models)
- [XAI Layer — SHAP + LIME](#xai-layer--shap--lime)
- [Web Dashboard](#web-dashboard)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [CI / CD Pipeline](#ci--cd-pipeline)
- [Tech Stack](#tech-stack)
- [Model Cards](#model-cards)
- [Research Paper](#research-paper)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## How It Works

A single network flow travels through this pipeline:

```
  Raw Network Traffic (PCAP / CSV)
           │
           ▼
  CICFlowMeter → 78 features extracted
  (duration, byte counts, flag counts, inter-arrival times…)
           │
           ▼
  Preprocessing Pipeline
  ├─ Drop infinities / NaNs
  ├─ Standard scaling
  └─ SMOTE class balancing (train split only)
           │
           ▼
  ┌─────────────────────────────────────────┐
  │  Ensemble of Three ML Models            │
  │  Random Forest │ XGBoost │ LSTM         │
  └─────────────────────────────────────────┘
           │
           ▼
  Prediction: "DDoS"  —  Confidence: 97.3%
           │
           ▼
  XAI Layer
  ├─ SHAP TreeExplainer  →  global + per-prediction waterfall charts
  └─ LIME TabularExplainer  →  local surrogate explanations
           │
           ▼
  Streamlit Dashboard
  ├─ Live detection tab (upload CSV or single flow)
  ├─ Model comparison (ROC curves, confusion matrices, F1 per class)
  └─ Global SHAP feature importance viewer
```

---

## Features

- **Three ML models** trained and evaluated side-by-side — Random Forest, XGBoost, LSTM
- **Dual XAI layer** — SHAP for tree/ensemble models, LIME for all models (including LSTM)
- **Live Streamlit dashboard** with real-time detection, SHAP waterfall charts, and model comparison
- **42 passing unit + integration tests** with pytest, `--cov` coverage reporting
- **Docker + docker-compose** for one-command deployment
- **Full CI/CD pipeline** — lint (flake8), tests (pytest + codecov), Docker build, security scan (Bandit), sample data smoke test
- **CICIDS-2017 dataset support** — 2.8M+ labeled flows, 78 features, 14 attack classes
- **PCAP ingestion utility** via CICFlowMeter wrapper
- **Research paper** included (`paper/`) in IEEE format

---

## Detected Attack Types

| # | Attack Class | MITRE ATT&CK | Description |
|---|-------------|-------------|-------------|
| 0 | BENIGN | — | Normal traffic |
| 1 | DDoS | T1498 | Volumetric flood |
| 2 | DoS Hulk | T1499 | HTTP flood |
| 3 | DoS GoldenEye | T1499 | Keep-alive DoS |
| 4 | DoS Slowloris | T1499 | Slow HTTP header attack |
| 5 | DoS Slowhttptest | T1499 | Slow HTTP body attack |
| 6 | FTP-Patator | T1110 | FTP brute force |
| 7 | SSH-Patator | T1110 | SSH brute force |
| 8 | PortScan | T1046 | Port sweep / reconnaissance |
| 9 | Web Attack — Brute Force | T1110 | HTTP login brute force |
| 10 | Web Attack — XSS | T1059.007 | Cross-site scripting injection |
| 11 | Web Attack — SQLi | T1190 | SQL injection |
| 12 | Infiltration | T1078 | Lateral movement / valid account abuse |
| 13 | Bot | T1071 | C2 botnet communications |
| 14 | Heartbleed | T1499 | OpenSSL memory leak exploit |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Data Layer                          │
│  PCAP → CICFlowMeter → CSV → data/samples/               │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                  Preprocessing (src/preprocessing/)      │
│  clean.py  →  scaler.py  →  balancer.py (SMOTE)          │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                    Model Layer (src/models/)              │
│   random_forest.py  │  xgboost_model.py  │  lstm.py      │
│   trainer.py        │  evaluator.py                      │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│              Explainability (src/explainability/)        │
│   shap_explainer.py   │   lime_explainer.py              │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                 Dashboard (dashboard/)                   │
│   app.py → pages/live_detection, model_comparison,       │
│            global_shap, batch_analysis                   │
└──────────────────────────────────────────────────────────┘
```

Full diagram: [`docs/architecture/architecture_diagram.png`](docs/architecture/architecture_diagram.png)

---

## Dataset

**CICIDS-2017** by the Canadian Institute for Cybersecurity.

| Property | Value |
|---|---|
| Total flows | 2,830,743 |
| Features | 78 (CICFlowMeter) |
| Attack classes | 14 |
| Benign traffic | ~80% |
| Time span | Monday–Friday, July 3–7 2017 |

Source: [University of New Brunswick — CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)

> The dataset is **not included** in this repo due to size. Download it from the UNB link above and place CSVs in `data/raw/`. The preprocessing pipeline handles the rest.

---

## ML Models

| Model | Accuracy | Macro F1 | FPR | Avg Inference |
|-------|----------|----------|-----|---------------|
| Random Forest | 99.94% | 0.997 | 0.28% | ~0.4 ms/sample |
| XGBoost | 99.91% | 0.994 | 0.21% | ~0.2 ms/sample |
| LSTM | 99.76% | 0.981 | 0.51% | ~1.8 ms/sample |

> Evaluated on CICIDS-2017 test split (20% holdout). SMOTE applied to training split only to prevent data leakage.

**Model implementations:**
- `src/models/random_forest.py` — scikit-learn `RandomForestClassifier`, tuned via `RandomizedSearchCV`
- `src/models/xgboost_model.py` — `XGBClassifier` with early stopping
- `src/models/lstm.py` — Keras sequential LSTM, sliding window of 5 timesteps
- `src/models/trainer.py` — unified training entrypoint for all three
- `src/models/evaluator.py` — classification report, ROC-AUC, confusion matrix

---

## XAI Layer — SHAP + LIME

### SHAP (`src/explainability/shap_explainer.py`)
Uses `shap.TreeExplainer` for Random Forest and XGBoost (exact Shapley values, fast). Outputs:
- **Waterfall chart** — per-prediction, shows each feature's push toward/away from the predicted class
- **Beeswarm plot** — global feature importance across all predictions
- **Decision plot** — cumulative feature contribution path

### LIME (`src/explainability/lime_explainer.py`)
Uses `LimeTabularExplainer` for local surrogate explanations. Works with all three models including LSTM. Outputs:
- **Feature weight bar chart** — top-N features driving the local prediction
- **JSON export** — serialisable explanation for SOC workflow integration

Both explainers expose a consistent API:
```python
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LIMEExplainer

# SHAP
exp = ShapExplainer(model, background_data, feature_names)
exp.explain_single(x)          # → shap.Explanation
exp.plot_waterfall(shap_vals)  # → matplotlib Figure

# LIME
exp = LIMEExplainer(training_data, feature_names, class_names)
result = exp.explain_instance(x, model.predict_proba)
exp.plot_explanation(result)   # → matplotlib Figure
exp.as_dict(result)            # → JSON-serialisable dict
```

---

## Web Dashboard

Built with **Streamlit**. Four pages:

| Page | What it does |
|---|---|
| **Live Detection** | Upload a CSV or enter a single flow → instant prediction + SHAP waterfall |
| **Model Comparison** | Side-by-side ROC curves, precision-recall, confusion matrices for all 3 models |
| **Global SHAP** | Beeswarm / bar plots for global feature importance across the full dataset |
| **Batch Analysis** | Upload a large CSV, get per-row predictions + downloadable results |

Launch:
```bash
streamlit run dashboard/app.py
```
Or via Docker (see below).

---

## Quick Start

### Prerequisites
- Python 3.11
- pip
- (Optional) Docker + docker-compose

### Install & Run

```bash
# 1. Clone
git clone https://github.com/ChandraVerse/xai-network-intrusion-detection.git
cd xai-network-intrusion-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic sample data (for testing without the full dataset)
python scripts/generate_samples.py --rows 500 --out data/samples/sample_100.csv

# 5. Launch dashboard
streamlit run dashboard/app.py
# → Open http://localhost:8501
```

### Run Tests

```bash
# Unit tests only (fast, no dataset needed)
PYTHONPATH=. pytest tests/ -v --ignore=tests/test_integration.py

# All tests including integration
PYTHONPATH=. pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

### Train Models (requires CICIDS-2017 dataset)

```bash
# Place CICIDS-2017 CSVs in data/raw/
# Then run the full pipeline:
bash scripts/run_pipeline.sh

# Or train a single model:
PYTHONPATH=. python -m src.models.trainer --model random_forest --out models/
```

---

## Docker Deployment

```bash
# Build and run everything (dashboard + all services)
docker-compose up --build

# Dashboard → http://localhost:8501
```

Single container:
```bash
docker build -t xai-nids .
docker run -p 8501:8501 xai-nids
```

---

## Project Structure

```
xai-network-intrusion-detection/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Main CI: lint → test → docker → security
│       └── train_and_commit_artifacts.yml
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py                 # Drop NaN, inf, duplicates
│   │   ├── scaler.py                  # StandardScaler wrapper
│   │   └── balancer.py                # SMOTE class balancing
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lstm.py
│   │   ├── trainer.py                 # Unified train entrypoint
│   │   └── evaluator.py               # Metrics + plots
│   ├── explainability/
│   │   ├── shap_explainer.py          # SHAP TreeExplainer wrapper
│   │   └── lime_explainer.py          # LIME TabularExplainer wrapper
│   └── utils/
│       ├── pcap_converter.py          # PCAP → feature CSV via CICFlowMeter
│       └── logger.py
│
├── dashboard/
│   ├── app.py                         # Streamlit entry point
│   ├── config.py                      # Paths, class names, constants
│   └── pages/
│       ├── live_detection.py
│       ├── model_comparison.py
│       ├── global_shap.py
│       └── batch_analysis.py
│
├── tests/
│   ├── conftest.py                    # Shared fixtures (fitted RF, sample data)
│   ├── test_preprocessing.py
│   ├── test_random_forest.py
│   ├── test_xgboost.py
│   ├── test_lstm.py
│   ├── test_shap_explainer.py
│   ├── test_lime_explainer.py
│   ├── test_dashboard.py
│   ├── test_generate_samples.py
│   └── test_integration.py            # End-to-end (skipped in CI unit run)
│
├── scripts/
│   ├── generate_samples.py            # Synthetic data generator (no dataset needed)
│   ├── run_pipeline.sh                # Full train + evaluate pipeline
│   └── run_tests.sh
│
├── data/
│   ├── raw/                           # CICIDS-2017 CSVs (not in repo — download separately)
│   └── samples/                       # Generated synthetic fixtures
│
├── models/                            # Serialised model artefacts (.pkl, .tar.gz)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_shap_analysis.ipynb
│
├── docs/
│   ├── architecture/                  # System diagram
│   ├── model_cards/                   # RF, XGBoost, LSTM model cards
│   └── screenshots/                   # Dashboard screenshots
│
├── paper/                             # IEEE-format research paper (PDF + LaTeX)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                   # Full runtime dependencies
├── requirements-ci.txt                # Lightweight CI-only dependencies
├── .flake8                            # Linting config
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

---

## CI / CD Pipeline

Every push to `main` or `develop` runs four parallel jobs after `test` passes:

```
push / PR
    │
    ▼
┌─────────────────────────────────────┐
│  test (Python 3.11)                 │
│  ├─ pip install -r requirements-ci  │
│  ├─ Generate synthetic fixture data │
│  ├─ pytest (42 tests, --cov)        │
│  ├─ Upload coverage → Codecov       │
│  └─ flake8 src/ scripts/            │
└──────────────┬──────────────────────┘
               │ needs: test
    ┌──────────┼──────────┬──────────────┐
    ▼          ▼          ▼              ▼
 docker   generate-   security      (future jobs)
 build    sample-data  bandit scan
```

**Status:** ✅ All jobs green on `main`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML — Tree models | scikit-learn, XGBoost |
| ML — Deep learning | TensorFlow / Keras (LSTM) |
| XAI | SHAP, LIME |
| Data | pandas, NumPy, imbalanced-learn (SMOTE) |
| Dashboard | Streamlit |
| PCAP processing | CICFlowMeter wrapper |
| Testing | pytest, pytest-cov |
| Linting | flake8 |
| Security scan | Bandit |
| Containerisation | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Coverage | Codecov |

---

## Model Cards

Detailed model cards covering performance, bias, limitations, and intended use:
- [Random Forest Model Card](docs/model_cards/random_forest_card.md)
- [XGBoost Model Card](docs/model_cards/xgboost_card.md)
- [LSTM Model Card](docs/model_cards/lstm_card.md)

---

## Research Paper

A companion IEEE-format research paper is included in [`paper/`](paper/):

> *"Explainable AI for Network Intrusion Detection: Combining SHAP and LIME for Transparent SOC Triage"*  
> Chandra Sekhar Chakraborty — April 2026

The paper covers methodology, dataset handling, model selection rationale, explainability framework design, and evaluation results.

---

## FAQ

**Q: Do I need the full CICIDS-2017 dataset to run the project?**  
No. `scripts/generate_samples.py` creates synthetic data with the same 78-feature schema. The dashboard, tests, and explainers all work without the real dataset.

**Q: Which model should I use in production?**  
XGBoost offers the best balance — lowest FPR (0.21%), fastest inference (~0.2ms), and SHAP support via `TreeExplainer`. Random Forest is more interpretable. LSTM is best if temporal sequence context matters.

**Q: Can I feed live PCAP traffic into this?**  
Yes. `src/utils/pcap_converter.py` wraps CICFlowMeter to convert PCAP files to feature CSVs compatible with the pipeline.

**Q: How do I retrain on my own network data?**  
Label your flows with the same 14-class schema, place CSVs in `data/raw/`, and run `bash scripts/run_pipeline.sh`. The pipeline handles preprocessing, training, and artefact export.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

In short: fork → create a branch (`feat/your-feature`) → make changes → run `pytest` → open a PR against `main`.

All PRs are automatically linted, tested, and security-scanned by CI before review.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Chandra Sekhar Chakraborty**  
Cybersecurity Analyst | SOC Analyst Aspirant | B.Tech CSE — Graduating 2026  
📍 West Bengal, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/chandrasekhar-chakraborty)
[![GitHub](https://img.shields.io/badge/GitHub-ChandraVerse-181717?logo=github)](https://github.com/ChandraVerse)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-4CAF50)](https://chandraverse.github.io/chandraverse-portfolio/)

---

<div align="center">

*Built with 🛡️ for the defensive security community — April 2026*

</div>
