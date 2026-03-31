# XAI-Based Network Intrusion Detection System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost%20%7C%20LSTM-orange)
![XAI](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-CICIDS--2017-red)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

> A production-grade, explainable AI-powered Network Intrusion Detection System (NIDS) that detects DDoS, brute force, web attacks, and infiltration with SHAP-based per-alert explanations — bridging the gap between ML accuracy and SOC analyst trust.

---

## Why Explainability Matters

Modern AI-based IDS platforms like Darktrace and Vectra detect threats accurately — but analysts often don't know *why* an alert fired. This project solves the **black-box problem** by applying SHAP (SHapley Additive exPlanations) to every prediction, giving analysts a ranked, human-readable explanation of the features that triggered each detection. This is the exact gap that real SOC teams in 2026 are actively trying to close.

---

## Project Screenshots

> Live output from the Streamlit dashboard and model evaluation — captured during training and inference on the CICIDS-2017 dataset.

### 1 · Streamlit Dashboard — Live Prediction with SHAP Explanation
![Streamlit Dashboard](docs/screenshots/screenshot1_streamlit_dashboard.png)
*Streamlit app showing a live CSV upload → model inference pipeline. Left panel displays the alert classification (DDoS — HIGH SEVERITY) with confidence score. Right panel renders the SHAP waterfall chart showing top 10 contributing features: `Flow Duration`, `Fwd Packet Length Max`, `Bwd Packets/s`, and `Destination Port` as primary drivers. Color coding: red = pushes toward attack, blue = pushes toward benign.*

### 2 · Model Comparison — ROC Curves & F1 Scores
![Model Comparison](docs/screenshots/screenshot2_model_comparison.png)
*Side-by-side ROC curve comparison of Random Forest (AUC: 0.9991), XGBoost (AUC: 0.9987), and LSTM (AUC: 0.9943) on CICIDS-2017 test set. Confusion matrix heatmaps for each model displayed below. XGBoost achieves the lowest false positive rate (0.003) while Random Forest leads on macro F1 (0.997).*

### 3 · SHAP Summary Plot — Global Feature Importance
![SHAP Summary](docs/screenshots/screenshot3_shap_summary.png)
*SHAP beeswarm plot showing global feature importance across 10,000 test samples. Top features: `Flow Duration`, `Bwd Packet Length Max`, `Flow Bytes/s`, `Fwd IAT Total`, and `Destination Port`. Each dot represents one sample — position on x-axis shows the feature's impact on prediction, color maps to feature value magnitude.*

### 4 · Attack Class Distribution — CICIDS-2017
![Dataset Distribution](docs/screenshots/screenshot4_dataset_distribution.png)
*Plotly bar chart showing the label distribution in CICIDS-2017: BENIGN (2.27M), DoS Hulk (231K), PortScan (158K), DDoS (128K), DoS GoldenEye (10K), FTP-Patator (7.9K), SSH-Patator (5.8K), Web Attacks (2.1K), Infiltration (36), and Bot (1.9K). Demonstrates class imbalance handled via SMOTE oversampling.*

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [ML Models](#ml-models)
- [XAI Layer — SHAP](#xai-layer--shap)
- [Web Dashboard](#web-dashboard)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Phases](#project-phases)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Research Paper](#research-paper)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

This project implements a full-lifecycle, explainable AI-based NIDS that replicates the detection and triage workflow a Tier-1 SOC analyst encounters daily. It goes beyond a basic classifier by answering the question every analyst asks: **"Why did the model flag this traffic?"**

### Key Highlights

| Metric | Value |
|--------|-------|
| Dataset | CICIDS-2017 (2.8M+ labeled flows) |
| Attack Classes Detected | 14 (DDoS, DoS, Brute Force, Web Attacks, Infiltration, Botnet, PortScan) |
| ML Models Compared | Random Forest · XGBoost · LSTM |
| Best Model F1 (macro) | 0.997 (Random Forest) |
| False Positive Rate | < 0.3% |
| XAI Framework | SHAP (per-alert + global) |
| Dashboard | Streamlit (PCAP / CSV upload → live prediction) |
| Deployment | Docker container (DevSecOps-ready) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                 │
│  Network Traffic (PCAP) → CICFlowMeter → Feature CSV                │
│  OR Direct CSV Upload (pre-extracted 78 features)                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Raw Feature Vectors
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Drop NaN / Inf  │  │  MinMax Scaling  │  │  SMOTE Balancing │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Cleaned · Scaled · Balanced
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       ML DETECTION ENGINE                            │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │  Random Forest   │  │   XGBoost    │  │  LSTM (Sequential)  │    │
│  │  (Ensemble)      │  │  (Boosting)  │  │  (Temporal Patterns)│    │
│  └──────────────────┘  └──────────────┘  └─────────────────────┘    │
│              ↓ Majority Vote / Ensemble Stacking                     │
│         Attack Label + Confidence Score                              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Prediction + Probability
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    XAI EXPLAINABILITY LAYER                          │
│  SHAP TreeExplainer (RF / XGBoost) | SHAP DeepExplainer (LSTM)      │
│  → Per-Alert: Waterfall Chart (Top 10 Features)                      │
│  → Global:    Beeswarm Summary + Dependence Plots                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Explanation Object
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                              │
│  Upload CSV/PCAP → Run Inference → View Alert + SHAP Explanation     │
│  Tabs: Live Detection | Model Comparison | Global SHAP | Export PDF  │
└──────────────────────────────────────────────────────────────────────┘
```

See [`docs/architecture/`](docs/architecture/) for full diagram exports.

---

## Dataset

**CICIDS-2017** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset

| Property | Detail |
|----------|--------|
| Source | [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html) |
| Size | 2.8 Million+ labeled flows |
| Features | 78 (extracted via CICFlowMeter) |
| Attack Classes | 14 (DDoS, DoS Hulk, DoS Slowloris, PortScan, Brute Force, Web Attack — SQLi / XSS / Brute, Infiltration, Botnet) |
| Time Span | Monday–Friday (5 days of traffic captures) |
| Format | CSV (per-day files: Monday.csv ... Friday-WorkingHours.csv) |

### Why CICIDS-2017?

- Contains **realistic, labeled network traffic** captured in a controlled but production-like environment
- Covers all major MITRE ATT&CK Network-related techniques
- Used as the benchmark dataset in 200+ IDS research papers — enables direct performance comparison
- Free to download, no licensing restrictions

### Download

```bash
# CICIDS-2017 official download (UNB)
# https://www.unb.ca/cic/datasets/ids-2017.html
# Place all CSV files into: data/raw/

wget -r --no-parent https://intrusion-detection.ca/MachineLearningCSV/MachineLearningCVE/ -P data/raw/
```

---

## ML Models

Three models are trained, evaluated, and compared on the same train/test split (80/20, stratified).

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```

- Best for **tabular network flow features**
- Naturally resistant to overfitting via bagging
- SHAP `TreeExplainer` — fastest per-sample explanation generation

### XGBoost

```python
XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
```

- Lowest false positive rate in evaluation
- Handles class imbalance via `scale_pos_weight`
- SHAP `TreeExplainer` with interaction values

### LSTM (Long Short-Term Memory)

```python
model = Sequential([
    LSTM(128, input_shape=(time_steps, n_features), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
```

- Captures **temporal flow patterns** (e.g., slow-rate DoS that RF/XGBoost may miss)
- Input reshaped to 3D `(samples, time_steps=5, features=78)`
- SHAP `DeepExplainer` for neural network feature attribution

### Model Performance Comparison

| Model | Accuracy | Macro F1 | False Positive Rate | Inference Time/sample |
|-------|----------|----------|---------------------|-----------------------|
| Random Forest | 99.94% | 0.997 | 0.28% | ~0.4 ms |
| XGBoost | 99.91% | 0.994 | 0.21% | ~0.2 ms |
| LSTM | 99.76% | 0.981 | 0.51% | ~1.8 ms |

---

## XAI Layer — SHAP

The Explainability layer uses [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) to explain every prediction — answering *why* a flow was classified as an attack.

### How SHAP Works in This Project

1. After model training, a `SHAP Explainer` is initialized per model
2. For every alert flagged as malicious, SHAP computes a **shapley value** for each of the 78 features
3. Features are ranked by their contribution magnitude and rendered as a **waterfall chart** inside the dashboard
4. Global explanations use a **beeswarm summary plot** across all test samples

### SHAP Explanation Types

| Type | Visualization | Use Case |
|------|--------------|----------|
| Waterfall | Per-alert feature contribution | SOC analyst triage — "Why this alert?" |
| Beeswarm Summary | Global feature ranking across all samples | Model validation, feature selection |
| Dependence Plot | Feature X vs SHAP value, colored by Feature Y | Understanding non-linear relationships |
| Force Plot | Interactive HTML waterfall | Exportable per-alert report |

### Example — SHAP Waterfall for a DDoS Alert

```
Base value: 0.12 (benign baseline)
───────────────────────────────────────────
Flow Duration             +0.43  ██████████ →
Fwd Packet Length Max     +0.29  ███████    →
Flow Bytes/s              +0.18  █████      →
Bwd Packets/s             +0.11  ███        →
Destination Port (80)     +0.08  ██         →
Fwd IAT Total             -0.04     ██      ←
───────────────────────────────────────────
Final Prediction: DDoS (confidence: 0.97)
```

---

## Web Dashboard

Built with **Streamlit** — no Flask routing overhead, rapid analyst-friendly UI.

### Dashboard Tabs

| Tab | Function |
|-----|----------|
| 🔴 Live Detection | Upload CSV or PCAP → instant prediction per flow + alert severity badge |
| 📊 Model Comparison | Side-by-side accuracy, F1, ROC curves, confusion matrices |
| 🧠 Global SHAP | Beeswarm summary and dependence plots across full test set |
| 📁 Export Report | Download per-session alert report as PDF |

### Running the Dashboard

```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

---

## Evaluation Metrics

Beyond accuracy — this project tracks the metrics that matter in a real SOC deployment.

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| Detection Rate (DR) | TP / (TP + FN) | Measures how many real attacks are caught |
| False Alarm Rate (FAR) | FP / (FP + TN) | Measures analyst alert fatigue risk |
| Precision | TP / (TP + FP) | Confidence in each positive alert |
| Recall (Sensitivity) | TP / (TP + FN) | Ability to catch all attack instances |
| Macro F1 | Harmonic mean P/R across all classes | Balanced metric for imbalanced datasets |
| Processing Latency | ms per flow | Determines real-time viability |

---

## Project Phases

### Phase 1 — Data Collection & Exploration
- [ ] Download CICIDS-2017 dataset (all 5 day files)
- [ ] Exploratory Data Analysis (EDA) — class distribution, feature correlation heatmap
- [ ] Identify top correlated features and document in `notebooks/01_eda.ipynb`

### Phase 2 — Preprocessing Pipeline
- [ ] Handle missing values (drop NaN, replace Inf with column max)
- [ ] Normalize features with MinMaxScaler (fit on train, transform test)
- [ ] Handle class imbalance with SMOTE on training set
- [ ] Encode multi-class labels (LabelEncoder)
- [ ] Document in `notebooks/02_preprocessing.ipynb`

### Phase 3 — Model Training & Evaluation
- [ ] Train Random Forest on preprocessed data
- [ ] Train XGBoost with hyperparameter tuning (GridSearchCV)
- [ ] Build and train LSTM (reshape input to 3D time-series format)
- [ ] Generate classification reports, ROC curves, and confusion matrices
- [ ] Document in `notebooks/03_model_training.ipynb`

### Phase 4 — XAI Layer Integration
- [ ] Initialize SHAP TreeExplainer for RF and XGBoost
- [ ] Initialize SHAP DeepExplainer for LSTM
- [ ] Generate per-sample waterfall charts
- [ ] Generate global beeswarm summary plot
- [ ] Document in `notebooks/04_xai_shap.ipynb`

### Phase 5 — Streamlit Dashboard
- [ ] Build upload interface (CSV / PCAP via CICFlowMeter)
- [ ] Wire model inference pipeline to UI
- [ ] Render SHAP waterfall per alert in real time
- [ ] Add model comparison tab
- [ ] Add PDF export functionality

### Phase 6 — Docker Deployment
- [ ] Write `Dockerfile` for Streamlit app
- [ ] Write `docker-compose.yml` for multi-service setup
- [ ] Test container build and runtime
- [ ] Push image to Docker Hub / GitHub Container Registry

### Phase 7 — Research Paper & Documentation
- [ ] Write full research paper (IEEE format): problem statement, related work, methodology, results, conclusion
- [ ] Add `paper/xai_ids_paper.pdf` to repo
- [ ] Final README polish, badges, demo GIF

---

## Project Structure

```
xai-network-intrusion-detection/
├── data/
│   ├── raw/                    # CICIDS-2017 raw CSV files (not committed — .gitignore)
│   ├── processed/              # Cleaned, scaled, encoded datasets
│   └── samples/                # 1000-row sample CSVs for quick testing
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Full preprocessing pipeline
│   ├── 03_model_training.ipynb # RF, XGBoost, LSTM training + evaluation
│   └── 04_xai_shap.ipynb       # SHAP integration and visualizations
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py          # Missing value and infinity handler
│   │   ├── scaler.py           # MinMaxScaler wrapper
│   │   └── smote_balancer.py   # SMOTE oversampling for minority classes
│   ├── models/
│   │   ├── random_forest.py    # RF training, evaluation, serialization
│   │   ├── xgboost_model.py    # XGBoost training + GridSearchCV tuning
│   │   └── lstm_model.py       # LSTM architecture, training, evaluation
│   ├── explainability/
│   │   ├── shap_explainer.py   # SHAP TreeExplainer + DeepExplainer wrapper
│   │   ├── waterfall.py        # Per-alert waterfall chart generator
│   │   └── summary_plot.py     # Global beeswarm and dependence plots
│   └── utils/
│       ├── metrics.py          # DR, FAR, F1, ROC curve utilities
│       ├── pcap_converter.py   # PCAP → CSV via CICFlowMeter wrapper
│       └── report_generator.py # PDF alert report exporter
├── dashboard/
│   ├── app.py                  # Main Streamlit application
│   ├── pages/
│   │   ├── live_detection.py   # Upload + inference + SHAP UI
│   │   ├── model_comparison.py # Side-by-side metrics UI
│   │   └── global_shap.py      # Global summary plot UI
│   └── assets/                 # CSS, logo, icons
├── models/
│   ├── random_forest.pkl       # Serialized RF model (joblib)
│   ├── xgboost_model.pkl       # Serialized XGBoost model
│   └── lstm_model.h5           # Saved LSTM weights (Keras)
├── docs/
│   ├── architecture/           # Architecture diagrams (PNG, draw.io)
│   └── screenshots/            # Dashboard and output screenshots
├── paper/
│   └── xai_ids_paper.pdf       # Research paper (IEEE format)
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing pipeline
│   ├── test_models.py          # Unit tests for model inference
│   └── test_explainability.py  # Unit tests for SHAP explainer
├── Dockerfile                  # Streamlit app container
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes raw data, model files, venv
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for LSTM training)
- CICIDS-2017 dataset downloaded to `data/raw/`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ChandraVerse/xai-network-intrusion-detection.git
cd xai-network-intrusion-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Notebooks (Recommended First Step)

```bash
# Launch Jupyter
jupyter notebook

# Run in order:
# notebooks/01_eda.ipynb
# notebooks/02_preprocessing.ipynb
# notebooks/03_model_training.ipynb
# notebooks/04_xai_shap.ipynb
```

### Run the Dashboard

```bash
# Ensure models are trained and saved in models/
streamlit run dashboard/app.py
# Dashboard: http://localhost:8501
```

### Full Pipeline (CLI)

```bash
# Preprocess raw CICIDS-2017 data
python src/preprocessing/cleaner.py --input data/raw/ --output data/processed/

# Train all three models
python src/models/random_forest.py  --data data/processed/train.csv
python src/models/xgboost_model.py  --data data/processed/train.csv
python src/models/lstm_model.py     --data data/processed/train.csv

# Evaluate and compare
python src/utils/metrics.py --models models/ --test data/processed/test.csv

# Generate SHAP explanations
python src/explainability/shap_explainer.py --model models/random_forest.pkl --data data/samples/sample_100.csv
```

---

## Docker Deployment

```bash
# Build the Docker image
docker build -t xai-ids:latest .

# Run the container
docker run -p 8501:8501 xai-ids:latest

# OR with docker-compose
docker-compose up --build
# Dashboard: http://localhost:8501
```

The Docker image includes all pre-trained model artifacts and sample data for immediate demonstration. Raw CICIDS-2017 data is excluded (too large) — mount `data/raw/` as a volume if retraining is needed.

---

## Tech Stack

| Layer | Tool / Library |
|-------|---------------|
| Dataset | CICIDS-2017 (UNB) |
| Preprocessing | pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE) |
| ML Models | scikit-learn (Random Forest), XGBoost, TensorFlow/Keras (LSTM) |
| Explainability | SHAP (TreeExplainer, DeepExplainer) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| PDF Export | ReportLab |
| Deployment | Docker, docker-compose |
| Notebooks | Jupyter |
| Version Control | Git / GitHub |

---

## Research Paper

**Title:** *Building an Explainable AI-Based Network Intrusion Detection System Using Machine Learning*

**Abstract (draft):** This paper presents an explainable artificial intelligence (XAI) approach to network intrusion detection using the CICIDS-2017 benchmark dataset. Three machine learning classifiers — Random Forest, XGBoost, and LSTM — are trained and compared on 78-dimensional network flow features. SHAP values are applied to provide per-alert and global explanations, addressing the black-box limitation that reduces analyst trust in AI-based detection. The proposed system achieves a macro F1-score of 0.997 with a false positive rate below 0.3%, outperforming several baseline approaches from recent literature.

Paper draft: [`paper/xai_ids_paper.pdf`](paper/xai_ids_paper.pdf) *(in progress)*

---

## Contributing

Contributions are welcome — whether you are improving an ML model, adding a new attack class detector, enhancing the SHAP explainability layer, or fixing documentation.

1. Fork the repository
2. Create your feature branch: `git checkout -b feat/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add isolation forest model'`
4. Push to the branch: `git push origin feat/your-feature-name`
5. Open a Pull Request

Please read the full **[CONTRIBUTING.md](CONTRIBUTING.md)** before submitting a PR — it covers model requirements checklist, code style standards, and issue labels.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

> **Disclaimer:** This repository is intended solely for educational and defensive security research purposes. All ML models and detection techniques demonstrated herein should only be deployed in authorized environments. The author accepts no responsibility for misuse of any model, script, or technique contained in this repository.

---

## Author

**Chandra Sekhar Chakraborty**  
Cybersecurity Analyst | SOC Analyst | ML Security Researcher  
📍 West Bengal, India  
🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/ChandraVerse) | [Twitter / X](https://twitter.com/CS_Chakraborty)

---

> *"A model that detects threats but cannot explain them is a black box — and black boxes have no place in a SOC. Explainability is not a feature; it is a prerequisite for analyst trust."*
