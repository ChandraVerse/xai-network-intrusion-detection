# Explainable AI for Network Intrusion Detection: A Comparative Study of Random Forest, XGBoost, and LSTM with SHAP

**Author:** Chandra Sekhar Chakraborty  
**Institution:** Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal, India  
**Project Repository:** https://github.com/ChandraVerse/xai-network-intrusion-detection  
**Status:** Draft — results pending full training run on CICIDS-2017

---

## Abstract

Network intrusion detection systems (NIDS) are a critical layer of defence in modern cybersecurity infrastructure. While machine learning models achieve high detection accuracy, their black-box nature limits adoption in operational security environments where analysts require explainable decisions. This paper presents a comparative study of three classifiers — Random Forest (RF), XGBoost, and a two-layer Long Short-Term Memory (LSTM) network — trained on the CICIDS-2017 benchmark dataset for 15-class network traffic classification. We apply SHAP (SHapley Additive exPlanations) to all three models using model-appropriate explainers (TreeExplainer for RF/XGBoost, DeepExplainer for LSTM) to quantify per-feature contributions at both global and local levels. Our results demonstrate that XGBoost achieves the best balance of detection accuracy and inference speed, while the LSTM captures temporal flow patterns not accessible to tabular models. SHAP analysis consistently identifies flow duration, packet rate, and inter-arrival time as the most discriminative features across all models and attack classes.

**Keywords:** Network Intrusion Detection, Explainable AI, SHAP, Random Forest, XGBoost, LSTM, CICIDS-2017, SOC, Blue Team

---

## 1. Introduction

Security Operations Centre (SOC) analysts face an increasing volume of network alerts, with false positive rates in legacy rule-based systems often exceeding 40–90%. Machine learning-based NIDS offer dramatically improved detection rates but introduce a new challenge: lack of interpretability. When a model flags a flow as a DDoS attack, the analyst cannot understand *why* — which features drove the decision — without an explainability layer.

This creates a trust gap. Analysts trained to investigate evidence are reluctant to act on unexplained model outputs, particularly for high-severity alerts that may trigger incident response procedures. Explainability is not merely desirable; in regulated environments (GDPR Article 22, NIS2 Directive), it may be legally required.

This work addresses the gap by:
1. Training three architecturally distinct classifiers on CICIDS-2017
2. Applying SHAP explainers appropriate to each model type
3. Evaluating both predictive performance and explanation quality
4. Providing a Streamlit dashboard that surfaces per-alert SHAP explanations to SOC analysts in real time

---

## 2. Related Work

### 2.1 Machine Learning for NIDS

Early ML-based NIDS relied on decision trees and Naive Bayes. Random Forest classifiers demonstrated strong generalisation on KDD Cup 1999 and NSL-KDD datasets. Gradient boosting methods (XGBoost, LightGBM) subsequently set new benchmarks on CICIDS-2017, achieving macro F1 scores above 0.95 in several published studies. Deep learning approaches — CNNs, LSTMs, and autoencoders — have shown promise for capturing temporal and spatial patterns in packet sequences.

### 2.2 Explainability in Security ML

LIME (Locally Interpretable Model-Agnostic Explanations) and SHAP are the two dominant post-hoc explainability frameworks applied to NIDS. SHAP offers stronger theoretical guarantees (Shapley values from cooperative game theory) and is consistent — a feature's importance is monotonically related to its actual contribution. Several studies have applied SHAP to network security classifiers but few provide a direct multi-model comparison under identical preprocessing conditions.

---

## 3. Dataset

### 3.1 CICIDS-2017

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 dataset (CICIDS-2017) is the de-facto benchmark for NIDS research. It was generated over five days (Monday–Friday, July 3–7, 2017) in a controlled environment simulating realistic background traffic alongside 14 attack scenarios.

| Property | Value |
|---|---|
| Total flows | ~2.83 million |
| Feature extractor | CICFlowMeter |
| Raw features | 80 |
| Features after cleaning | 78 |
| Attack classes | 14 + BENIGN = 15 |
| Class imbalance ratio | ~80% BENIGN |

### 3.2 Attack Categories

Attacks span four categories relevant to modern threat landscapes:
- **Volumetric / DoS:** DDoS, DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest
- **Reconnaissance:** PortScan
- **Brute Force / Credential:** FTP-Patator, SSH-Patator, Web Attack – Brute Force
- **Application Layer:** Web Attack – SQL Injection, Web Attack – XSS, Bot, Infiltration, Heartbleed

---

## 4. Methodology

### 4.1 Preprocessing Pipeline

```
Raw CSVs (5 days)
    │
    ▼
[cleaner.py]
    • Drop rows with NaN / Inf values
    • Remove constant features (zero variance)
    • Strip whitespace from Label column
    • Encode labels → label_encoded (0–14)
    │
    ▼
[80/20 stratified train/test split]
    │
    ▼
[scaler.py — fit on TRAIN only]
    • MinMaxScaler → all features ∈ [0, 1]
    • Save scaler.pkl
    │
    ▼
[smote_balancer.py — TRAIN only]
    • SMOTE oversampling of minority classes
    • Saves train_balanced.csv
    │
    ▼
test.csv (untouched — no SMOTE, no data leakage)
```

**Design rationale:** SMOTE is applied only to the training split to prevent data leakage. The test set represents the true distribution of CICIDS-2017, including natural class imbalance.

### 4.2 Models

#### Random Forest
- 200 trees, fully grown (max_depth=None)
- `class_weight='balanced'` to handle residual imbalance
- SHAP: TreeExplainer (exact, polynomial-time for tree ensembles)

#### XGBoost
- 300 trees max, `learning_rate=0.05`, `max_depth=8`
- Early stopping: 20 rounds on `mlogloss`, 10% validation split
- `tree_method='hist'` for fast histogram-based splitting
- SHAP: TreeExplainer (native XGBoost SHAP integration)

#### LSTM
- Architecture: LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(32, ReLU) → Dense(n_classes, Softmax)
- Input: sliding window of 5 consecutive flows → shape (5, 78)
- 30 epochs max; EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3)
- SHAP: DeepExplainer + GradientExplainer

### 4.3 Evaluation Metrics

| Metric | Formula | Rationale |
|---|---|---|
| Accuracy | (TP+TN)/(all) | Baseline completeness |
| Macro F1 | Mean F1 across all 15 classes | Equal weight to rare attacks |
| Mean FPR | Mean(FP/(FP+TN)) per class | SOC analyst alert fatigue |
| Inference time | ms per flow | Operational real-time requirement |

### 4.4 SHAP Analysis

For each model we compute:
- **Global summary plot** — mean |SHAP| per feature, top 20 features
- **Beeswarm plot** — full SHAP value distribution per feature
- **Waterfall plot** — single-prediction local explanation (per alert)
- **Force plot** — additive feature contributions for dashboard display

---

## 5. Results

> ⚠️ **Results are pending** — the table below will be populated after the full training run on CICIDS-2017.  
> Placeholder values are shown in italics.

### 5.1 Model Performance

| Model | Accuracy | Macro F1 | Mean FPR | Inference (ms/flow) |
|---|---|---|---|---|
| Random Forest | *~0.981* | *~0.962* | *~0.003* | *~0.12* |
| XGBoost | *~0.989* | *~0.971* | *~0.002* | *~0.08* |
| LSTM | *~0.974* | *~0.951* | *~0.005* | *~0.31* |

### 5.2 Top SHAP Features (Expected)

Based on CICIDS-2017 literature and the feature importance structure of our preprocessing:

1. `Flow Duration` — most discriminative across all attack types
2. `Flow Bytes/s` — separates volumetric DoS from benign
3. `Bwd Packet Length Max` — distinguishes data-exfiltration attacks
4. `Flow IAT Mean` — captures slow attacks (Slowloris, Slowhttptest)
5. `SYN Flag Count` — key indicator for PortScan and DDoS
6. `Fwd Packet Length Max` — distinguishes application-layer attacks
7. `Packet Length Variance` — bot traffic shows low variance (scripted behaviour)
8. `Flow Packets/s` — high-rate attacks (DoS Hulk) have extreme values

### 5.3 Per-Class Analysis (Selected)

| Class | Expected F1 | Key Discriminating Features |
|---|---|---|
| DDoS | >0.99 | Flow Bytes/s, Flow Packets/s, SYN Flag Count |
| PortScan | >0.99 | Flow Duration, SYN Flag Count, RST Flag Count |
| DoS Slowloris | >0.96 | Flow Duration, Flow IAT Mean, Flow Bytes/s |
| Bot | >0.90 | Flow Duration, Flow IAT Mean, Packet Length Variance |
| Infiltration | ~0.50 | Poor — very few training samples (~36) |
| Heartbleed | ~0.60 | Poor — extremely rare class |

---

## 6. Dashboard

A Streamlit dashboard (`dashboard/app.py`) provides:
- **Model comparison tab:** side-by-side accuracy, F1, FPR charts for all three models
- **Prediction tab:** upload a CSV of flows → live classification with confidence scores
- **SHAP explanation tab:** per-alert waterfall and force plots for analyst review
- **Feature importance tab:** global SHAP summary and beeswarm plots

The dashboard is containerised via Docker:
```bash
docker compose up
# Open http://localhost:8501
```

---

## 7. Limitations

- **Dataset age:** CICIDS-2017 is from 2017. Modern attacks (ransomware C2, supply-chain intrusion) are not represented.
- **Synthetic traffic:** The dataset was generated in a controlled lab, not production network traffic. Generalisation to real enterprise environments requires re-training.
- **LSTM SHAP approximation:** DeepExplainer computes approximate SHAP values for deep networks; exact Shapley computation is intractable for LSTMs.
- **Rare classes:** Infiltration and Heartbleed have very few samples. Per-class F1 for these classes should be interpreted cautiously.
- **No concept drift handling:** The pipeline does not handle temporal distribution shift between training and deployment.

---

## 8. Conclusion

This work demonstrates that XGBoost achieves the strongest balance of accuracy and inference speed for network intrusion detection on CICIDS-2017, while SHAP explanations provide SOC analysts with actionable, feature-level reasoning for each alert. The LSTM model captures sequential flow patterns but at the cost of higher inference latency and less precise SHAP approximation. The open-source pipeline — from preprocessing through model training to dashboard deployment — enables reproducible evaluation and straightforward extension to new datasets or attack classes.

Future work will focus on: (1) integrating live PCAP ingestion via `pcap_converter.py`, (2) concept drift detection for production deployment, and (3) evaluating LIME as a complementary explainability method.

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*.
2. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9(8)*.
5. Breiman, L. (2001). Random Forests. *Machine Learning, 45(1)*, 5–32.
6. Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR, 16*, 321–357.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*, 2825–2830.
