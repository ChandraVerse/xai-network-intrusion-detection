# Model Card — Random Forest Classifier

## Model Details

| Property | Value |
|----------|-------|
| **Model Type** | Random Forest (Ensemble of Decision Trees) |
| **Framework** | scikit-learn 1.4+ |
| **Serialization** | `joblib` → `models/random_forest.pkl` |
| **Version** | 1.0 (April 2026) |
| **Author** | Chandra Sekhar Chakraborty |
| **License** | MIT |

### Hyperparameters

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

---

## Intended Use

### Primary Use
Real-time classification of network traffic flows into 15 classes (BENIGN + 14 attack types) using 78 CICFlowMeter-extracted features.

### Intended Users
- SOC analysts using the Streamlit dashboard for alert triage
- Security researchers benchmarking NIDS models against CICIDS-2017
- Students learning ML-based intrusion detection

### Out-of-Scope Use
- **Do not use** as a standalone production firewall without human oversight
- **Do not use** on raw packet payloads — model expects CICFlowMeter flow-level features only
- **Do not use** on encrypted traffic where flow metadata is absent

---

## Training Data

| Property | Detail |
|----------|--------|
| **Dataset** | CICIDS-2017 (Canadian Institute for Cybersecurity) |
| **Total Flows** | ~2.8 million labeled flow records |
| **Features** | 78 numerical (CICFlowMeter: duration, byte counts, flag counts, IAT stats) |
| **Classes** | 15 (BENIGN + 14 attack types) |
| **Split** | 80% train / 20% test, stratified by class |
| **Balancing** | SMOTE applied to training set only (`not majority` strategy) |
| **Scaling** | MinMaxScaler fitted on training set only (no data leakage) |
| **Label Encoding** | Integer-encoded via `LabelEncoder` (see `data/processed/label_map.json`) |

### Class Distribution (Training Set, Pre-SMOTE)

| Class | Approximate Count |
|-------|------------------|
| BENIGN | 2,273,097 |
| DoS Hulk | 231,073 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| DoS GoldenEye | 10,293 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS Slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,966 |
| Web Attack — Brute Force | 1,507 |
| Web Attack — XSS | 652 |
| Infiltration | 36 |
| Web Attack — SQL Injection | 21 |

> SMOTE generates synthetic minority samples so all classes are represented proportionally during training.

---

## Evaluation Results

> Evaluated on 20% stratified holdout test set. SMOTE **not** applied to test set.

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.94% |
| **Macro F1** | 0.997 |
| **Macro Precision** | 0.996 |
| **Macro Recall** | 0.998 |
| **False Positive Rate** | 0.28% |
| **Inference Time** | ~0.4 ms per flow |

### Strengths
- Highest macro F1 of the three models (0.997)
- Naturally handles mixed feature scales without normalization
- Fast SHAP `TreeExplainer` — milliseconds per alert explanation
- Bootstrap aggregation prevents overfitting on majority class

### Limitations
- Larger serialized file size compared to XGBoost
- Slower inference than XGBoost for single-sample predictions
- Does not model temporal sequences (see LSTM card for slow-rate attacks)
- SMOTE-generated minority samples may not fully represent real infiltration traffic

---

## Explainability

**SHAP Method:** `TreeExplainer` (exact Shapley values for tree ensembles)

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)  # shape: (n_samples, n_features, n_classes)
```

**Top 5 Features by Mean |SHAP| (Global):**
1. `Flow Duration`
2. `Bwd Packet Length Max`
3. `Flow Bytes/s`
4. `Fwd IAT Total`
5. `Destination Port`

See `notebooks/04_xai_shap.ipynb` for full beeswarm plots and dependence analysis.

---

## Ethical Considerations & Bias

- **Dataset bias:** CICIDS-2017 was captured in a controlled lab (University of New Brunswick) — not a live enterprise network. Real-world traffic distributions differ; false positive rates may be higher in production.
- **Temporal bias:** All captures are from 2017. Modern attack variants (e.g., encrypted C2, HTTP/3 floods) are not represented.
- **Class imbalance:** Infiltration (36 samples) and Web Attack — SQL Injection (21 samples) are extremely rare. Despite SMOTE, performance on these classes should be interpreted cautiously.
- **No demographic bias:** Network flow features are purely technical (byte counts, timing) — no personal data is involved.
- **Misuse risk:** This model should complement, not replace, human analyst judgment. All high-severity alerts require human review before action.

---

## Caveats & Recommendations

- Retrain periodically as network baselines evolve
- Use alongside XGBoost (lower FPR) for high-precision deployments
- Do not deploy without a human-in-the-loop review workflow for Severity=HIGH alerts
- Monitor for concept drift if deploying on live traffic

---

*Model card follows [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) model card framework.*
