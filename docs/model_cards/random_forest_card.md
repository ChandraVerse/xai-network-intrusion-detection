# Model Card — Random Forest Classifier

**Author:** Chandra Sekhar Chakraborty  
**Date:** 2026-04-05  
**Version:** 1.0.0  
**Task:** Multi-class Network Intrusion Detection (14 classes)

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest (sklearn `RandomForestClassifier`) |
| Estimators | 200 trees |
| Max depth | None (fully grown) |
| Min samples split | 2 |
| Feature importance | Gini impurity |
| Serialisation | `joblib` pickle — `models/random_forest.pkl` |
| File size | ~280 KB (demo-scale; full CICIDS model ~2 GB) |

## Intended Use

- **Primary use:** Real-time network flow classification in a SOC analyst dashboard.
- **Secondary use:** SHAP explainability baseline for comparing feature importance against XGBoost.
- **Out-of-scope:** Deployment on raw PCAP without preprocessing pipeline. Production IDS with sub-millisecond latency requirements.

## Training Data

| Property | Value |
|----------|-------|
| Dataset | CICIDS-2017 (Canadian Institute for Cybersecurity) |
| Features | 78 network flow features (CICFlowMeter) |
| Classes | 14 (BENIGN + 13 attack types) |
| Train split | 80 % stratified |
| Class balancing | SMOTE on minority classes |
| Preprocessing | MinMaxScaler, NaN/Inf removal |

## Evaluation Metrics (test split)

| Metric | Score |
|--------|-------|
| Accuracy | 99.72 % |
| Macro Precision | 99.68 % |
| Macro Recall | 99.74 % |
| Macro F1 | 99.71 % |
| Inference (batch=100) | ~0.8 ms |

## Limitations & Bias

- Trained on 2017 traffic captures; **may not generalise to post-2020 attack vectors** (e.g., supply-chain attacks, novel ransomware C2 patterns).
- CICIDS-2017 is lab-generated traffic — real-world noise and encrypted flows may degrade performance.
- Rare classes (Infiltration: 36 samples) are underrepresented; recall for those classes is lower.
- SMOTE creates synthetic minority samples which may not reflect real attack distributions.

## Explainability

- `shap.TreeExplainer` is fully compatible — exact SHAP values, no approximation.
- Top-3 globally influential features: `Flow Duration`, `Fwd Pkt Len Mean`, `Bwd Pkt Len Mean`.
- See `notebooks/04_shap_explainability.ipynb` for full SHAP analysis.

## Ethical Considerations

- Model decisions should be reviewed by a human SOC analyst before any network block action.
- False positives on BENIGN traffic could cause service disruption.
- Not intended for surveillance of individual users.
