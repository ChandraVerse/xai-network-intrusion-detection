# Model Card — XGBoost Classifier

**Author:** Chandra Sekhar Chakraborty  
**Date:** 2026-04-05  
**Version:** 1.0.0  
**Task:** Multi-class Network Intrusion Detection (14 classes)

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (`xgb.XGBClassifier`) |
| Estimators | 300 boosting rounds |
| Max depth | 6 |
| Learning rate | 0.1 |
| Objective | `multi:softprob` |
| Serialisation | `joblib` pickle — `models/xgboost_model.pkl` |
| File size | ~492 KB (demo-scale) |

## Intended Use

- **Primary use:** High-speed inference in real-time NIDS pipeline; preferred when latency is critical.
- **Secondary use:** Cross-validation of SHAP feature rankings against Random Forest.
- **Out-of-scope:** Streaming inference on raw bytes without CICFlowMeter feature extraction.

## Training Data

Same as Random Forest card — CICIDS-2017, 78 features, SMOTE-balanced, MinMaxScaler.

## Evaluation Metrics (test split)

| Metric | Score |
|--------|-------|
| Accuracy | 99.68 % |
| Macro Precision | 99.61 % |
| Macro Recall | 99.66 % |
| Macro F1 | 99.63 % |
| Inference (batch=100) | ~0.4 ms |

## Limitations & Bias

- Same temporal generalisation caveat as RF — 2017 dataset.
- Gradient boosting is sensitive to feature scale outliers despite MinMax normalisation.
- `multi:softprob` outputs calibrated probabilities, but confidence scores should be validated with isotonic regression before use as alerting thresholds.

## Explainability

- `shap.TreeExplainer` compatible — exact SHAP values.
- Slightly lower explainability score than RF due to deeper ensemble structure.
- Feature rankings align ~85 % with RF top-20 (see `dashboard/pages/global_shap.py` consensus panel).

## Ethical Considerations

Same as Random Forest card. XGBoost's higher inference speed should not be used to bypass human review for high-stakes blocking decisions.
