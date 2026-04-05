# XAI-Based Network Intrusion Detection: A Comparative Study
## Paper Outline

**Authors**: Chandra Sekhar Chakraborty  
**Dataset**: CICIDS-2017  
**GitHub**: https://github.com/ChandraVerse/xai-network-intrusion-detection

---

## Abstract (Target: 150 words)

Network intrusion detection systems (NIDS) based on machine learning achieve high accuracy but lack transparency, making them difficult to trust and audit in operational Security Operations Centres (SOCs). This paper presents XAI-NIDS, a comparative framework that trains three classifiers — Random Forest, XGBoost, and LSTM — on the CICIDS-2017 benchmark dataset and evaluates two post-hoc explainability methods: SHAP and LIME. We compare models on accuracy, macro F1, false positive rate, and inference latency, and we measure the feature-importance agreement between SHAP and LIME explanations. Our results show that tree-based models (RF, XGBoost) achieve superior F1 and near-real-time inference, while LIME produces more actionable rule-based explanations for SOC analysts. We release a Streamlit dashboard that provides real-time per-alert SHAP waterfall explanations.

---

## 1. Introduction

- Rise of ML-based NIDS and the black-box problem
- Importance of explainability for SOC analyst trust
- Research questions:
  1. Which model (RF / XGBoost / LSTM) best balances F1 and FPR on CICIDS-2017?
  2. Do SHAP and LIME agree on the most important features?
  3. How do explanation methods affect analyst decision time?

## 2. Related Work

- ML for NIDS: survey of tree-based vs deep learning approaches
- XAI methods: SHAP (Lundberg & Lee, 2017), LIME (Ribeiro et al., 2016)
- Previous XAI-NIDS work: cite 5–8 recent papers

## 3. Dataset & Preprocessing

- CICIDS-2017: 78 flow features, 14 classes, class imbalance
- Cleaning: infinite/NaN removal, MinMax scaling
- Class distribution and imbalance handling (class_weight="balanced")

## 4. Model Architectures

### 4.1 Random Forest
- Hyperparameters and training setup
- Feature importance via Gini impurity

### 4.2 XGBoost
- Gradient boosting with histogram method
- Early stopping strategy

### 4.3 LSTM
- Architecture: 2-layer LSTM with dropout
- Sequence construction from flow features

## 5. Explainability Methods

### 5.1 SHAP (SHapley Additive exPlanations)
- TreeExplainer for RF / XGBoost
- DeepExplainer for LSTM
- Global (beeswarm) vs local (waterfall) explanations

### 5.2 LIME (Local Interpretable Model-Agnostic Explanations)
- LimeTabularExplainer configuration
- Neighbourhood perturbation for tabular data
- Comparison with SHAP on top-15 feature agreement

## 6. Experimental Results

- Table 1: Accuracy, Macro F1, Mean FPR, Inference latency
- Figure 1: Confusion matrices (3 models)
- Figure 2: ROC-AUC curves
- Figure 3: SHAP global importance (beeswarm)
- Figure 4: SHAP vs LIME top-15 feature agreement
- Figure 5: Per-alert SHAP waterfall example (DDoS)
- Figure 6: LIME local explanation example (PortScan)

## 7. Dashboard

- Streamlit real-time demo
- SOC workflow integration
- Screenshot and description

## 8. Discussion

- RF and XGBoost: fast, accurate, SHAP-transparent
- LSTM: lower F1 on synthetic data; meaningful only with real sequences
- SHAP vs LIME agreement: X/15 features in common
- Limitations: synthetic data, balanced classes, no live traffic test

## 9. Conclusion

- XGBoost + SHAP recommended for SOC deployment
- LIME adds complementary rule-based explanations
- Future work: real CICIDS-2017 raw CSV, ensemble voting, drift detection

## References

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. KDD.
- Sharafaldin, I., et al. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP.
- [Add 5–8 more relevant citations]
