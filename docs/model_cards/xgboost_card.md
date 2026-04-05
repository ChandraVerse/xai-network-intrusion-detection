# Model Card — XGBoost Classifier

## Model Details

| Property | Value |
|----------|-------|
| **Model Type** | XGBoost (Gradient Boosted Trees) |
| **Framework** | xgboost 2.0+ |
| **Serialization** | `joblib` → `models/xgboost_model.pkl` |
| **Version** | 1.0 (April 2026) |
| **Author** | Chandra Sekhar Chakraborty |
| **License** | MIT |

### Hyperparameters

```python
XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
```

---

## Intended Use

### Primary Use
High-precision classification of network traffic flows for SOC deployments where **minimizing false positives** is the top priority. Best suited for environments where analyst alert fatigue is the primary concern.

### Intended Users
- SOC analysts who need high-confidence alerts with minimal noise
- DevSecOps pipelines requiring automated alert triage with low FPR
- Security researchers benchmarking boosting-based NIDS

### Out-of-Scope Use
- Same exclusions as Random Forest card — see [random_forest_card.md](random_forest_card.md)
- Not intended for raw packet-level analysis

---

## Training Data

Identical to Random Forest — see [random_forest_card.md](random_forest_card.md#training-data) for full details.

| Property | Detail |
|----------|--------|
| **Dataset** | CICIDS-2017 |
| **Split** | 80% train / 20% test, stratified |
| **Balancing** | SMOTE on training set only |
| **Scaling** | MinMaxScaler (train-fit only) |

---

## Evaluation Results

> Evaluated on 20% stratified holdout test set.

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.91% |
| **Macro F1** | 0.994 |
| **Macro Precision** | 0.995 |
| **Macro Recall** | 0.993 |
| **False Positive Rate** | **0.21%** |
| **Inference Time** | ~0.2 ms per flow |

> ⭐ **Lowest false positive rate of all three models (0.21%)** — recommended for high-precision SOC deployments.

### Strengths
- **Lowest FPR** — fewest benign flows incorrectly flagged as attacks
- **Fastest inference** — ~0.2 ms per flow, suitable for near-real-time pipelines
- Handles class imbalance natively via gradient weighting
- Built-in early stopping prevents overfitting
- SHAP `TreeExplainer` with interaction values for feature-pair analysis

### Limitations
- Slightly lower macro F1 than Random Forest (0.994 vs 0.997)
- More hyperparameter-sensitive than RF — requires tuning for new datasets
- Does not model temporal flow sequences (see LSTM card)
- Prone to memorizing minority-class SMOTE artifacts if `max_depth` is too high

---

## Explainability

**SHAP Method:** `TreeExplainer` with `approximate=False` for exact interaction values

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap_interaction = explainer.shap_interaction_values(X_sample)  # feature pair analysis
```

**Top 5 Features by Mean |SHAP| (Global):**
1. `Flow Duration`
2. `Flow Bytes/s`
3. `Fwd Packet Length Max`
4. `Bwd IAT Total`
5. `Destination Port`

---

## Ethical Considerations & Bias

- **Same dataset-level biases as RF** — 2017 lab captures, modern attack gaps
- **FPR advantage is relative** — 0.21% still means ~2,100 false alerts per million flows in a high-traffic environment
- **Feature interaction values** (via `shap_interaction_values`) expose which feature *pairs* drive detections — useful for auditing potential proxy discrimination on port/protocol combinations
- **Recommendation:** Always surface SHAP explanations to analysts rather than delivering bare labels

---

## Caveats & Recommendations

- Use as the **primary production model** when FPR is the dominant operational constraint
- Combine with Random Forest via soft-voting ensemble for best overall F1 + FPR balance
- Retrain every 6–12 months or when new attack categories emerge
- Log all predictions for drift monitoring

---

*Model card follows [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) model card framework.*
