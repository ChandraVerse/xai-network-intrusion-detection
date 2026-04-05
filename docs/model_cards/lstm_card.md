# Model Card — LSTM Neural Network Classifier

## Model Details

| Property | Value |
|----------|-------|
| **Model Type** | LSTM (Long Short-Term Memory) Recurrent Neural Network |
| **Framework** | TensorFlow / Keras 2.15+ |
| **Serialization** | `models/lstm_model.tar.gz` (contains `lstm_model.h5` weights + architecture) |
| **Version** | 1.0 (April 2026) |
| **Author** | Chandra Sekhar Chakraborty |
| **License** | MIT |

### Architecture

```python
model = Sequential([
    LSTM(128, input_shape=(time_steps=5, n_features=78), return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes=15, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| LSTM-1 (128 units) | (batch, 5, 128) | 106,496 |
| Dropout (0.3) | (batch, 5, 128) | 0 |
| LSTM-2 (64 units) | (batch, 64) | 49,408 |
| Dropout (0.3) | (batch, 64) | 0 |
| Dense-1 (32, ReLU) | (batch, 32) | 2,080 |
| Dense-out (15, Softmax) | (batch, 15) | 495 |
| **Total** | | **~158K trainable params** |

---

## Intended Use

### Primary Use
Detection of **temporally-structured attack patterns** — slow-rate DoS attacks (Slowloris, GoldenEye, Slowhttptest) and botnet C2 communications that build gradually over multiple consecutive flows and are missed by single-snapshot classifiers.

### How Temporal Input Works

Each sample is a **sliding window of 5 consecutive flows** from the same source IP:

```
Input shape: (batch_size, time_steps=5, features=78)

  t-4   t-3   t-2   t-1   t
  [f1]  [f2]  [f3]  [f4]  [f5]  → LSTM → "DoS Slowloris"
```

Flows are sorted by timestamp and grouped by `(Src IP, Dst IP, Dst Port)` tuple before reshaping. This is the key architectural difference from RF and XGBoost.

### Intended Users
- Security researchers studying temporal pattern detection in network flows
- Deployment environments where slow-rate attacks are a primary concern
- ML engineers exploring SHAP DeepExplainer for RNN architectures

### Out-of-Scope Use
- **Do not use** where single-flow, low-latency (<1 ms) inference is required — use XGBoost instead
- **Do not use** without a 5-flow sliding window buffer — single isolated flows cannot be processed
- Not suitable for protocols where flow order is meaningless

---

## Training Data

| Property | Detail |
|----------|--------|
| **Dataset** | CICIDS-2017 |
| **Input Reshaping** | Flows grouped by (Src IP, Dst IP, Dst Port), sorted by timestamp, windowed to `(n, 5, 78)` |
| **Split** | 80% train / 20% test, stratified |
| **Balancing** | SMOTE applied before reshaping, on training set only |
| **Scaling** | MinMaxScaler (train-fit only) |
| **Training** | 20 epochs, batch size 512, early stopping (patience=3, monitor=val_loss) |

---

## Evaluation Results

> Evaluated on 20% stratified holdout test set.

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.76% |
| **Macro F1** | 0.981 |
| **Macro Precision** | 0.979 |
| **Macro Recall** | 0.983 |
| **False Positive Rate** | 0.51% |
| **Inference Time** | ~1.8 ms per flow |

### Where LSTM Outperforms RF/XGBoost

| Attack Class | RF F1 | XGBoost F1 | LSTM F1 |
|-------------|-------|------------|--------|
| DoS Slowloris | 0.94 | 0.93 | **0.98** |
| DoS Slowhttptest | 0.93 | 0.92 | **0.97** |
| DoS GoldenEye | 0.96 | 0.95 | **0.98** |
| Bot (C2) | 0.91 | 0.90 | **0.95** |

> Slow-rate attacks that build over multiple flows are LSTM's primary advantage domain.

### Limitations
- Highest FPR of the three models (0.51%)
- Requires 5 consecutive flows per inference — not suitable for isolated single-flow analysis
- Slowest inference (~1.8 ms) — not ideal for sub-millisecond latency requirements
- SHAP DeepExplainer produces approximate (not exact) Shapley values
- Most sensitive to distribution shift — requires retraining more frequently

---

## Explainability

**SHAP Method:** `DeepExplainer` (gradient-based approximation for neural networks)

```python
# Background sample (100 representative training flows)
background = X_train_reshaped[:100]  # shape: (100, 5, 78)
explainer = shap.DeepExplainer(model, background)

# Explain a batch
shap_values = explainer.shap_values(X_test_reshaped[:10])  # list of arrays per class
# Aggregate across time steps for per-feature attribution
shap_per_feature = np.mean(np.abs(shap_values[predicted_class]), axis=1)  # (n_samples, 78)
```

**Important:** SHAP values are averaged across the 5 time steps for visualization. The waterfall chart shows mean feature contribution across the temporal window.

**Top 5 Features by Mean |SHAP| for Slow-Rate DoS:**
1. `Flow Duration`
2. `Fwd IAT Mean` (inter-arrival time — key temporal signal)
3. `Fwd IAT Std`
4. `Active Mean`
5. `Flow IAT Mean`

> Temporal features (IAT — inter-arrival time) dominate LSTM explanations, unlike RF/XGBoost where byte-count features lead.

---

## Ethical Considerations & Bias

- **Temporal grouping assumption:** Flows are grouped by (Src IP, Dst IP, Dst Port) — this grouping may be invalid in NAT environments or for load-balanced traffic, leading to spurious temporal sequences
- **2017 dataset gap:** Modern slow-rate attacks using HTTP/2 multiplexing are not represented in CICIDS-2017
- **Higher FPR:** 0.51% FPR means more false alerts than RF/XGBoost — analysts must be warned that LSTM alerts carry slightly higher uncertainty
- **Approximate SHAP:** DeepExplainer approximation means explanations are less precise than TreeExplainer — treat LSTM SHAP waterfall charts as directional, not exact
- **No demographic data:** All features are network metadata — no personal data is processed

---

## Loading the Model

```python
import tarfile, os
import tensorflow as tf

# Extract from tar.gz
with tarfile.open('models/lstm_model.tar.gz', 'r:gz') as tar:
    tar.extractall('models/')

# Load Keras model
model = tf.keras.models.load_model('models/lstm_model.h5')
```

> **Note:** The model is stored as `.tar.gz` to reduce repository size. Extract before loading.

---

## Caveats & Recommendations

- Deploy alongside RF or XGBoost — LSTM is best as a **second-opinion model** for slow-rate attack classes specifically
- Use the ensemble voting approach in `dashboard/pages/model_comparison.py` for best overall coverage
- Retrain every 3–6 months given higher sensitivity to concept drift
- For production latency under 1 ms, fall back to XGBoost

---

*Model card follows [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) model card framework.*
