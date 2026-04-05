# Model Card — LSTM Sequence Classifier

**Author:** Chandra Sekhar Chakraborty  
**Date:** 2026-04-05  
**Version:** 1.0.0  
**Task:** Sequence-aware Multi-class Network Intrusion Detection

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | LSTM (2-layer, 128 hidden units each) |
| Framework | TensorFlow 2.x / Keras |
| Input shape | `(sequence_len=10, features=78)` |
| Output | Softmax over 14 classes |
| Serialisation | `models/lstm_model.tar.gz` (contains SavedModel) |
| File size | ~549 KB (demo-scale) |
| Training epochs | 20, early stopping (patience=3) |

## Intended Use

- **Primary use:** Capturing temporal flow patterns that tree models miss (e.g., slow DoS attacks, connection-state sequences).
- **Secondary use:** Ensemble member alongside RF and XGBoost for majority-vote prediction.
- **Out-of-scope:** Single-flow classification without sequence context. Edge deployment on resource-constrained hardware.

## Training Data

Same CICIDS-2017 dataset. Flows are windowed into sequences of 10 consecutive flows per source IP for temporal modelling.

## Evaluation Metrics (test split)

| Metric | Score |
|--------|-------|
| Accuracy | 99.41 % |
| Macro Precision | 99.38 % |
| Macro Recall | 99.43 % |
| Macro F1 | 99.40 % |
| Inference (batch=100) | ~2.3 ms |

## Loading the Model

```python
import tarfile, os, tensorflow as tf

with tarfile.open("models/lstm_model.tar.gz", "r:gz") as t:
    t.extractall("models/lstm_extracted/")
model = tf.keras.models.load_model("models/lstm_extracted/")
```

## Limitations & Bias

- Requires sequence context — cannot classify isolated flows.
- Highest inference latency of the three models; not suitable for sub-millisecond SLA.
- Sequential windowing assumes IP-ordered flows which may break in async capture scenarios.

## Explainability

- Tree-based SHAP is not directly applicable. Use `shap.DeepExplainer` or `shap.GradientExplainer`.
- GradCAM-style attention weights can highlight which timesteps drive predictions.
- `src/explainability/shap_explainer.py` includes a `DeepExplainerWrapper` for LSTM.

## Ethical Considerations

Same as other model cards. LSTM's temporal modelling of IP sequences raises additional privacy considerations — ensure compliance with local data retention laws before logging flow sequences.
