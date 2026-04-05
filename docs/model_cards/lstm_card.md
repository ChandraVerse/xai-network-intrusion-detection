# Model Card: LSTM Classifier

## Model Details

- **Architecture**: Keras `Sequential` — LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(32, relu) → Dense(14, softmax)
- **Input shape**: `(batch, 5, 78)` — 5 time-steps of 78 features (feature vector repeated)
- **Training**: Adam(lr=1e-3), categorical cross-entropy, EarlyStopping(patience=5), ReduceLROnPlateau
- **Training script**: `scripts/bootstrap_artifacts.py`
- **Artifact**: `models/lstm_model.h5`

## Intended Use

LSTM serves as the deep-learning baseline in this project. It is most meaningful when actual sequential flow data (multiple packets in time order) is available. In the current synthetic setup, the same feature vector is repeated across 5 time-steps as a placeholder.

## Dataset

Identical split to RF / XGBoost. Reshaped to `(n, 5, 78)` for LSTM input.

## Performance (Test Set)

| Metric | Value |
|---|---|
| Accuracy | See `models/lstm_metrics.json` |
| Macro F1 | See `models/lstm_metrics.json` |
| Mean FPR | See `models/lstm_metrics.json` |
| Inference | ~2–5 ms / flow (CPU) |

## Explainability

- **Method**: SHAP `DeepExplainer` or LIME `LimeTabularExplainer` via `make_keras_predict_fn()`
- See `src/explainability/lime_explainer.py::make_keras_predict_fn()`

## Limitations

- Repeated-feature time-steps are not meaningful temporal sequences — replace with real packet-level sequences for production.
- TensorFlow dependency adds ~1 GB to the Docker image; use `--skip-lstm` flag if TF is unavailable.
- Requires `tensorflow>=2.13` on load.
