# `models/` — Trained Artifact Store

This directory holds **all serialised model artifacts** produced by the training scripts in `src/models/`.  
Artifacts are **not committed to Git** (see `.gitignore`) — they are reproduced locally by running the training pipeline.

---

## Directory Layout

```
models/
├── README.md                        ← this file
│
├── random_forest.pkl                ← joblib-serialised RandomForestClassifier  (compress=3)
├── rf_metrics.json                  ← accuracy / macro-F1 / FPR / inference time
├── feature_importance_rf.json       ← Gini feature importance (sorted desc)
│
├── xgboost_model.pkl                ← joblib-serialised XGBClassifier  (compress=3)
├── xgb_metrics.json                 ← accuracy / macro-F1 / FPR / best iteration
│
├── lstm_model.h5                    ← Keras HDF5 weights
├── lstm_metrics.json                ← accuracy / macro-F1 / FPR / inference time
│
├── metrics_summary.json             ← combined three-model comparison table
└── model_registry.yaml             ← version metadata & artifact checksums
```

---

## How to Reproduce Artifacts

### 1. Prerequisites

```bash
pip install -r requirements.txt
```

Ensure `data/processed/train_balanced.csv` and `data/processed/test.csv` exist.  
If not, run the preprocessing pipeline first:

```bash
python scripts/preprocess.py
```

### 2. Train all three models

```bash
# Random Forest
python src/models/random_forest.py \
    --data data/processed/train_balanced.csv \
    --test data/processed/test.csv \
    --out  models/

# XGBoost
python src/models/xgboost_model.py \
    --data data/processed/train_balanced.csv \
    --test data/processed/test.csv \
    --out  models/

# LSTM
python src/models/lstm_model.py \
    --data data/processed/train_balanced.csv \
    --test data/processed/test.csv \
    --out  models/
```

### 3. Train all via Docker

```bash
docker compose run app python src/models/random_forest.py
docker compose run app python src/models/xgboost_model.py
docker compose run app python src/models/lstm_model.py
```

---

## Artifact Descriptions

### Random Forest (`random_forest.pkl`)

| Parameter | Value |
|---|---|
| Algorithm | `sklearn.ensemble.RandomForestClassifier` |
| `n_estimators` | 200 |
| `max_depth` | None (fully grown) |
| `class_weight` | `balanced` |
| `random_state` | 42 |
| Serialiser | `joblib.dump(..., compress=3)` |
| Extra output | `feature_importance_rf.json` — Gini importances sorted descending |

### XGBoost (`xgboost_model.pkl`)

| Parameter | Value |
|---|---|
| Algorithm | `xgboost.XGBClassifier` |
| `n_estimators` | 300 (max; early-stopped) |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `tree_method` | `hist` (GPU: change to `gpu_hist`) |
| Early stopping | 20 rounds on 10% validation split |
| Serialiser | `joblib.dump(..., compress=3)` |

### LSTM (`lstm_model.h5`)

| Parameter | Value |
|---|---|
| Architecture | `LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(32, relu) → Dense(n_classes, softmax)` |
| Input shape | `(time_steps=5, n_features=78)` |
| Optimizer | Adam (`lr=1e-3`) |
| Loss | `categorical_crossentropy` |
| `batch_size` | 512 |
| Max epochs | 30 (early stopping: patience=5 on `val_loss`) |
| LR schedule | `ReduceLROnPlateau` factor=0.5, patience=3, min_lr=1e-6 |
| Serialiser | `model.save("lstm_model.h5")` (Keras HDF5) |

---

## Metrics Reference

All metrics files share a common schema:

```json
{
  "model": "<ModelName>",
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "false_positive_rate": 0.0,
  "inference_ms_per_flow": 0.0,
  "n_test_samples": 0
}
```

See `models/metrics_summary.json` for a combined side-by-side view of all three models after training.

---

## Loading a Saved Model

```python
import joblib
import tensorflow as tf

# Random Forest
rf  = joblib.load("models/random_forest.pkl")

# XGBoost
xgb = joblib.load("models/xgboost_model.pkl")

# LSTM
lstm = tf.keras.models.load_model("models/lstm_model.h5")
```

---

## Notes

- All `.pkl` and `.h5` files are excluded from Git via `.gitignore`.
- `metrics_summary.json` **is** committed after training so reviewers can see benchmark results without re-running training.
- `model_registry.yaml` tracks version metadata, dataset hash, and SHA-256 checksums of artifacts.
- To use GPU for LSTM training, set `TF_FORCE_GPU_ALLOW_GROWTH=true` in your environment and ensure CUDA is available.
- For XGBoost GPU acceleration, change `tree_method` from `hist` to `gpu_hist` in `src/models/xgboost_model.py`.
