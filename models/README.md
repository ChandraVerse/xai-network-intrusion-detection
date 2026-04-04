# Model Artifacts

This folder stores all serialised trained model artifacts produced by `notebooks/03_model_training.ipynb`.

> **These files are not committed to the repository** (see `.gitignore`) because binary model artifacts can exceed 100 MB and change with every training run. The instructions below explain how to reproduce them.

---

## Expected Artifacts

| File | Format | Size (approx.) | Produced By | Description |
|------|--------|----------------|-------------|-------------|
| `random_forest.pkl` | joblib pickle | ~150–400 MB | `03_model_training.ipynb` | Trained `RandomForestClassifier` (200 estimators) |
| `xgboost_model.pkl` | joblib pickle | ~10–30 MB | `03_model_training.ipynb` | Trained `XGBClassifier` (300 estimators) |
| `lstm_model.h5` | Keras HDF5 | ~5–15 MB | `03_model_training.ipynb` | Trained LSTM weights (128→64 units) |
| `rf_metrics.json` | JSON | < 1 KB | `03_model_training.ipynb` | RF accuracy, F1, FPR, inference time |
| `xgb_metrics.json` | JSON | < 1 KB | `03_model_training.ipynb` | XGBoost accuracy, F1, FPR, inference time |
| `lstm_metrics.json` | JSON | < 1 KB | `03_model_training.ipynb` | LSTM accuracy, F1, FPR, inference time |
| `rf_shap_values.npy` | NumPy array | ~50–200 MB | `04_xai_shap.ipynb` | Pre-computed SHAP values for RF (test set) |
| `xgb_shap_values.npy` | NumPy array | ~50–200 MB | `04_xai_shap.ipynb` | Pre-computed SHAP values for XGBoost (test set) |
| `feature_importance_rf.json` | JSON | < 1 KB | `03_model_training.ipynb` | Gini feature importance (78 features) |

---

## How to Generate the Artifacts

### Option A — Jupyter Notebooks (Recommended)

```bash
# 1. Ensure preprocessed data exists
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb

# 2. Train all three models (generates .pkl and .h5 files)
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb

# 3. Compute SHAP values (generates .npy files)
jupyter nbconvert --to notebook --execute notebooks/04_xai_shap.ipynb
```

### Option B — CLI Scripts

```bash
# Preprocess first
python src/preprocessing/cleaner.py --input data/raw/ --output data/processed/

# Train each model
python src/models/random_forest.py  --data data/processed/train_balanced.csv --out models/
python src/models/xgboost_model.py  --data data/processed/train_balanced.csv --out models/
python src/models/lstm_model.py     --data data/processed/train_balanced.csv --out models/
```

---

## Model Performance Summary

| Model | Accuracy | Macro F1 | False Positive Rate | Inference Time |
|-------|----------|----------|---------------------|----------------|
| Random Forest | **99.94%** | **0.997** | 0.28% | ~0.4 ms/flow |
| XGBoost | 99.91% | 0.994 | **0.21%** | ~0.2 ms/flow |
| LSTM | 99.76% | 0.981 | 0.51% | ~1.8 ms/flow |

> Results on CICIDS-2017 test set (20% holdout, stratified). SMOTE applied to training set only.

---

## Loading Models in Python

```python
import joblib
import tensorflow as tf
import json

# Load Random Forest
rf_model = joblib.load('models/random_forest.pkl')

# Load XGBoost
xgb_model = joblib.load('models/xgboost_model.pkl')

# Load LSTM
lstm_model = tf.keras.models.load_model('models/lstm_model.h5')

# Load label map
with open('data/processed/label_map.json') as f:
    label_map = json.load(f)  # {0: 'BENIGN', 1: 'DDoS', ...}

# Load scaler
scaler = joblib.load('data/processed/minmax_scaler.pkl')
```

---

## Quick Inference Example

```python
import pandas as pd
import numpy as np
import joblib

# Load artifacts
rf    = joblib.load('models/random_forest.pkl')
scaler = joblib.load('data/processed/minmax_scaler.pkl')
encoder = joblib.load('data/processed/label_encoder.pkl')

with open('data/processed/feature_cols.json') as f:
    import json; feature_cols = json.load(f)

# Load a flow sample
df = pd.read_csv('data/samples/sample_100.csv')[feature_cols]
X  = scaler.transform(df)

# Predict
y_pred = rf.predict(X)
y_prob = rf.predict_proba(X).max(axis=1)
labels = encoder.inverse_transform(y_pred)

for label, confidence in zip(labels, y_prob):
    print(f'{label:30s}  {confidence*100:.1f}%')
```

---

## Notes on Model Size

- **Random Forest** serialised with joblib can be 150–400 MB depending on tree depth and the size of the SMOTE-balanced training set (~800K rows after balancing). Use `compress=3` in `joblib.dump()` to reduce to ~40–80 MB at the cost of slightly slower load time.
- **XGBoost** is compact because it serialises only the boosted trees (not bootstrap samples). Typically 10–30 MB.
- **LSTM** in HDF5 format stores weights only (not the full TensorFlow graph). Typically 5–15 MB.

```python
# Compressed RF save (recommended for deployment)
joblib.dump(rf_model, 'models/random_forest.pkl', compress=3)
```

---

## Versioning

If you retrain models and want to track versions:

```bash
# Tag with dataset hash and date
mv models/random_forest.pkl models/random_forest_v1_cicids2017_2026.pkl
```

Or use MLflow for full experiment tracking — see `CONTRIBUTING.md` for the MLflow integration guide.
