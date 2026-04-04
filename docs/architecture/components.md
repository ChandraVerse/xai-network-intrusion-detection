# Component Reference

Detailed specification of every module in the XAI-NIDS codebase.

---

## 1. Preprocessing Layer (`src/preprocessing/`)

### `cleaner.py`

**Purpose:** Remove rows and features that would corrupt model training.

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Drop infinite values | `df.replace([np.inf, -np.inf], np.nan)` |
| 2 | Fill NaN with median | Column-wise median fill — preserves distribution |
| 3 | Drop zero-variance columns | Features with `std == 0` carry no signal |
| 4 | Drop duplicate rows | Exact duplicates removed |

**CLI usage:**
```bash
python src/preprocessing/cleaner.py --input data/raw/ --output data/processed/
```

---

### `scaler.py`

**Purpose:** Normalise all 78 features to [0, 1] to prevent scale dominance in distance-sensitive operations.

- Uses `sklearn.preprocessing.MinMaxScaler`
- **Fitted on training data only** — prevents data leakage
- Saved as `data/processed/minmax_scaler.pkl` via joblib
- Applied to test set using `scaler.transform()` (not `fit_transform()`)

---

### `smote_balancer.py`

**Purpose:** Synthetically oversample minority classes on the training set to address extreme class imbalance.

- Uses `imblearn.over_sampling.SMOTE(sampling_strategy='not majority')`
- Applied **after** the train/test split — test set is never touched
- Critical for `Infiltration` class (36 real samples → synthetically expanded)
- Produces `data/processed/train_balanced.csv`

---

## 2. ML Detection Engine (`src/models/`)

### `random_forest.py`

**Architecture:**
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

**Why RF for NIDS?**
- Bootstrap aggregation resists overfitting on 78-feature tabular data
- `class_weight='balanced'` is a secondary guard against imbalance after SMOTE
- `TreeExplainer` can compute exact SHAP values in milliseconds — critical for real-time SOC use
- No feature scaling needed internally (but applied consistently for uniformity)

**Outputs:** `models/random_forest.pkl`

---

### `xgboost_model.py`

**Architecture:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)
```

**Why XGBoost for NIDS?**
- Achieves the **lowest false positive rate (0.21%)** — the most critical metric for SOC deployment
- `subsample` and `colsample_bytree < 1.0` add regularisation, reducing false alerts from noise features
- Gradient boosting catches subtle non-linear patterns that RF misses
- `TreeExplainer` compatible — SHAP values computed identically to RF

**Outputs:** `models/xgboost_model.pkl`

---

### `lstm_model.py`

**Architecture:**
```python
Sequential([
    LSTM(128, input_shape=(5, 78), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(14, activation='softmax')  # 14 attack classes
])
```

**Input reshaping:** Flat 78-feature vectors → 3D tensor `(samples, time_steps=5, features=78)` via sliding window.

**Why LSTM for NIDS?**
- Slow-rate DoS attacks (Slowloris, GoldenEye) unfold over multiple consecutive flows. Single-snapshot models (RF, XGBoost) evaluate each flow in isolation and miss the temporal build-up. LSTM sees a window of 5 consecutive flows.
- The hidden state carries temporal context: if the previous 4 flows showed gradually increasing `Bwd Packet Length` and decreasing `Flow IAT Mean`, the LSTM learns this as a DoS signature.

**Outputs:** `models/lstm_model.h5`

---

## 3. XAI Explainability Layer (`src/explainability/`)

### `shap_explainer.py`

**Purpose:** Unified SHAP explainer wrapper that selects `TreeExplainer` or `DeepExplainer` based on model type.

```python
def explain(model, X, model_type='tree'):
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)  # shape: [n_classes, n_samples, n_features]
    elif model_type == 'deep':
        background = shap.sample(X_train, 100)  # 100 background samples
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value
```

---

### `waterfall.py`

**Purpose:** Render a per-alert SHAP waterfall chart — the primary SOC analyst output.

- Selects the SHAP values for the **predicted class only**
- Ranks features by absolute SHAP value, shows top 10
- Red bars = pushed toward attack, Blue bars = pushed toward benign
- Output: Matplotlib figure or `shap.plots.waterfall()` for Streamlit rendering

---

### `summary_plot.py`

**Purpose:** Global SHAP summary visualisations — used for model validation, not per-alert triage.

| Plot | Function | Output |
|------|----------|--------|
| Beeswarm | `shap.summary_plot(shap_values, X_test)` | Feature ranking across all test samples |
| Dependence | `shap.dependence_plot('Flow Duration', shap_values, X_test)` | Non-linear feature interaction |
| Bar | `shap.summary_plot(..., plot_type='bar')` | Mean absolute SHAP per feature |

---

## 4. Dashboard (`dashboard/`)

### `app.py`

**Purpose:** Main Streamlit entry point. Handles model loading, tab routing, and global state.

```
app.py
  └── loads: random_forest.pkl, xgboost_model.pkl, lstm_model.h5
  └── loads: minmax_scaler.pkl, label_encoder.pkl, feature_cols.json
  └── routes to: pages/live_detection.py
               pages/model_comparison.py
               pages/global_shap.py
```

### `pages/live_detection.py`

1. CSV file uploader widget
2. For each uploaded flow: scale → predict (RF/XGB/LSTM) → ensemble vote
3. Render severity badge (colour-coded by attack class)
4. Call `shap_explainer.py` → render `waterfall.py` chart
5. Offer PDF export via `src/utils/report_generator.py`

### `pages/model_comparison.py`

- Loads pre-computed metrics from `models/` (accuracy, F1, confusion matrices, ROC data)
- Side-by-side Plotly charts for all three models

### `pages/global_shap.py`

- Loads pre-computed SHAP values from `models/` (saved during notebook 04)
- Renders `summary_plot.py` in Streamlit via `st.pyplot()`

---

## 5. Utilities (`src/utils/`)

| Module | Purpose |
|--------|---------|
| `metrics.py` | Computes DR, FAR, Precision, Recall, Macro F1, ROC-AUC per class |
| `pcap_converter.py` | Wraps CICFlowMeter CLI to convert PCAP → 78-feature CSV |
| `report_generator.py` | Builds PDF alert report from session alerts using ReportLab |

---

## 6. Tests (`tests/`)

| Test File | What It Covers |
|-----------|---------------|
| `test_preprocessing.py` | Cleaner, scaler, SMOTE on synthetic data |
| `test_models.py` | RF/XGBoost/LSTM prediction shapes, label counts, serialisation |
| `test_explainability.py` | SHAP value shapes, non-null check, waterfall output |

Run all tests:
```bash
python -m pytest tests/ -v
```
