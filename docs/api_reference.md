# API Reference

This document covers the public Python API for all modules in `src/`. Import paths assume the repo root is on `sys.path`.

---

## `src.utils`

```python
from src.utils import (
    get_logger,
    compute_metrics,
    print_metrics_table,
    format_metrics_for_dashboard,
    ReportGenerator,
    PcapConverter,
)
```

---

### `get_logger(name, level="INFO") -> logging.Logger`

Returns a configured `logging.Logger` instance with a consistent format.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Logger name (use `__name__`) |
| `level` | `str` | `"INFO"` | Logging level: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

**Returns:** `logging.Logger`

**Example:**
```python
from src.utils import get_logger
log = get_logger(__name__)
log.info("Pipeline started")
```

---

### `compute_metrics(y_true, y_pred, class_names) -> dict`

Computes a comprehensive set of classification metrics.

| Parameter | Type | Description |
|---|---|---|
| `y_true` | `np.ndarray` (int) | Ground-truth class indices |
| `y_pred` | `np.ndarray` (int) | Predicted class indices |
| `class_names` | `list[str]` | Ordered list of class label strings |

**Returns:** `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `accuracy` | `float` | Overall accuracy |
| `macro_f1` | `float` | Macro-averaged F1 score |
| `macro_precision` | `float` | Macro-averaged precision |
| `macro_recall` | `float` | Macro-averaged recall |
| `mean_fpr` | `float` | Mean false positive rate across classes |
| `confusion_matrix` | `np.ndarray` | Shape `(n_classes, n_classes)` |
| `per_class` | `dict` | Per-class `{precision, recall, f1, support}` |

**Example:**
```python
from src.utils import compute_metrics
m = compute_metrics(y_test, y_pred, CLASS_NAMES)
print(f"Accuracy: {m['accuracy']:.4f}  Macro F1: {m['macro_f1']:.4f}")
```

---

### `print_metrics_table(metrics, title="") -> None`

Prints a formatted per-class metrics table to stdout.

| Parameter | Type | Description |
|---|---|---|
| `metrics` | `dict` | Output of `compute_metrics()` |
| `title` | `str` | Optional header line |

---

### `format_metrics_for_dashboard(metrics) -> dict`

Converts `compute_metrics()` output into a JSON-serialisable dict safe for Streamlit / Dash callbacks (removes `np.ndarray` values, rounds floats).

| Parameter | Type | Description |
|---|---|---|
| `metrics` | `dict` | Output of `compute_metrics()` |

**Returns:** `dict` — all values are Python native types (`float`, `int`, `list`).

---

### `class ReportGenerator`

Generates JSON and Markdown summary reports from evaluation metrics.

```python
from src.utils import ReportGenerator
rg = ReportGenerator()
```

#### `ReportGenerator.save_json(metrics, path) -> None`

| Parameter | Type | Description |
|---|---|---|
| `metrics` | `dict` | Output of `compute_metrics()` |
| `path` | `str` | Output file path (e.g. `reports/report.json`) |

#### `ReportGenerator.save_markdown(metrics, path, title="") -> None`

Writes a Markdown summary table to `path`.

---

### `class PcapConverter`

Converts raw PCAP files to the 78-column CICIDS-2017 feature vector format.

```python
from src.utils import PcapConverter
pc = PcapConverter()
```

#### `PcapConverter.convert(pcap_path) -> pd.DataFrame`

| Parameter | Type | Description |
|---|---|---|
| `pcap_path` | `str` | Path to a `.pcap` or `.pcapng` file |

**Returns:** `pd.DataFrame` with columns matching `data/processed/feature_names.json`.

> **Note:** Requires `scapy` (`pip install scapy`). Not installed by default.

---

## `src.explainability`

```python
from src.explainability.shap_explainer import explain, SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.summary_plot   import plot_summary
from src.explainability.waterfall      import plot_waterfall
```

---

### `explain(model, X, model_type="tree") -> tuple[np.ndarray, shap.Explainer]`

Convenience wrapper — creates the appropriate SHAP explainer and returns SHAP values.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | fitted estimator | required | RF, XGBoost, or Keras model |
| `X` | `np.ndarray` | required | Samples to explain, shape `(n, features)` |
| `model_type` | `str` | `"tree"` | `"tree"` \| `"deep"` \| `"linear"` \| `"kernel"` |

**Returns:** `(shap_values, explainer)` tuple.

**Example:**
```python
from src.explainability.shap_explainer import explain
shap_vals, exp = explain(rf_model, X_test[:100], model_type="tree")
```

---

### `class SHAPExplainer`

Object-oriented wrapper around SHAP with caching and plot helpers.

#### `SHAPExplainer(model, model_type="tree")`
#### `SHAPExplainer.fit(X_background) -> self`
#### `SHAPExplainer.explain(X) -> np.ndarray`
#### `SHAPExplainer.plot_summary(X, feature_names, max_display=20) -> None`
#### `SHAPExplainer.plot_waterfall(X, idx, feature_names) -> None`

---

### `class LIMEExplainer`

Local interpretable model-agnostic explanations for individual network flow alerts.

#### `LIMEExplainer(training_data, feature_names, class_names, mode="classification")`

| Parameter | Type | Description |
|---|---|---|
| `training_data` | `np.ndarray` | Background training samples for perturbation |
| `feature_names` | `list[str]` | Column names matching `data/processed/feature_names.json` |
| `class_names` | `list[str]` | Ordered attack/benign class labels |
| `mode` | `str` | `"classification"` (default) |

#### `LIMEExplainer.explain_instance(x, predict_fn, num_features=15, num_samples=500) -> lime.Explanation`

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` (1-D) | Single flow feature vector |
| `predict_fn` | `callable` | `model.predict_proba` |
| `num_features` | `int` | Top-k features to include |
| `num_samples` | `int` | Perturbation samples for local linear fit |

**Returns:** `lime.explanation.Explanation` — call `.as_list()` for `(feature_condition, weight)` pairs.

#### `LIMEExplainer.explain_batch(X, predict_fn, indices=None, num_features=15) -> list[dict]`

Batch-explain multiple rows. Returns list of `{"index": int, "weights": list[tuple]}` dicts.

#### `LIMEExplainer.get_weights(explanation) -> list[tuple[str, float]]`

Returns sorted `(feature_condition, weight)` list from a single explanation.

#### `LIMEExplainer.plot_weights(explanation, title="") -> matplotlib.figure.Figure`

Returns a horizontal bar chart of LIME weights. Does not call `plt.show()` — caller must display or save.

#### `LIMEExplainer.save_explanation(explanation, path) -> None`

Serialises explanation to JSON (schema mirrors SHAP output for dashboard compatibility).

---

### `plot_summary(shap_values, X, feature_names, max_display=20, plot_type="dot") -> None`

Thin wrapper around `shap.summary_plot`. Saves to `reports/figures/shap_summary.png` if `save=True`.

---

### `plot_waterfall(shap_values, idx, feature_names, class_idx=0) -> None`

Renders a SHAP waterfall chart for sample at position `idx`.

---

## `src.preprocessing`

```python
from src.preprocessing.cleaner   import clean_dataframe
from src.preprocessing.encoder   import encode_labels
from src.preprocessing.scaler    import fit_scaler, transform
```

---

### `clean_dataframe(df) -> pd.DataFrame`

Drops duplicate rows, removes `inf`/`-inf`, fills NaN with column median, drops zero-variance columns.

### `encode_labels(y, return_encoder=False) -> np.ndarray | tuple`

Ordinal-encodes string labels. If `return_encoder=True`, returns `(encoded_array, LabelEncoder)`.

### `fit_scaler(X_train) -> MinMaxScaler`

Fits and returns a `MinMaxScaler` on `X_train`.

### `transform(X, scaler) -> np.ndarray`

Applies a pre-fitted scaler and returns `float32` array.

---

## `src.models`

```python
from src.models.random_forest import train_rf, load_rf
from src.models.xgboost_model import train_xgb, load_xgb
from src.models.lstm_model    import train_lstm, load_lstm, build_lstm
```

---

### `train_rf(X_train, y_train, **kwargs) -> RandomForestClassifier`

Trains and returns a fitted `RandomForestClassifier`. Passes `**kwargs` to the constructor.

### `load_rf(path="models/random_forest.pkl") -> RandomForestClassifier`

Loads a serialised RF model via `joblib`.

### `train_xgb(X_train, y_train, **kwargs) -> XGBClassifier`

Trains and returns a fitted `XGBClassifier`.

### `load_xgb(path="models/xgboost_model.pkl") -> XGBClassifier`

Loads a serialised XGBoost model via `joblib`.

### `build_lstm(input_dim, n_classes, timesteps=5) -> tf.keras.Model`

Builds and compiles an LSTM model.

| Parameter | Type | Description |
|---|---|---|
| `input_dim` | `int` | Number of features (78 for CICIDS-2017) |
| `n_classes` | `int` | Number of output classes (14) |
| `timesteps` | `int` | Sequence length fed to LSTM |

### `train_lstm(model, X_train, y_train, epochs=30, batch_size=256) -> tf.keras.callbacks.History`

### `load_lstm(path="models/lstm_model") -> tf.keras.Model`

Extracts `lstm_model.tar.gz` if needed, then loads with `tf.keras.models.load_model`.

---

## Quick-Start Cheatsheet

```python
import joblib, numpy as np
from src.utils import compute_metrics, get_logger
from src.explainability.shap_explainer import explain
from src.explainability.lime_explainer import LIMEExplainer

log  = get_logger(__name__)
rf   = joblib.load("models/random_forest.pkl")
X    = np.load("data/processed/X_test.npz")["data"]
y    = np.load("data/processed/y_test.npy")

# Metrics
m = compute_metrics(y, rf.predict(X), CLASS_NAMES)
log.info("Accuracy: %.4f  F1: %.4f", m["accuracy"], m["macro_f1"])

# SHAP (global)
shap_vals, _ = explain(rf, X[:200], model_type="tree")

# LIME (single alert)
lime = LIMEExplainer(X[:500], FEATURE_NAMES, CLASS_NAMES)
exp  = lime.explain_instance(X[0], rf.predict_proba)
for feat, weight in lime.get_weights(exp):
    print(f"  {feat:45s}  {weight:+.4f}")
```
