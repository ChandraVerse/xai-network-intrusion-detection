# `data/processed/` — Preprocessed Pipeline Artifacts

> **This folder is intentionally empty in the repository.**
> Binary artifacts (`.npy`, `.pkl`) are excluded via `.gitignore` because they are
> large, machine-specific, and fully reproducible from raw data.
> Run the [Regeneration Steps](#-regeneration-steps) below to populate it locally.

---

## Folder Contents (after running the pipeline)

```
data/processed/
├── X_train.npy          # Scaled, SMOTE-balanced feature matrix  — training set
├── X_test.npy           # Scaled feature matrix                  — test set (no SMOTE)
├── y_train.npy          # Integer-encoded label vector            — training set (SMOTE-balanced)
├── y_test.npy           # Integer-encoded label vector            — test set
├── feature_names.json   # Ordered list of feature column names used by the model
├── label_encoder.pkl    # Fitted sklearn LabelEncoder  (class name ↔ integer)
├── scaler.pkl           # Fitted sklearn MinMaxScaler  (fit on X_train only)
└── README.md            # This file
```

---

## File Descriptions

### `X_train.npy` · `X_test.npy`
- **Type:** NumPy float32 array
- **Shape:** `X_train` — `(N_train, 78)` after SMOTE · `X_test` — `(N_test, 78)`
- **Contents:** MinMax-scaled network flow feature vectors (all values in `[0, 1]`)
- **Produced by:** `src/preprocessing/scaler.py` → `fit_scaler()` / `apply_scaler()`
- **⚠️ Important:** The scaler is **fit on `X_train` only** to prevent data leakage.
  `X_test` is transformed with the already-fitted scaler — never re-fit on test data.

### `y_train.npy` · `y_test.npy`
- **Type:** NumPy int32 array
- **Shape:** `(N_train,)` · `(N_test,)`
- **Contents:** Integer class labels (e.g. `0 = BENIGN`, `1 = Bot`, `2 = DDoS`, …)
- **Produced by:** `LabelEncoder` inside `cleaner.py` pipeline; SMOTE operates on `y_train`
- **Seed:** `random_state=42` is used for the train/test split **and** SMOTE — required
  to reproduce the published metrics (Accuracy 99.94%, Macro F1 0.997).

### `feature_names.json`
- **Type:** JSON array of strings
- **Contents:** Ordered list of the 78 feature column names that survive cleaning
  (after dropping `Flow ID`, `Source IP`, `Destination IP`, `Protocol`, `Timestamp`,
  and any zero-variance columns — see `src/preprocessing/cleaner.py`).
- **Used by:** `dashboard/app.py` to align live Zeek/CICFlowMeter flow records to the
  correct column order before calling `scaler.transform()`, and by SHAP to label
  waterfall / beeswarm plots with human-readable feature names.

### `label_encoder.pkl`
- **Type:** Serialised `sklearn.preprocessing.LabelEncoder` (joblib)
- **Contents:** Fitted encoder — maps class name string → integer and back.
- **Used by:** `dashboard/app.py` — `label_encoder.inverse_transform(prediction)`
  converts the model's integer output to a display string like `"DDoS"` or `"BENIGN"`.
- **Load with:**
  ```python
  import joblib
  le = joblib.load("data/processed/label_encoder.pkl")
  print(le.classes_)   # array(['BENIGN', 'Bot', 'DDoS', ...])
  ```

### `scaler.pkl`
- **Type:** Serialised `sklearn.preprocessing.MinMaxScaler` (joblib)
- **Contents:** MinMaxScaler fitted **only on the training set** (`X_train` pre-SMOTE).
  Stores `data_min_`, `data_max_`, `scale_`, and `feature_range=(0, 1)`.
- **Used by:** `dashboard/app.py` — every incoming live network flow is transformed
  via `scaler.transform([raw_flow_vector])` before being passed to the model.
- **Load with:**
  ```python
  import joblib
  scaler = joblib.load("data/processed/scaler.pkl")
  X_live_scaled = scaler.transform(X_live_raw)
  ```

---

## 🔄 Regeneration Steps

### Prerequisites

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Download CICIDS-2017 dataset
#    Source: https://www.unb.ca/cic/datasets/ids-2017.html
#    Place all 8 day-CSVs inside:  data/raw/
#    Expected files:
#      MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv
#      MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv
#      MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv
#      MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
#      MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
#      MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv
#      MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
#      MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

### Run the Full Pipeline

```bash
# Step 1 — Clean: strip whitespace, drop Inf/NaN, remove zero-variance columns
python src/preprocessing/cleaner.py \
    --input  data/raw/MachineLearningCVE/ \
    --output data/processed/

# Step 2 — Split + Scale + SMOTE + Save all artifacts
#          (run from repo root so relative paths resolve correctly)
python scripts/build_processed.py
```

> **`scripts/build_processed.py`** is the master pipeline script that:
> 1. Loads `data/processed/cleaned.csv`
> 2. Encodes labels → saves `label_encoder.pkl`
> 3. Saves `feature_names.json`
> 4. Performs stratified train/test split (`test_size=0.20, random_state=42`)
> 5. Fits `MinMaxScaler` on `X_train` → saves `scaler.pkl`
> 6. Applies SMOTE to training set (`random_state=42, strategy="not majority"`)
> 7. Saves `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

### Expected Output Shapes

| File | Shape | Notes |
|------|-------|-------|
| `X_train.npy` | `(≈2,270,000, 78)` | After SMOTE balancing |
| `X_test.npy` | `(≈566,000, 78)` | 20% holdout, untouched |
| `y_train.npy` | `(≈2,270,000,)` | Balanced across 14 classes |
| `y_test.npy` | `(≈566,000,)` | Original distribution |

### Verify the Artifacts

```python
import numpy as np, json, joblib

X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

with open("data/processed/feature_names.json") as f:
    features = json.load(f)

le      = joblib.load("data/processed/label_encoder.pkl")
scaler  = joblib.load("data/processed/scaler.pkl")

print(f"X_train : {X_train.shape}  dtype={X_train.dtype}")
print(f"X_test  : {X_test.shape}   dtype={X_test.dtype}")
print(f"y_train : {y_train.shape}  classes={len(np.unique(y_train))}")
print(f"y_test  : {y_test.shape}")
print(f"Features: {len(features)} columns")
print(f"Classes : {list(le.classes_)}")
print(f"Scaler  : min={X_train.min():.4f}  max={X_train.max():.4f}  (should be 0.0 / 1.0)")
```

Expected output:
```
X_train : (2270xxx, 78)  dtype=float32
X_test  : (566xxx, 78)   dtype=float32
y_train : (2270xxx,)     classes=14
y_test  : (566xxx,)
Features: 78 columns
Classes : ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk',
           'DoS Slowhttptest', 'DoS Slowloris', 'FTP-Patator',
           'Infiltration', 'PortScan', 'SSH-Patator',
           'Web Attack - Brute Force', 'Web Attack - SQLi', 'Web Attack - XSS']
Scaler  : min=0.0000  max=1.0000  (should be 0.0 / 1.0)
```

---

## ⚙️ Pipeline Architecture

```
data/raw/*.csv  (8 day-CSVs, ~2.8 GB total)
       │
       ▼
 cleaner.py          → strip whitespace, drop Inf/NaN/zero-variance
       │
       ▼
  cleaned.csv        (intermediate, not committed)
       │
       ├──── LabelEncoder.fit()    → label_encoder.pkl
       ├──── feature_names saved   → feature_names.json
       │
       ▼
 train_test_split()  (80/20 stratified, random_state=42)
       │
       ├── X_train (raw) ──► MinMaxScaler.fit()  → scaler.pkl
       │                     MinMaxScaler.transform()
       │                     SMOTE(random_state=42)
       │                          │
       │                          └──► X_train.npy  y_train.npy
       │
       └── X_test (raw)  ──► MinMaxScaler.transform() (fitted scaler only!)
                                   │
                                   └──► X_test.npy   y_test.npy
```

---

## 🔒 Why These Files Are Git-Ignored

Binary NumPy arrays and pickled scikit-learn objects are excluded from version control because:

1. **Size** — `X_train.npy` alone is ~700 MB; GitHub hard-limits files to 100 MB.
2. **Reproducibility** — They are 100% deterministic given `random_state=42`; committing
   them would create false confidence that the data cannot change.
3. **Security** — Pickle files can execute arbitrary code; never load `.pkl` files from
   untrusted sources. Generating them locally from clean source ensures trust.
4. **DVC / LFS** — For teams needing shared artifact storage, configure
   [DVC](https://dvc.org) or [Git LFS](https://git-lfs.github.com) instead.

The `.gitignore` entries responsible:
```gitignore
data/processed/*.npy
data/processed/*.pkl
data/raw/
```

---

## Related Files

| Path | Role |
|------|------|
| `src/preprocessing/cleaner.py` | Step 1 — raw CSV cleaning |
| `src/preprocessing/scaler.py` | MinMaxScaler fit/transform/load helpers |
| `src/preprocessing/smote_balancer.py` | SMOTE oversampling (train only) |
| `scripts/build_processed.py` | Master pipeline — runs all steps end-to-end |
| `notebooks/01_eda.ipynb` | Exploratory analysis of the cleaned dataset |
| `models/` | Trained model artifacts (`.pkl`, `.h5`) — separate from data artifacts |
| `dashboard/app.py` | Loads `scaler.pkl` + `label_encoder.pkl` at startup |
