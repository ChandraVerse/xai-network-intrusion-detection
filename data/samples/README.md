# `data/samples/` — Ready-to-Use Sample Flows

This folder contains **small, pre-built CSV samples** derived from CICIDS-2017
feature distributions. They let you:

- Test the dashboard (`streamlit run dashboard/app.py`) without running the full
  2.8 GB preprocessing pipeline.
- Quickly smoke-test new code changes against known inputs.
- Demonstrate the project to reviewers who cannot download the full dataset.

> **These samples are synthetically generated from CICIDS-2017 statistical profiles**  
> (min-max scaled, `random_state=42`). They are **not** raw packet captures.  
> For full reproducibility of published metrics, always use the real pipeline:  
> `python scripts/build_processed.py`

---

## Files

| File | Rows | Classes | Purpose |
|------|------|---------|----------|
| `sample_10rows.csv` | 10 | 10 (1 per class) | Quickest smoke test — one sample per major attack class |
| `sample_100rows.csv` | 100 | All 14 | Balanced across all 14 CICIDS-2017 classes, shuffled |
| `sample_benign_only.csv` | 50 | BENIGN only | Baseline / normal traffic — use to verify zero false positives |
| `sample_attack_mix.csv` | 190 | All 14 | Attack-heavy (70% attacks, 30% benign) — ideal for dashboard demo |

All files share the same schema:
- **Column 0:** `Label` — string class name (e.g. `"DDoS"`, `"BENIGN"`)
- **Columns 1–78:** 78 MinMax-scaled network flow features (float, range `[0, 1]`)

---

## Column Schema

```
Label  (string)         — attack class name
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Bwd Packet Length Mean
Bwd Packet Length Std
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Total
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Total
Bwd IAT Mean
Bwd IAT Std
Bwd IAT Max
Bwd IAT Min
Fwd PSH Flags
Bwd PSH Flags
Fwd URG Flags
Bwd URG Flags
Fwd Header Length
Bwd Header Length
Fwd Packets/s
Bwd Packets/s
Min Packet Length
Max Packet Length
Packet Length Mean
Packet Length Std
Packet Length Variance
FIN Flag Count
SYN Flag Count
RST Flag Count
PSH Flag Count
ACK Flag Count
URG Flag Count
CWE Flag Count
ECE Flag Count
Down/Up Ratio
Average Packet Size
Avg Fwd Segment Size
Avg Bwd Segment Size
Fwd Header Length.1
Subflow Fwd Packets
Subflow Fwd Bytes
Subflow Bwd Packets
Subflow Bwd Bytes
Init_Win_bytes_forward
Init_Win_bytes_backward
act_data_pkt_fwd
min_seg_size_forward
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min
Inbound
Fwd Avg Bytes/Bulk
Fwd Avg Packets/Bulk
Fwd Avg Bulk Rate
Bwd Avg Bytes/Bulk
Bwd Avg Packets/Bulk
Bwd Avg Bulk Rate
```

---

## Usage Examples

### Load and inspect

```python
import pandas as pd

df = pd.read_csv("data/samples/sample_100rows.csv")
print(df.shape)              # (100, 79)
print(df["Label"].value_counts())
print(df.describe())
```

### Run inference with the trained Random Forest

```python
import pandas as pd, numpy as np, joblib, json

# Load artifacts
scaler = joblib.load("models/minmax_scaler.pkl")
le     = joblib.load("models/label_encoder.pkl")
rf     = joblib.load("models/random_forest.pkl")

with open("data/processed/feature_names.json") as f:
    feature_cols = json.load(f)

# Load sample
df = pd.read_csv("data/samples/sample_attack_mix.csv")
X  = df[feature_cols].values          # already scaled — skip scaler.transform()
y_true = le.transform(df["Label"])    # encode string labels to integers

# Predict
y_pred = rf.predict(X)
print(le.inverse_transform(y_pred))   # back to string class names
```

> ⚠️ **Note:** These sample features are **already MinMax-scaled** (values in `[0, 1]`).  
> Do **not** call `scaler.transform()` on them — that would double-scale.  
> Raw CICFlowMeter outputs (unscaled) must be scaled before inference.

### Use as the dashboard simulation pool

```bash
# Copy the attack-heavy sample to where app.py expects the test pool:
cp data/samples/sample_attack_mix.csv data/processed/test.csv

# Then run the dashboard:
streamlit run dashboard/app.py
```

---

## Regenerate Samples

If you want to regenerate these files from scratch:

```bash
python scripts/generate_samples.py
```

---

## Class Distribution in Each File

### `sample_100rows.csv`

| Class | Count |
|-------|-------|
| BENIGN | 8 |
| Bot | 7 |
| DDoS | 7 |
| DoS GoldenEye | 7 |
| DoS Hulk | 7 |
| DoS Slowhttptest | 7 |
| DoS Slowloris | 7 |
| FTP-Patator | 7 |
| Infiltration | 7 |
| PortScan | 7 |
| SSH-Patator | 7 |
| Web Attack - Brute Force | 7 |
| Web Attack - SQLi | 7 |
| Web Attack - XSS | 7 |

### `sample_attack_mix.csv`

| Class | Count |
|-------|-------|
| BENIGN | 60 |
| All 13 attack classes | 10 each |

---

## Related Files

| Path | Role |
|------|------|
| `data/processed/README.md` | Full pipeline documentation |
| `scripts/build_processed.py` | Generates real train/test artifacts |
| `scripts/generate_samples.py` | Regenerates these sample CSVs |
| `dashboard/app.py` | Loads `data/processed/test.csv` as simulation pool |
