# `data/` — Dataset Guide

This directory holds the CICIDS-2017 network traffic dataset in two forms:
- `processed/` — cleaned, scaled, SMOTE-balanced CSVs used for model training
- `samples/` — small synthetic CSVs for quick demos and CI smoke tests

> **Raw CSVs are not committed to Git** (they are ~500 MB total).  
> Binary artifacts in `processed/` (`.pkl` scalers) are also excluded.  
> Everything in `samples/` is auto-generated and safe to commit.

---

## Directory Layout

```
data/
├── README.md                        ← this file
│
├── processed/
│   ├── train_balanced.csv           ← SMOTE-balanced training split (80%)
│   ├── test.csv                     ← held-out test split (20%)
│   ├── scaler.pkl                   ← fitted MinMaxScaler (joblib)
│   └── label_map.json               ← {encoded_int: "attack_name"} mapping
│
└── samples/
    ├── sample_10rows.csv            ← 1 row × 10 attack classes (demo)
    ├── sample_100rows.csv           ← 100 rows, all 14 classes balanced
    ├── sample_benign_only.csv       ← 50 benign-only rows
    └── sample_attack_mix.csv        ← 190 rows, attack-heavy mix
```

---

## Dataset: CICIDS-2017

| Property | Value |
|---|---|
| Source | Canadian Institute for Cybersecurity (UNB) |
| URL | https://www.unb.ca/cic/datasets/ids-2017.html |
| Raw size | ~500 MB (CSV files across 5 days) |
| Total flows | ~2.83 million |
| Features | 78 numerical (after cleaning) |
| Target column | `Label` (string) → `label_encoded` (integer, 0–14) |
| Classes | 15 (BENIGN + 14 attack types) |
| Class imbalance | Severe — BENIGN ≈ 80% of raw data |

### Attack Classes (15 total)

| Encoded | Label |
|---|---|
| 0 | BENIGN |
| 1 | Bot |
| 2 | DDoS |
| 3 | DoS GoldenEye |
| 4 | DoS Hulk |
| 5 | DoS Slowhttptest |
| 6 | DoS Slowloris |
| 7 | FTP-Patator |
| 8 | Infiltration |
| 9 | PortScan |
| 10 | SSH-Patator |
| 11 | Web Attack – Brute Force |
| 12 | Web Attack – SQL Injection |
| 13 | Web Attack – XSS |
| 14 | Heartbleed |

---

## How to Build `processed/`

### Step 1: Download the raw dataset

```bash
# Visit: https://www.unb.ca/cic/datasets/ids-2017.html
# Download all 5 day CSVs into data/raw/
mkdir -p data/raw
# Place Monday.csv, Tuesday.csv, ..., Friday.csv into data/raw/
```

### Step 2: Run the preprocessing pipeline

```bash
python scripts/build_processed.py \
    --raw_dir  data/raw/ \
    --out_dir  data/processed/
```

This script runs the full pipeline:
1. `cleaner.py` — drops NaN/Inf rows, removes constant/duplicate columns, strips whitespace from labels
2. `scaler.py` — fits MinMaxScaler on training data only, saves `scaler.pkl`
3. `smote_balancer.py` — applies SMOTE to training split only (never to test)
4. Saves `train_balanced.csv`, `test.csv`, and `label_map.json`

### Step 3: Generate demo samples (optional)

```bash
python scripts/generate_samples.py
```

Produces 4 small synthetic CSVs in `data/samples/` — useful for dashboard demos without the full dataset.

---

## Feature Schema

All 78 features are numerical (float32). After `MinMaxScaler`, all values are in `[0, 1]`.  
Feature names follow the original CICFlowMeter column headers with minor whitespace cleaning.

Key features (highest Gini importance in Random Forest):

| Rank | Feature | Description |
|---|---|---|
| 1 | `Flow Duration` | Total duration of the flow (μs) |
| 2 | `Bwd Packet Length Max` | Maximum backward packet size (bytes) |
| 3 | `Fwd Packet Length Max` | Maximum forward packet size (bytes) |
| 4 | `Flow IAT Mean` | Mean inter-arrival time between packets |
| 5 | `Bwd IAT Total` | Total inter-arrival time, backward direction |
| 6 | `Flow Bytes/s` | Bytes transferred per second |
| 7 | `Flow Packets/s` | Packets per second |
| 8 | `Packet Length Variance` | Variance of all packet lengths |

Full feature list: see `scripts/generate_samples.py` → `FEATURE_NAMES`.

---

## Using Samples for a Quick Demo

```bash
# Generate sample files
python scripts/generate_samples.py

# Use attack mix as a test input for the dashboard
cp data/samples/sample_attack_mix.csv data/processed/test.csv

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Notes

- **Never apply `scaler.transform()` to files in `data/samples/`** — they are already MinMax-scaled to `[0, 1]`.
- The `scaler.pkl` must be loaded from `data/processed/` when preprocessing new live traffic.
- The `Infiltration` class has very few samples in CICIDS-2017 (~36 rows) — expect lower per-class F1 for this class.
- `Heartbleed` is similarly rare — some model runs may not see it in the test split.
