# data/samples/

Synthetic CICIDS-2017 sample CSVs for unit tests, dashboard smoke-testing, and CI validation.

## Files

| File | Rows | Description |
|---|---|---|
| `sample_10rows.csv` | 10 | One sample per class (10 of 14 classes) |
| `sample_100rows.csv` | 100 | Balanced across all 14 classes, shuffled |
| `sample_benign_only.csv` | 50 | BENIGN traffic only |
| `sample_attack_mix.csv` | 190 | 60 BENIGN + 10 per attack class (13 types) |

## Format

- **79 columns**: `Label` + 78 CICIDS-2017 features
- **Values**: MinMax-scaled to `[0.0, 1.0]` — matches output of `data/processed/scaler.pkl`
- **Do NOT** pass these through `scaler.transform()` again
- Feature column order matches `models/feature_importance_rf.json` and `dashboard/config.py`

## Regenerating

```bash
python scripts/generate_samples.py
```

Uses `random_state=42` for reproducibility. Output is deterministic.

## Class Labels

```
BENIGN, Bot, DDoS, DoS GoldenEye, DoS Hulk,
DoS Slowhttptest, DoS Slowloris, FTP-Patator,
Infiltration, PortScan, SSH-Patator,
Web Attack - Brute Force, Web Attack - SQLi, Web Attack - XSS
```
