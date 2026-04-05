#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end XAI-NIDS pipeline
# Author: Chandra Sekhar Chakraborty
# Usage:  bash scripts/run_pipeline.sh [--skip-data] [--skip-train] [--skip-shap]
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$ROOT"

SKIP_DATA=false; SKIP_TRAIN=false; SKIP_SHAP=false
for arg in "$@"; do
  case $arg in
    --skip-data)  SKIP_DATA=true  ;;
    --skip-train) SKIP_TRAIN=true ;;
    --skip-shap)  SKIP_SHAP=true  ;;
  esac
done

echo "╔══════════════════════════════════════════╗"
echo "║         XAI-NIDS Pipeline Runner         ║"
echo "╚══════════════════════════════════════════╝"

# 1. Generate sample data if not skipped
if [ "$SKIP_DATA" = false ]; then
  echo "[1/4] Generating synthetic sample data…"
  python scripts/generate_sample_data.py
else
  echo "[1/4] Skipping data generation."
fi

# 2. Preprocess
echo "[2/4] Running preprocessing…"
python -c "
import sys; sys.path.insert(0,'.')
from src.preprocessing.cleaner import clean_dataframe
from src.preprocessing.scaler import fit_scaler
import pandas as pd
df = pd.read_csv('data/samples/sample_100.csv')
df = clean_dataframe(df)
print(f'  Cleaned: {len(df)} rows, {len(df.columns)} cols')
"

# 3. Train models if not skipped
if [ "$SKIP_TRAIN" = false ]; then
  echo "[3/4] Training models…"
  jupyter nbconvert --to notebook --execute notebooks/02_random_forest.ipynb --output /tmp/rf_out.ipynb 2>/dev/null && echo "  ✓ RF" || echo "  ⚠ RF notebook error (check manually)"
  jupyter nbconvert --to notebook --execute notebooks/03_xgboost_lstm.ipynb --output /tmp/xgb_out.ipynb 2>/dev/null && echo "  ✓ XGB/LSTM" || echo "  ⚠ XGB/LSTM notebook error"
else
  echo "[3/4] Skipping training."
fi

# 4. SHAP analysis if not skipped
if [ "$SKIP_SHAP" = false ]; then
  echo "[4/4] Running SHAP analysis…"
  jupyter nbconvert --to notebook --execute notebooks/04_shap_explainability.ipynb --output /tmp/shap_out.ipynb 2>/dev/null && echo "  ✓ SHAP" || echo "  ⚠ SHAP notebook error"
else
  echo "[4/4] Skipping SHAP."
fi

echo ""
echo "✅  Pipeline complete.  Run dashboard: streamlit run dashboard/app.py"
