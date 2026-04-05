#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  End-to-end XAI-NIDS pipeline
#
# Usage:
#   bash scripts/run_pipeline.sh [--sample]
#
# Flags:
#   --sample    Use data/samples/sample_100.csv instead of full CICIDS-2017
#               (fast mode for CI and demo purposes)
# =============================================================================

set -euo pipefail

SAMPLE_MODE=false
for arg in "$@"; do
  [[ "$arg" == "--sample" ]] && SAMPLE_MODE=true
done

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${BLUE}[PIPELINE]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ---- 0. Guard ------------------------------------------------------------
[[ -f requirements.txt ]] || fail "Run from repo root: cd xai-network-intrusion-detection"
python -c "import pandas, sklearn, xgboost, shap, streamlit" 2>/dev/null \
  || fail "Missing dependencies. Run: pip install -r requirements.txt"

# ---- 1. Generate sample data (if needed) --------------------------------
if $SAMPLE_MODE; then
  log "Sample mode: generating synthetic data..."
  python scripts/generate_sample_data.py --rows 500 --out data/samples/sample_100.csv
  ok  "Sample data ready at data/samples/sample_100.csv"
else
  [[ -d data/raw ]] || fail "data/raw/ not found. Download CICIDS-2017 first."
  log "Full dataset mode: using data/raw/"
fi

# ---- 2. Preprocessing ----------------------------------------------------
log "Running preprocessing pipeline..."
if $SAMPLE_MODE; then
  python src/preprocessing/cleaner.py \
    --input  data/samples/sample_100.csv \
    --output data/processed/
else
  python src/preprocessing/cleaner.py \
    --input  data/raw/ \
    --output data/processed/
fi
ok "Preprocessing complete. Artifacts in data/processed/"

# ---- 3. Model Training ---------------------------------------------------
log "Training Random Forest..."
python src/models/random_forest.py  --data data/processed/train.csv
ok  "Random Forest saved -> models/random_forest.pkl"

log "Training XGBoost..."
python src/models/xgboost_model.py  --data data/processed/train.csv
ok  "XGBoost saved -> models/xgboost_model.pkl"

log "Training LSTM..."
python src/models/lstm_model.py     --data data/processed/train.csv
ok  "LSTM saved -> models/lstm_model.tar.gz"

# ---- 4. Evaluation -------------------------------------------------------
log "Evaluating all models on test set..."
python src/utils/metrics.py \
  --models models/ \
  --test   data/processed/test.csv
ok "Metrics saved -> models/*_metrics.json"

# ---- 5. SHAP Explanations ------------------------------------------------
log "Generating SHAP explanations on sample..."
python src/explainability/shap_explainer.py \
  --model  models/random_forest.pkl \
  --data   data/samples/sample_100.csv
ok  "SHAP plots saved -> docs/eda_plots/"

# ---- 6. Done -------------------------------------------------------------
echo
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  Pipeline complete! Launch dashboard with:${NC}"
echo -e "${GREEN}  streamlit run dashboard/app.py${NC}"
echo -e "${GREEN}=====================================================${NC}"
