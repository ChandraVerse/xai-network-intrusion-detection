#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — Run full test suite with coverage report
# Author: Chandra Sekhar Chakraborty
# Usage:  bash scripts/run_tests.sh [--fast] [--no-cov]
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$ROOT"

FAST=false; COV=true
for arg in "$@"; do
  case $arg in
    --fast)   FAST=true ;;
    --no-cov) COV=false ;;
  esac
done

echo "╔══════════════════════════════════════════╗"
echo "║         XAI-NIDS Test Runner             ║"
echo "╚══════════════════════════════════════════╝"

# Ensure fixtures exist
python scripts/generate_sample_data.py --quiet 2>/dev/null || true

if [ "$COV" = true ]; then
  COV_FLAGS="--cov=src --cov-report=term-missing --cov-report=html:reports/coverage"
else
  COV_FLAGS=""
fi

if [ "$FAST" = true ]; then
  echo "Running fast tests (unit only)…"
  python -m pytest tests/unit/ -v $COV_FLAGS -x --tb=short
else
  echo "Running full test suite…"
  python -m pytest tests/ -v $COV_FLAGS --tb=short 2>&1 | tee reports/test_output.txt
fi

echo ""
echo "✅  Tests complete."
[ "$COV" = true ] && echo "    Coverage report: reports/coverage/index.html"
