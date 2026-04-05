#!/usr/bin/env bash
# =============================================================================
# run_tests.sh  —  Run all unit + integration tests
#
# Usage:
#   bash scripts/run_tests.sh [--verbose] [--coverage]
#
# Flags:
#   --verbose    Show per-test output (passes -v to pytest)
#   --coverage   Generate HTML coverage report in htmlcov/
# =============================================================================

set -euo pipefail

VERBOSE=false
COVERAGE=false
for arg in "$@"; do
  [[ "$arg" == "--verbose"  ]] && VERBOSE=true
  [[ "$arg" == "--coverage" ]] && COVERAGE=true
done

BLUE='\033[0;34m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${BLUE}[TESTS]${NC} $1"; }
ok()   { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ---- Guard ---------------------------------------------------------------
[[ -f requirements.txt ]] || fail "Run from repo root."
python -c "import pytest" 2>/dev/null || pip install pytest pytest-cov -q

# ---- Ensure sample data exists -------------------------------------------
if [[ ! -f data/samples/sample_100.csv ]]; then
  log "Sample data not found. Generating..."
  python scripts/generate_sample_data.py --rows 500 --out data/samples/sample_100.csv
  ok  "Sample data generated."
fi

# ---- Build pytest args ---------------------------------------------------
PYTEST_ARGS="tests/"
$VERBOSE  && PYTEST_ARGS="-v $PYTEST_ARGS"
$COVERAGE && PYTEST_ARGS="--cov=src --cov-report=html --cov-report=term-missing $PYTEST_ARGS"

# ---- Run -----------------------------------------------------------------
log "Running: pytest $PYTEST_ARGS"
python -m pytest $PYTEST_ARGS

echo
if $COVERAGE; then
  ok "Coverage report generated -> htmlcov/index.html"
fi
ok "All tests passed."
