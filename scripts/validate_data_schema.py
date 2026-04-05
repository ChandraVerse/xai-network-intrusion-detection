#!/usr/bin/env python
"""
scripts/validate_data_schema.py
--------------------------------
Pandera schema validation for all sample CSVs in data/samples/.
Run standalone or via CI (Job: data-validation).

Exit codes
----------
  0  All files pass schema
  1  One or more files fail validation
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"

# ---------------------------------------------------------------------------
# Schema — validated against the CICIDS-2017 feature set
# ---------------------------------------------------------------------------

# Columns that MUST exist in every sample CSV
REQUIRED_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Label",
]

SCHEMA = DataFrameSchema(
    columns={
        "Flow Duration":         Column(float, Check.ge(0), nullable=False),
        "Total Fwd Packets":     Column(float, Check.ge(0), nullable=False),
        "Total Backward Packets":Column(float, Check.ge(0), nullable=False),
        "Flow Bytes/s":          Column(float, nullable=True),   # can be NaN / Inf before cleaning
        "Flow Packets/s":        Column(float, nullable=True),
        "Label":                 Column(str,   Check(lambda s: s.str.len() > 0), nullable=False),
    },
    # Allow extra columns (the full 78-feature set)
    strict=False,
    coerce=True,
)


def validate_file(csv_path: Path) -> bool:
    """Return True if file passes schema, False otherwise."""
    try:
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names (common in CICIDS-2017 exports)
        df.columns = df.columns.str.strip()

        # Check required columns exist before schema validation
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"  [FAIL] {csv_path.name}: missing columns {missing}")
            return False

        SCHEMA.validate(df)
        print(f"  [PASS] {csv_path.name}  ({len(df)} rows, {len(df.columns)} cols)")
        return True

    except pa.errors.SchemaError as e:
        print(f"  [FAIL] {csv_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {csv_path.name}: {e}")
        return False


def main() -> int:
    csv_files = sorted(SAMPLES_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found in {SAMPLES_DIR} — skipping validation.")
        return 0

    print(f"Validating {len(csv_files)} sample CSV(s) in {SAMPLES_DIR}...\n")
    results = [validate_file(f) for f in csv_files]
    passed  = sum(results)
    failed  = len(results) - passed

    print(f"\nResult: {passed}/{len(results)} passed", "✓" if failed == 0 else "✗")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
