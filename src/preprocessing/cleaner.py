"""Data cleaner for CICIDS-2017 raw CSV files.

Handles: infinite values, NaN filling, zero-variance column removal,
non-numeric/identifier column removal, whitespace-stripping in column
names, and label normalisation.

Public API
----------
clean_dataframe(df, label_col)     -- clean an in-memory DataFrame (used by tests)
load_and_clean_cicids(raw_dir)     -- load CSVs from a directory then clean

Usage::
    from src.preprocessing.cleaner import clean_dataframe, load_and_clean_cicids
    df_clean = clean_dataframe(df_raw)
    df_clean = load_and_clean_cicids("data/raw/MachineLearningCVE")
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# Known label normalisation map for CICIDS-2017 raw CSV
_LABEL_MAP = {
    "Web Attack  Brute Force": "Web Attack - Brute Force",
    "Web Attack  XSS": "Web Attack - XSS",
    "Web Attack  Sql Injection": "Web Attack - SQLi",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS slowloris": "DoS Slowloris",
    "DoS Hulk": "DoS Hulk",
    "DoS GoldenEye": "DoS GoldenEye",
    "BENIGN": "BENIGN",
}

# Non-feature identifier columns that must be dropped
_ID_COLUMNS = {
    "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Destination Port", "Protocol",
    "Timestamp",
}


# ---------------------------------------------------------------------------
# Core clean function  (public, used directly by tests)
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Clean a CICIDS-style DataFrame in memory.

    Steps applied in order:
    1. Strip whitespace from all column names.
    2. Auto-detect label column (column whose stripped name contains 'label'
       or 'Label') if *label_col* is not provided.
    3. Replace inf / -inf with NaN.
    4. Fill NaN in numeric columns with the column median.
    5. Drop zero-variance numeric columns (excluding the label column).
    6. Drop non-numeric identifier columns (Flow ID, Source IP, etc.).
    7. Normalise label strings using the CICIDS-2017 label map.

    Args:
        df:        Raw DataFrame (will not be mutated; a copy is made).
        label_col: Name of the label column.  If None, the function
                   auto-detects the first column whose name (stripped)
                   contains 'Label' or 'label'.

    Returns:
        Cleaned DataFrame with numeric features + label column only.
    """
    df = df.copy()

    # 1. Strip column name whitespace
    df.columns = df.columns.str.strip()

    # 2. Auto-detect label column
    if label_col is None:
        for col in df.columns:
            if "label" in col.lower():
                label_col = col
                break

    # 3. Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4. Fill NaN with column median (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 5. Remove zero-variance numeric columns (excluding label)
    feature_num_cols = [c for c in numeric_cols if c != label_col]
    variances = df[feature_num_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        log.info("Dropping %d zero-variance columns: %s", len(zero_var), zero_var)
        df = df.drop(columns=zero_var)

    # 6. Drop non-numeric identifier columns
    id_cols_present = [c for c in df.columns if c in _ID_COLUMNS]
    if id_cols_present:
        log.info("Dropping identifier columns: %s", id_cols_present)
        df = df.drop(columns=id_cols_present)
    # Also drop any remaining non-numeric, non-label columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_non_label = [c for c in non_numeric if c != label_col]
    if non_numeric_non_label:
        log.info("Dropping non-numeric columns: %s", non_numeric_non_label)
        df = df.drop(columns=non_numeric_non_label)

    # 7. Normalise label strings
    if label_col and label_col in df.columns:
        df[label_col] = (
            df[label_col]
            .astype(str)
            .str.strip()
            .replace(_LABEL_MAP)
        )

    log.info("Cleaned shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Directory-level loader (original API — unchanged)
# ---------------------------------------------------------------------------

def _load_csv_dir(directory: str | Path) -> pd.DataFrame:
    """Concatenate all CSV files in *directory* into a single DataFrame."""
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    dfs = []
    for fp in csv_files:
        log.info("  Loading %s ...", fp.name)
        raw = pd.read_csv(fp, low_memory=False, encoding="utf-8")
        raw.columns = raw.columns.str.strip()
        dfs.append(raw)

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Combined shape (raw): %s", combined.shape)
    return combined


def load_and_clean_cicids(
    raw_dir: str | Path,
    label_col: str = "Label",
) -> pd.DataFrame:
    """Load raw CICIDS-2017 CSVs, clean, and return a ready DataFrame."""
    log.info("Loading raw CSVs from: %s", os.path.abspath(raw_dir))
    df = _load_csv_dir(raw_dir)
    log.info("Cleaning ...")
    return clean_dataframe(df, label_col=label_col)
