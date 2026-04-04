"""Data cleaner for CICIDS-2017 raw CSV files.

Handles: infinite values, NaN filling, zero-variance column removal,
whitespace-stripping in column names, and label normalisation.

Usage:
    from src.preprocessing.cleaner import load_and_clean_cicids
    df = load_and_clean_cicids("data/raw/MachineLearningCVE")
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


def _load_csv_dir(directory: str | Path) -> pd.DataFrame:
    """Concatenate all CSV files in *directory* into a single DataFrame."""
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    dfs = []
    for fp in csv_files:
        log.info("  Loading %s ...", fp.name)
        df = pd.read_csv(fp, low_memory=False, encoding="utf-8")
        df.columns = df.columns.str.strip()  # strip leading/trailing spaces
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Combined shape (raw): %s", combined.shape)
    return combined


def _clean(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Apply all cleaning steps and return the cleaned DataFrame."""
    # 1. Replace inf/-inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    inf_replaced = df.isna().sum().sum()
    log.info("  Inf values replaced with NaN: %d", inf_replaced)

    # 2. Fill NaN with column median (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    log.info("  NaN filled with column median for %d numeric columns", len(numeric_cols))

    # 3. Remove zero-variance columns (except label)
    feature_cols = [c for c in numeric_cols if c != label_col]
    variances = df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        log.info("  Dropping %d zero-variance columns: %s", len(zero_var), zero_var)
        df = df.drop(columns=zero_var)
    else:
        log.info("  No zero-variance columns found.")

    # 4. Normalise label strings
    if label_col in df.columns:
        df[label_col] = (
            df[label_col]
            .str.strip()
            .replace(_LABEL_MAP)
        )
        log.info("  Label counts:\n%s", df[label_col].value_counts().to_string())

    log.info("  Cleaned shape: %s", df.shape)
    return df


def load_and_clean_cicids(
    raw_dir: str | Path,
    label_col: str = "Label",
) -> pd.DataFrame:
    """Load raw CICIDS-2017 CSVs, clean, and return a ready DataFrame."""
    log.info("Loading raw CSVs from: %s", os.path.abspath(raw_dir))
    df = _load_csv_dir(raw_dir)
    log.info("Cleaning ...")
    df = _clean(df, label_col=label_col)
    return df
