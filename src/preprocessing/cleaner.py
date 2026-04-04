"""Data cleaner for CICIDS-2017 raw CSV files.

Handles: infinite values, NaN filling, zero-variance features,
duplicate rows, whitespace-stripped column names.

Usage (CLI):
    python src/preprocessing/cleaner.py \\
        --input  data/raw/ \\
        --output data/processed/

Outputs:
    data/processed/combined_clean.csv   Merged + cleaned dataframe
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LABEL_COL = " Label"  # raw column name in CICIDS-2017 CSVs (note leading space)


def load_raw_csvs(raw_dir: str | Path) -> pd.DataFrame:
    """Concatenate all CSV files in raw_dir into a single DataFrame."""
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for f in csv_files:
        log.info("Reading %s", f.name)
        df = pd.read_csv(f, low_memory=False)
        log.info("  → %d rows, %d cols", *df.shape)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined shape: %s", combined.shape)
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline — returns cleaned DataFrame."""
    original_shape = df.shape

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()
    log.info("Columns normalised")

    # 2. Replace Inf / -Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    inf_count = df.isnull().sum().sum()
    log.info("Replaced Inf values → %d NaN introduced", inf_count)

    # 3. Fill NaN with per-column median (numeric cols only)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    log.info("NaN values filled with column medians")

    # 4. Drop zero-variance columns
    zero_var = [c for c in num_cols if df[c].std() == 0]
    if zero_var:
        df.drop(columns=zero_var, inplace=True)
        log.info("Dropped %d zero-variance columns: %s", len(zero_var), zero_var)

    # 5. Drop exact duplicate rows
    n_dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    log.info("Dropped %d duplicate rows", n_dupes)

    # 6. Drop rows where label is missing
    label_col = "Label"  # after stripping
    if label_col in df.columns:
        before = len(df)
        df.dropna(subset=[label_col], inplace=True)
        log.info("Dropped %d rows with missing label", before - len(df))

    log.info(
        "Cleaning complete: %s → %s  (removed %d rows)",
        original_shape,
        df.shape,
        original_shape[0] - df.shape[0],
    )
    return df


def save(df: pd.DataFrame, out_dir: str | Path, filename: str = "combined_clean.csv") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / filename
    df.to_csv(out_path, index=False)
    log.info("Saved → %s  (%d rows)", out_path, len(df))
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean CICIDS-2017 raw CSVs")
    p.add_argument("--input",  default="data/raw/",       help="Directory of raw CICIDS-2017 CSVs")
    p.add_argument("--output", default="data/processed/", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_raw_csvs(args.input)
    df = clean(df)
    save(df, args.output)


if __name__ == "__main__":
    main()
