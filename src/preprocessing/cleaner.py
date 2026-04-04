"""
cleaner.py
----------
Handles NaN, Inf, and zero-variance feature removal for CICIDS-2017 flows.

Usage:
    python src/preprocessing/cleaner.py --input data/raw/ --output data/processed/
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd

# Features to always drop (labels, non-numeric identifiers)
DROP_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP",
             "Destination Port", "Protocol", "Timestamp"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a CICIDS-2017 flow DataFrame:
      1. Strip column name whitespace
      2. Replace Inf/-Inf with NaN
      3. Fill NaN with column median
      4. Drop zero-variance columns
      5. Drop non-feature identifier columns

    Returns the cleaned DataFrame.
    """
    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Drop non-feature columns that exist
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 3. Replace Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4. Fill NaN with column median (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 5. Drop zero-variance columns
    zero_var = [c for c in numeric_cols if df[c].std() == 0]
    if zero_var:
        print(f"  [cleaner] Dropping {len(zero_var)} zero-variance columns: {zero_var}")
        df = df.drop(columns=zero_var)

    return df


def load_and_clean_cicids(input_dir: str) -> pd.DataFrame:
    """Load all CSVs from input_dir, concatenate, and clean."""
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dfs = []
    for f in sorted(csv_files):
        print(f"  [cleaner] Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  [cleaner] Combined shape: {combined.shape}")
    return clean_dataframe(combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CICIDS-2017 raw CSVs")
    parser.add_argument("--input",  required=True, help="Directory of raw CSVs")
    parser.add_argument("--output", required=True, help="Directory for cleaned output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cleaned = load_and_clean_cicids(args.input)
    out_path = os.path.join(args.output, "cleaned.csv")
    cleaned.to_csv(out_path, index=False)
    print(f"  [cleaner] Saved cleaned data to {out_path}")
