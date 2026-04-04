"""MinMaxScaler wrapper for XAI-NIDS preprocessing.

Fits MinMaxScaler on training data only (no data leakage).
Transforms both train and test sets. Saves fitted scaler as a pickle.

Usage (CLI):
    python src/preprocessing/scaler.py \\
        --train data/processed/train_split.csv \\
        --test  data/processed/test_split.csv \\
        --out   data/processed/

Outputs:
    data/processed/train_scaled.csv
    data/processed/test_scaled.csv
    data/processed/minmax_scaler.pkl
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def fit_transform(
    train_path: str | Path,
    test_path: str | Path,
    label_col: str = "label_encoded",
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Fit scaler on train, transform both splits. Returns scaled DataFrames + scaler."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_cols = [c for c in train.columns if c != label_col]
    log.info("Fitting MinMaxScaler on %d training samples, %d features",
             len(train), len(feature_cols))

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit on training features ONLY
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    # Apply (not fit) on test
    test[feature_cols] = scaler.transform(test[feature_cols])

    log.info("Scaling complete.  Feature range: [0, 1]")
    return train, test, scaler


def save(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scaler: MinMaxScaler,
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.to_csv(out / "train_scaled.csv", index=False)
    test.to_csv(out / "test_scaled.csv",   index=False)
    joblib.dump(scaler, out / "minmax_scaler.pkl")

    log.info("Saved train_scaled.csv, test_scaled.csv, minmax_scaler.pkl → %s", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scale train/test splits with MinMaxScaler")
    p.add_argument("--train", default="data/processed/train_split.csv")
    p.add_argument("--test",  default="data/processed/test_split.csv")
    p.add_argument("--out",   default="data/processed/")
    p.add_argument("--label", default="label_encoded")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train, test, scaler = fit_transform(args.train, args.test, label_col=args.label)
    save(train, test, scaler, args.out)


if __name__ == "__main__":
    main()
