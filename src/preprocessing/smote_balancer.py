"""SMOTE oversampler for XAI-NIDS minority class balancing.

Applied to the TRAINING SET ONLY after the train/test split.
Raises all minority classes to match the majority class count.

Usage (CLI):
    python src/preprocessing/smote_balancer.py \\
        --train data/processed/train_scaled.csv \\
        --out   data/processed/

Outputs:
    data/processed/train_balanced.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def balance(
    train_path: str | Path,
    label_col: str = "label_encoded",
) -> pd.DataFrame:
    """Load training CSV, apply SMOTE, return balanced DataFrame."""
    df = pd.read_csv(train_path)
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values
    y = df[label_col].values

    log.info("Pre-SMOTE class distribution: %s", dict(Counter(y)))

    smote = SMOTE(
        sampling_strategy="not majority",  # raises all minority to match majority
        k_neighbors=5,
        random_state=42,
        n_jobs=-1,
    )
    X_res, y_res = smote.fit_resample(X, y)

    log.info("Post-SMOTE class distribution: %s", dict(Counter(y_res)))
    log.info(
        "Balanced training set: %d → %d rows  (+%d synthetic)",
        len(X), len(X_res), len(X_res) - len(X),
    )

    balanced = pd.DataFrame(X_res, columns=feature_cols)
    balanced[label_col] = y_res
    return balanced


def save(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "train_balanced.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved → %s  (%d rows)", out_path, len(df))
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMOTE balance the training set")
    p.add_argument("--train", default="data/processed/train_scaled.csv")
    p.add_argument("--out",   default="data/processed/")
    p.add_argument("--label", default="label_encoded")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    balanced = balance(args.train, label_col=args.label)
    save(balanced, args.out)


if __name__ == "__main__":
    main()
