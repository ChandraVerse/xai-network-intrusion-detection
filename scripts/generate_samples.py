"""
scripts/generate_samples.py
----------------------------
Synthetic CICIDS-2017-compatible sample generator.

Exports used by tests/conftest.py:
    FEATURE_NAMES  list[str]  -- 78 CICIDS-2017 feature column names
    CLASSES        list[str]  -- 14 attack/benign class labels
    gen_class_samples(cls, n, rng) -> list[list[float]]

Also runnable as a CLI script to generate data/samples/sample_100.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

assert len(FEATURE_NAMES) == 78, f"Expected 78 features, got {len(FEATURE_NAMES)}"

CLASSES: list[str] = [
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS Slowloris",
    "DoS Slowhttptest",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Bot",
    "Infiltration",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - SQLi",
]

assert len(CLASSES) == 14, f"Expected 14 classes, got {len(CLASSES)}"

# Per-class mean offsets so each class is statistically distinguishable
_CLASS_OFFSETS: dict[str, float] = {
    "BENIGN":                   0.0,
    "DDoS":                     0.6,
    "DoS Hulk":                 0.5,
    "DoS GoldenEye":            0.4,
    "DoS Slowloris":            0.35,
    "DoS Slowhttptest":         0.3,
    "PortScan":                 0.7,
    "FTP-Patator":              0.55,
    "SSH-Patator":              0.45,
    "Bot":                      0.25,
    "Infiltration":             0.15,
    "Web Attack - Brute Force": 0.5,
    "Web Attack - XSS":         0.4,
    "Web Attack - SQLi":        0.35,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def gen_class_samples(
    cls: str,
    n: int,
    rng: np.random.Generator,
) -> list[list[float]]:
    """Return *n* synthetic feature rows for class *cls*.

    Values are drawn from U[0,1] shifted by a class-specific offset
    and clipped to [0, 1] so the result mimics MinMax-scaled data.

    Args:
        cls: Class name -- must be in CLASSES.
        n:   Number of rows to generate.
        rng: Seeded numpy random Generator for reproducibility.

    Returns:
        List of n lists, each of length 78 (float values in [0, 1]).
    """
    if cls not in _CLASS_OFFSETS:
        raise ValueError(f"Unknown class '{cls}'. Must be one of: {CLASSES}")

    offset = _CLASS_OFFSETS[cls]
    raw = rng.random((n, len(FEATURE_NAMES))) + offset
    clipped = np.clip(raw, 0.0, 1.0).astype(np.float32)
    return clipped.tolist()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _generate_csv(out_path: Path, total_rows: int = 112) -> None:
    """Write a balanced synthetic CSV with all 14 classes.

    Output schema: 78 feature columns + 'Label' = 79 columns total.
    Rows are distributed as evenly as possible across all 14 classes
    so that the total is exactly *total_rows*.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    n_classes = len(CLASSES)
    base = total_rows // n_classes
    remainder = total_rows % n_classes
    # distribute remainder across the first classes
    counts = [base + (1 if i < remainder else 0) for i in range(n_classes)]

    # 79 columns: 78 features + Label
    header = FEATURE_NAMES + ["Label"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for cls, n in zip(CLASSES, counts):
            rows = gen_class_samples(cls, n, rng)
            for row in rows:
                writer.writerow([f"{v:.6f}" for v in row] + [cls])

    print(f"Generated {total_rows} rows x {len(header)} cols -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic CICIDS-2017 sample CSV"
    )
    parser.add_argument(
        "--out", default="data/samples/sample_100.csv",
        help="Output CSV path (default: data/samples/sample_100.csv)",
    )
    # --rows is the canonical name; --rows-per-class kept as alias for
    # backwards-compatibility with any local scripts.
    rows_group = parser.add_mutually_exclusive_group()
    rows_group.add_argument(
        "--rows", type=int, default=None,
        help="Total number of rows to generate (distributed across 14 classes).",
    )
    rows_group.add_argument(
        "--rows-per-class", type=int, default=None,
        help="Rows per class (14 classes total). Legacy alias for --rows.",
    )
    args = parser.parse_args()

    if args.rows is not None:
        total = args.rows
    elif args.rows_per_class is not None:
        total = args.rows_per_class * len(CLASSES)
    else:
        total = 112  # default: 8 per class

    _generate_csv(Path(args.out), total)
