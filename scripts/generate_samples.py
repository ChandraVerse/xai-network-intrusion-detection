#!/usr/bin/env python3
"""
scripts/generate_samples.py

Public API used by tests/conftest.py:
    FEATURE_NAMES  list[str]     -- 78 CICFlowMeter column names
    CLASSES        list[str]     -- 14 CICIDS-2017 attack/benign class labels
    gen_class_samples(label, n, rng) -> list[list[float]]

Also importable as a stand-alone generator:
    python scripts/generate_samples.py --rows 500 --out data/samples/sample_100.csv
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

CLASSES: List[str] = [
    "BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris",
    "DoS Slowhttptest", "FTP-Patator", "SSH-Patator", "PortScan",
    "Web Attack - Brute Force", "Web Attack - XSS",
    "Web Attack - Sql Injection", "Infiltration", "Bot",
]

# Per-class realistic flow statistics (mean, std) derived from CICIDS-2017
_PROFILES = {
    "BENIGN":                    {"dur": (1_200_000, 800_000), "fwd": (12, 8),    "bps": (4500, 3000)},
    "DDoS":                      {"dur": (3_000, 2_000),      "fwd": (47000, 8000), "bps": (980_000, 50_000)},
    "DoS Hulk":                  {"dur": (5_000, 3_000),      "fwd": (3000, 500), "bps": (750_000, 40_000)},
    "DoS GoldenEye":             {"dur": (50_000, 20_000),    "fwd": (150, 40),   "bps": (95_000, 10_000)},
    "DoS Slowloris":             {"dur": (900_000, 200_000),  "fwd": (4, 2),      "bps": (200, 80)},
    "DoS Slowhttptest":          {"dur": (950_000, 250_000),  "fwd": (5, 2),      "bps": (250, 90)},
    "FTP-Patator":               {"dur": (200_000, 80_000),   "fwd": (7, 3),      "bps": (3000, 900)},
    "SSH-Patator":               {"dur": (250_000, 90_000),   "fwd": (8, 3),      "bps": (3500, 1000)},
    "PortScan":                  {"dur": (2_000, 1_500),      "fwd": (1, 0.5),    "bps": (80, 40)},
    "Web Attack - Brute Force":  {"dur": (120_000, 60_000),   "fwd": (20, 8),     "bps": (12_000, 4000)},
    "Web Attack - XSS":          {"dur": (130_000, 55_000),   "fwd": (18, 6),     "bps": (11_000, 3500)},
    "Web Attack - Sql Injection":{"dur": (140_000, 65_000),   "fwd": (22, 9),     "bps": (13_000, 5000)},
    "Infiltration":              {"dur": (600_000, 150_000),  "fwd": (35, 10),    "bps": (8000, 2000)},
    "Bot":                       {"dur": (400_000, 100_000),  "fwd": (15, 5),     "bps": (5000, 1500)},
}


def gen_class_samples(
    label: str,
    n: int,
    rng: np.random.Generator,
) -> List[List[float]]:
    """Return n rows of float features for the given class label."""
    profile = _PROFILES[label]
    rows: List[List[float]] = []
    for _ in range(n):
        dur  = max(1.0, rng.normal(*profile["dur"]))
        fwd  = max(1.0, rng.normal(*profile["fwd"]))
        bps  = max(0.0, rng.normal(*profile["bps"]))
        bwd  = max(0.0, rng.normal(fwd * 0.6, fwd * 0.2 + 1e-9))
        pmean = rng.uniform(40, 1500)
        iat  = dur / fwd

        row: List[float] = [
            float(rng.choice([80, 443, 22, 21, 8080, int(rng.integers(1024, 65535))])),  # Destination Port
            dur,                                     # Flow Duration
            fwd,                                     # Total Fwd Packets
            bwd,                                     # Total Backward Packets
            fwd * rng.uniform(40, 1500),             # Total Length of Fwd Packets
            bwd * rng.uniform(40, 1500),             # Total Length of Bwd Packets
            pmean + rng.uniform(0, 500),             # Fwd Packet Length Max
            max(0, pmean - rng.uniform(0, 400)),     # Fwd Packet Length Min
            pmean,                                   # Fwd Packet Length Mean
            rng.uniform(0, 300),                     # Fwd Packet Length Std
            pmean + rng.uniform(0, 600),             # Bwd Packet Length Max
            max(0, pmean - rng.uniform(0, 400)),     # Bwd Packet Length Min
            pmean * rng.uniform(0.5, 1.2),           # Bwd Packet Length Mean
            rng.uniform(0, 350),                     # Bwd Packet Length Std
            bps,                                     # Flow Bytes/s
            (fwd + bwd) / (dur / 1e6 + 1e-9),       # Flow Packets/s
            iat,                                     # Flow IAT Mean
            rng.uniform(0, iat + 1e-9),              # Flow IAT Std
            iat + rng.uniform(0, iat * 2 + 1e-9),   # Flow IAT Max
            max(0, iat - rng.uniform(0, iat + 1e-9)), # Flow IAT Min
            dur * rng.uniform(0.6, 0.9),             # Fwd IAT Total
            iat * rng.uniform(0.8, 1.1),             # Fwd IAT Mean
            rng.uniform(0, iat * 0.5 + 1e-9),       # Fwd IAT Std
            iat * rng.uniform(1.0, 2.5),             # Fwd IAT Max
            max(0, iat * rng.uniform(0, 0.5)),       # Fwd IAT Min
            dur * rng.uniform(0.4, 0.8),             # Bwd IAT Total
            iat * rng.uniform(0.7, 1.2),             # Bwd IAT Mean
            rng.uniform(0, iat * 0.5 + 1e-9),       # Bwd IAT Std
            iat * rng.uniform(1.0, 2.5),             # Bwd IAT Max
            max(0, iat * rng.uniform(0, 0.5)),       # Bwd IAT Min
            float(rng.integers(0, 2)),               # Fwd PSH Flags
            float(rng.integers(0, 2)),               # Bwd PSH Flags
            0.0,                                     # Fwd URG Flags
            0.0,                                     # Bwd URG Flags
            fwd * 20,                                # Fwd Header Length
            bwd * 20,                                # Bwd Header Length
            fwd / (dur / 1e6 + 1e-9),               # Fwd Packets/s
            bwd / (dur / 1e6 + 1e-9),               # Bwd Packets/s
            max(0, pmean - rng.uniform(0, 400)),     # Min Packet Length
            pmean + rng.uniform(0, 600),             # Max Packet Length
            pmean,                                   # Packet Length Mean
            rng.uniform(0, 350),                     # Packet Length Std
            rng.uniform(0, 350) ** 2,                # Packet Length Variance
            float(rng.integers(0, 3)),               # FIN Flag Count
            float(rng.integers(0, 3)),               # SYN Flag Count
            float(rng.integers(0, 2)),               # RST Flag Count
            float(rng.integers(0, 5)),               # PSH Flag Count
            float(rng.integers(0, 10)),              # ACK Flag Count
            0.0,                                     # URG Flag Count
            0.0,                                     # CWE Flag Count
            float(rng.integers(0, 2)),               # ECE Flag Count
            bwd / fwd,                               # Down/Up Ratio
            pmean,                                   # Average Packet Size
            pmean,                                   # Avg Fwd Segment Size
            pmean * rng.uniform(0.5, 1.2),           # Avg Bwd Segment Size
            fwd * 20,                                # Fwd Header Length.1
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,          # Bulk features (6)
            fwd,                                     # Subflow Fwd Packets
            fwd * pmean,                             # Subflow Fwd Bytes
            bwd,                                     # Subflow Bwd Packets
            bwd * pmean,                             # Subflow Bwd Bytes
            float(rng.choice([8192, 16384, 32768, 65535, -1])),  # Init_Win_bytes_forward
            float(rng.choice([8192, 16384, 32768, 65535, -1])),  # Init_Win_bytes_backward
            max(0, fwd - 1),                         # act_data_pkt_fwd
            20.0,                                    # min_seg_size_forward
            rng.uniform(0, dur * 0.3),               # Active Mean
            rng.uniform(0, dur * 0.15 + 1e-9),      # Active Std
            rng.uniform(0, dur * 0.3),               # Active Max
            rng.uniform(0, dur * 0.15 + 1e-9),      # Active Min
            rng.uniform(0, dur * 0.5),               # Idle Mean
            rng.uniform(0, dur * 0.25 + 1e-9),      # Idle Std
            rng.uniform(0, dur * 0.5),               # Idle Max
            rng.uniform(0, dur * 0.25 + 1e-9),      # Idle Min
        ]
        rows.append(row)
    return rows


def generate_csv(rows: int, out: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    weights = [0.60] + [0.40 / (len(CLASSES) - 1)] * (len(CLASSES) - 1)
    labels = rng.choice(CLASSES, size=rows, p=weights)
    data = [gen_class_samples(lbl, 1, rng)[0] + [lbl] for lbl in labels]
    df = pd.DataFrame(data, columns=FEATURE_NAMES + ["Label"])
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CICIDS-2017 sample CSV")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--out",  type=str, default="data/samples/sample_100.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate_csv(args.rows, args.out, args.seed)
    print(f"[OK] {args.rows} rows -> {args.out}  shape={df.shape}")
    print(f"     {df['Label'].value_counts().to_dict()}")
