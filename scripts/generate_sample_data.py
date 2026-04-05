#!/usr/bin/env python3
"""
generate_sample_data.py

Generates a synthetic 500-row sample CSV with realistic CICIDS-2017 feature
distributions. Output: data/samples/sample_100.csv

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --rows 1000 --out data/samples/my_sample.csv
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

FEATURE_COLS_PATH = "data/processed/consensus_features.json"
LABEL_MAP_PATH = "data/processed/label_map.json"
DEFAULT_OUT = "data/samples/sample_100.csv"

ATTACK_LABELS = [
    "BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris",
    "DoS Slowhttptest", "FTP-Patator", "SSH-Patator", "PortScan",
    "Web Attack - Brute Force", "Web Attack - XSS",
    "Web Attack - Sql Injection", "Infiltration", "Bot",
]

# Realistic per-class flow statistics (mean, std) tuned from CICIDS-2017 paper
CLASS_PROFILES = {
    "BENIGN":                    {"flow_duration": (1_200_000, 800_000),  "fwd_packets": (12, 8),   "bytes_s": (4500, 3000)},
    "DDoS":                      {"flow_duration": (3_000,    2_000),    "fwd_packets": (47000, 8000), "bytes_s": (980_000, 50_000)},
    "DoS Hulk":                  {"flow_duration": (5_000,    3_000),    "fwd_packets": (3000, 500),  "bytes_s": (750_000, 40_000)},
    "DoS GoldenEye":             {"flow_duration": (50_000,   20_000),   "fwd_packets": (150, 40),   "bytes_s": (95_000, 10_000)},
    "DoS Slowloris":             {"flow_duration": (900_000,  200_000),  "fwd_packets": (4, 2),      "bytes_s": (200, 80)},
    "DoS Slowhttptest":          {"flow_duration": (950_000,  250_000),  "fwd_packets": (5, 2),      "bytes_s": (250, 90)},
    "FTP-Patator":               {"flow_duration": (200_000,  80_000),   "fwd_packets": (7, 3),      "bytes_s": (3000, 900)},
    "SSH-Patator":               {"flow_duration": (250_000,  90_000),   "fwd_packets": (8, 3),      "bytes_s": (3500, 1000)},
    "PortScan":                  {"flow_duration": (2_000,    1_500),    "fwd_packets": (1, 0.5),    "bytes_s": (80, 40)},
    "Web Attack - Brute Force":  {"flow_duration": (120_000,  60_000),   "fwd_packets": (20, 8),     "bytes_s": (12_000, 4000)},
    "Web Attack - XSS":          {"flow_duration": (130_000,  55_000),   "fwd_packets": (18, 6),     "bytes_s": (11_000, 3500)},
    "Web Attack - Sql Injection":{"flow_duration": (140_000,  65_000),   "fwd_packets": (22, 9),     "bytes_s": (13_000, 5000)},
    "Infiltration":              {"flow_duration": (600_000,  150_000),  "fwd_packets": (35, 10),    "bytes_s": (8000, 2000)},
    "Bot":                       {"flow_duration": (400_000,  100_000),  "fwd_packets": (15, 5),     "bytes_s": (5000, 1500)},
}

FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def generate_row(label: str, rng: np.random.Generator) -> dict:
    profile = CLASS_PROFILES[label]
    row = {}

    flow_dur = max(1, rng.normal(*profile["flow_duration"]))
    fwd_pkts = max(1, rng.normal(*profile["fwd_packets"]))
    bytes_s   = max(0, rng.normal(*profile["bytes_s"]))

    row["Destination Port"]              = rng.choice([80, 443, 22, 21, 8080, rng.integers(1024, 65535)])
    row["Flow Duration"]                 = flow_dur
    row["Total Fwd Packets"]             = int(fwd_pkts)
    row["Total Backward Packets"]        = int(max(0, rng.normal(fwd_pkts * 0.6, fwd_pkts * 0.2)))
    row["Total Length of Fwd Packets"]   = fwd_pkts * rng.uniform(40, 1500)
    row["Total Length of Bwd Packets"]   = row["Total Backward Packets"] * rng.uniform(40, 1500)
    pkt_len_mean = rng.uniform(40, 1500)
    row["Fwd Packet Length Max"]         = pkt_len_mean + rng.uniform(0, 500)
    row["Fwd Packet Length Min"]         = max(0, pkt_len_mean - rng.uniform(0, 400))
    row["Fwd Packet Length Mean"]        = pkt_len_mean
    row["Fwd Packet Length Std"]         = rng.uniform(0, 300)
    row["Bwd Packet Length Max"]         = pkt_len_mean + rng.uniform(0, 600)
    row["Bwd Packet Length Min"]         = max(0, pkt_len_mean - rng.uniform(0, 400))
    row["Bwd Packet Length Mean"]        = pkt_len_mean * rng.uniform(0.5, 1.2)
    row["Bwd Packet Length Std"]         = rng.uniform(0, 350)
    row["Flow Bytes/s"]                  = bytes_s
    row["Flow Packets/s"]               = (fwd_pkts + row["Total Backward Packets"]) / (flow_dur / 1e6 + 1e-9)
    iat_mean = flow_dur / max(fwd_pkts, 1)
    row["Flow IAT Mean"]                 = iat_mean
    row["Flow IAT Std"]                  = rng.uniform(0, iat_mean)
    row["Flow IAT Max"]                  = iat_mean + rng.uniform(0, iat_mean * 2)
    row["Flow IAT Min"]                  = max(0, iat_mean - rng.uniform(0, iat_mean))
    row["Fwd IAT Total"]                 = flow_dur * rng.uniform(0.6, 0.9)
    row["Fwd IAT Mean"]                  = iat_mean * rng.uniform(0.8, 1.1)
    row["Fwd IAT Std"]                   = rng.uniform(0, iat_mean * 0.5)
    row["Fwd IAT Max"]                   = iat_mean * rng.uniform(1.0, 2.5)
    row["Fwd IAT Min"]                   = max(0, iat_mean * rng.uniform(0, 0.5))
    row["Bwd IAT Total"]                 = flow_dur * rng.uniform(0.4, 0.8)
    row["Bwd IAT Mean"]                  = iat_mean * rng.uniform(0.7, 1.2)
    row["Bwd IAT Std"]                   = rng.uniform(0, iat_mean * 0.5)
    row["Bwd IAT Max"]                   = iat_mean * rng.uniform(1.0, 2.5)
    row["Bwd IAT Min"]                   = max(0, iat_mean * rng.uniform(0, 0.5))
    row["Fwd PSH Flags"]                 = rng.integers(0, 2)
    row["Bwd PSH Flags"]                 = rng.integers(0, 2)
    row["Fwd URG Flags"]                 = 0
    row["Bwd URG Flags"]                 = 0
    row["Fwd Header Length"]             = int(fwd_pkts * 20)
    row["Bwd Header Length"]             = int(row["Total Backward Packets"] * 20)
    row["Fwd Packets/s"]                 = fwd_pkts / (flow_dur / 1e6 + 1e-9)
    row["Bwd Packets/s"]                 = row["Total Backward Packets"] / (flow_dur / 1e6 + 1e-9)
    row["Min Packet Length"]             = max(0, pkt_len_mean - rng.uniform(0, 400))
    row["Max Packet Length"]             = pkt_len_mean + rng.uniform(0, 600)
    row["Packet Length Mean"]            = pkt_len_mean
    row["Packet Length Std"]             = rng.uniform(0, 350)
    row["Packet Length Variance"]        = row["Packet Length Std"] ** 2
    row["FIN Flag Count"]                = rng.integers(0, 3)
    row["SYN Flag Count"]                = rng.integers(0, 3)
    row["RST Flag Count"]                = rng.integers(0, 2)
    row["PSH Flag Count"]                = rng.integers(0, 5)
    row["ACK Flag Count"]                = rng.integers(0, 10)
    row["URG Flag Count"]                = 0
    row["CWE Flag Count"]                = 0
    row["ECE Flag Count"]                = rng.integers(0, 2)
    row["Down/Up Ratio"]                 = row["Total Backward Packets"] / max(fwd_pkts, 1)
    row["Average Packet Size"]           = pkt_len_mean
    row["Avg Fwd Segment Size"]          = pkt_len_mean
    row["Avg Bwd Segment Size"]          = pkt_len_mean * rng.uniform(0.5, 1.2)
    row["Fwd Header Length.1"]           = row["Fwd Header Length"]
    row["Fwd Avg Bytes/Bulk"]            = 0
    row["Fwd Avg Packets/Bulk"]          = 0
    row["Fwd Avg Bulk Rate"]             = 0
    row["Bwd Avg Bytes/Bulk"]            = 0
    row["Bwd Avg Packets/Bulk"]          = 0
    row["Bwd Avg Bulk Rate"]             = 0
    row["Subflow Fwd Packets"]           = int(fwd_pkts)
    row["Subflow Fwd Bytes"]             = int(row["Total Length of Fwd Packets"])
    row["Subflow Bwd Packets"]           = row["Total Backward Packets"]
    row["Subflow Bwd Bytes"]             = int(row["Total Length of Bwd Packets"])
    row["Init_Win_bytes_forward"]        = rng.choice([8192, 16384, 32768, 65535, -1])
    row["Init_Win_bytes_backward"]       = rng.choice([8192, 16384, 32768, 65535, -1])
    row["act_data_pkt_fwd"]              = int(max(0, fwd_pkts - 1))
    row["min_seg_size_forward"]          = 20
    active_mean = rng.uniform(0, flow_dur * 0.3)
    row["Active Mean"]                   = active_mean
    row["Active Std"]                    = rng.uniform(0, active_mean * 0.5)
    row["Active Max"]                    = active_mean + rng.uniform(0, active_mean)
    row["Active Min"]                    = max(0, active_mean - rng.uniform(0, active_mean * 0.5))
    idle_mean = rng.uniform(0, flow_dur * 0.5)
    row["Idle Mean"]                     = idle_mean
    row["Idle Std"]                      = rng.uniform(0, idle_mean * 0.5)
    row["Idle Max"]                      = idle_mean + rng.uniform(0, idle_mean)
    row["Idle Min"]                      = max(0, idle_mean - rng.uniform(0, idle_mean * 0.5))
    row["Label"]                         = label
    return row


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CICIDS-2017 sample CSV")
    parser.add_argument("--rows",  type=int, default=500,           help="Number of rows to generate (default: 500)")
    parser.add_argument("--out",   type=str, default=DEFAULT_OUT,   help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument("--seed",  type=int, default=42,            help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Distribute rows across attack classes (weighted toward BENIGN)
    weights = [0.60] + [0.40 / (len(ATTACK_LABELS) - 1)] * (len(ATTACK_LABELS) - 1)
    labels  = rng.choice(ATTACK_LABELS, size=args.rows, p=weights)

    rows = [generate_row(lbl, rng) for lbl in labels]
    df   = pd.DataFrame(rows, columns=FEATURE_NAMES + ["Label"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Generated {args.rows} synthetic flows -> {args.out}")
    print(f"     Shape  : {df.shape}")
    print(f"     Classes: {df['Label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
