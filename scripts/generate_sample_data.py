#!/usr/bin/env python3
"""
generate_sample_data.py
=======================
Generates a synthetic 1000-row CSV with 78 CICIDS-2017 feature columns
and saves it to data/samples/sample_100.csv (name kept for legacy compat).

Usage:
    python scripts/generate_sample_data.py [--rows N] [--seed S] [--quiet]
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FEATURE_COLS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
    "Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
    "URG Flag Count","CWE Flag Count","ECE Flag Count",
    "Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size",
    "Fwd Header Length.1","Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
]

LABEL_MAP = {
    0:"BENIGN",1:"DDoS",2:"PortScan",3:"Bot",4:"Infiltration",
    5:"Web Attack - Brute Force",6:"Web Attack - SQLi",7:"Web Attack - XSS",
    8:"DoS Hulk",9:"DoS GoldenEye",10:"DoS Slowloris",
    11:"DoS Slowhttptest",12:"SSH-Patator",13:"FTP-Patator"
}

def generate(n_rows=1000, seed=42):
    rng = np.random.default_rng(seed)
    # Class distribution roughly mirrors CICIDS-2017 (BENIGN heavy)
    probs = [0.60,0.10,0.08,0.02,0.005,0.015,0.002,0.005,0.08,0.02,0.01,0.01,0.015,0.013]
    labels = rng.choice(len(LABEL_MAP), size=n_rows, p=probs)
    data = rng.uniform(0, 1, size=(n_rows, len(FEATURE_COLS))).astype(np.float32)
    df = pd.DataFrame(data, columns=FEATURE_COLS)
    df["Label"] = [LABEL_MAP[l] for l in labels]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",  type=int, default=1000)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "data", "samples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_100.csv")

    df = generate(args.rows, args.seed)
    df.to_csv(out_path, index=False)
    if not args.quiet:
        print(f"✅  Saved {len(df)} rows × {len(df.columns)} cols → {out_path}")
        print(f"   Label distribution:\n{df['Label'].value_counts().to_string()}")

if __name__ == "__main__":
    main()
