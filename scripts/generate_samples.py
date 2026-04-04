"""
generate_samples.py
-------------------
Generates synthetic sample CSVs for data/samples/ from CICIDS-2017
feature distributions (MinMax-scaled profiles, random_state=42).

Run from repo root:
    python scripts/generate_samples.py

Outputs:
    data/samples/sample_10rows.csv
    data/samples/sample_100rows.csv
    data/samples/sample_benign_only.csv
    data/samples/sample_attack_mix.csv

All values are MinMax-scaled to [0, 1] — matching the output of
data/processed/scaler.pkl. Do NOT pass these through scaler.transform().
"""

import os
import json
import numpy as np
import pandas as pd

OUT_DIR = "data/samples"
os.makedirs(OUT_DIR, exist_ok=True)

# 78 feature columns (CICIDS-2017 after cleaning)
FEATURE_NAMES = [
    "Flow Duration","Total Fwd Packets","Total Backward Packets",
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
    "Fwd Header Length.1",
    "Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes",
    "Init_Win_bytes_forward","Init_Win_bytes_backward",
    "act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
    "Inbound",
    "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate",
]

CLASSES = [
    "BENIGN","Bot","DDoS","DoS GoldenEye","DoS Hulk",
    "DoS Slowhttptest","DoS Slowloris","FTP-Patator",
    "Infiltration","PortScan","SSH-Patator",
    "Web Attack - Brute Force","Web Attack - SQLi","Web Attack - XSS",
]


def gen_class_samples(label: str, n: int, rng: np.random.Generator) -> list:
    """Generate n synthetic MinMax-scaled feature vectors for a given class."""
    rows = []
    for _ in range(n):
        if label == "BENIGN":
            f = rng.uniform(0.0, 0.4, 78)
            f[0]  = rng.uniform(0.3, 0.9)   # long flow duration
            f[13] = rng.uniform(0.1, 0.5)   # moderate bytes/s
            f[46] = rng.choice([0, 1], p=[0.3, 0.7])  # ACK dominant
        elif label == "DDoS":
            f = rng.uniform(0.0, 0.2, 78)
            f[0]  = rng.uniform(0.0, 0.1)   # very short duration
            f[13] = rng.uniform(0.7, 1.0)   # very high bytes/s
            f[14] = rng.uniform(0.8, 1.0)   # very high packets/s
            f[43] = rng.choice([0, 1])       # SYN flood indicator
        elif label == "PortScan":
            f = rng.uniform(0.0, 0.15, 78)
            f[0]  = rng.uniform(0.0, 0.05)  # extremely short flows
            f[1]  = rng.uniform(0.0, 0.1)   # few forward packets
            f[43] = 1.0                      # SYN always set
            f[44] = rng.choice([0, 1])       # RST common
        elif label == "DoS Hulk":
            f = rng.uniform(0.0, 0.3, 78)
            f[13] = rng.uniform(0.6, 1.0)   # high bytes/s
            f[5]  = rng.uniform(0.5, 1.0)   # large forward packets
        elif label == "Bot":
            f = rng.uniform(0.05, 0.35, 78)
            f[0]  = rng.uniform(0.2, 0.8)   # medium duration (periodic C2)
            f[15] = rng.uniform(0.3, 0.7)   # regular IAT pattern
        elif label in ("FTP-Patator", "SSH-Patator"):
            f = rng.uniform(0.0, 0.4, 78)
            f[0]  = rng.uniform(0.1, 0.6)
            f[43] = 1.0                      # SYN flag
            f[46] = 1.0                      # ACK flag
        elif label in ("DoS Slowloris", "DoS Slowhttptest"):
            f = rng.uniform(0.0, 0.2, 78)
            f[0]  = rng.uniform(0.7, 1.0)   # very long duration (slow attack)
            f[13] = rng.uniform(0.0, 0.1)   # very low bytes/s
        elif label.startswith("Web Attack"):
            f = rng.uniform(0.0, 0.5, 78)
            f[5]  = rng.uniform(0.3, 0.8)   # larger packet sizes (payloads)
            f[46] = 1.0                      # ACK flag
        elif label == "DoS GoldenEye":
            f = rng.uniform(0.0, 0.3, 78)
            f[0]  = rng.uniform(0.05, 0.3)
            f[13] = rng.uniform(0.5, 0.9)
        elif label == "Infiltration":
            f = rng.uniform(0.05, 0.6, 78)  # mimics benign
        else:
            f = rng.uniform(0.0, 0.5, 78)
        rows.append(np.clip(f, 0.0, 1.0))
    return rows


def build_df(rows: list, labels: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df.insert(0, "Label", labels)
    return df


def main():
    rng = np.random.default_rng(42)
    print("=" * 55)
    print("XAI-NIDS  |  Sample CSV Generator")
    print("=" * 55)

    # 1. 10 rows — one per class
    print("\n[1/4] Generating sample_10rows.csv ...")
    rows, labels = [], []
    for cls in CLASSES[:10]:
        rows += gen_class_samples(cls, 1, rng)
        labels.append(cls)
    build_df(rows, labels).to_csv(
        os.path.join(OUT_DIR, "sample_10rows.csv"), index=False, float_format="%.6f"
    )
    print("      Saved -> data/samples/sample_10rows.csv  (10 rows)")

    # 2. 100 rows — balanced, all 14 classes
    print("[2/4] Generating sample_100rows.csv ...")
    rows, labels = [], []
    per_class  = 100 // len(CLASSES)
    remainder  = 100 % len(CLASSES)
    for i, cls in enumerate(CLASSES):
        n = per_class + (1 if i < remainder else 0)
        rows  += gen_class_samples(cls, n, rng)
        labels += [cls] * n
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    build_df([rows[i] for i in idx], [labels[i] for i in idx]).to_csv(
        os.path.join(OUT_DIR, "sample_100rows.csv"), index=False, float_format="%.6f"
    )
    print("      Saved -> data/samples/sample_100rows.csv  (100 rows)")

    # 3. 50 rows — benign only
    print("[3/4] Generating sample_benign_only.csv ...")
    rows = gen_class_samples("BENIGN", 50, rng)
    build_df(rows, ["BENIGN"] * 50).to_csv(
        os.path.join(OUT_DIR, "sample_benign_only.csv"), index=False, float_format="%.6f"
    )
    print("      Saved -> data/samples/sample_benign_only.csv  (50 rows)")

    # 4. 190 rows — attack-heavy demo mix
    print("[4/4] Generating sample_attack_mix.csv ...")
    rows, labels = [], []
    rows  += gen_class_samples("BENIGN", 60, rng)
    labels += ["BENIGN"] * 60
    attack_classes = [c for c in CLASSES if c != "BENIGN"]
    for cls in attack_classes:
        rows  += gen_class_samples(cls, 10, rng)
        labels += [cls] * 10
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    build_df([rows[i] for i in idx], [labels[i] for i in idx]).to_csv(
        os.path.join(OUT_DIR, "sample_attack_mix.csv"), index=False, float_format="%.6f"
    )
    print("      Saved -> data/samples/sample_attack_mix.csv  (190 rows)")

    print("\n" + "=" * 55)
    print("✅  All 4 sample files saved to data/samples/")
    print("=" * 55)
    print("\nQuick-start dashboard demo:")
    print("  cp data/samples/sample_attack_mix.csv data/processed/test.csv")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
