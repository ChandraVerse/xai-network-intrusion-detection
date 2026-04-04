"""
build_processed.py
------------------
Master pipeline script: clean → encode → split → scale → SMOTE → save.

Run from the repository root:
    python scripts/build_processed.py

Outputs written to data/processed/:
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
    feature_names.json, label_encoder.pkl, scaler.pkl

All random operations use random_state=42 for full reproducibility.
Published metrics: Accuracy=99.94%  Macro-F1=0.997  (Random Forest, 20% holdout)
"""

import os
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Local imports (run from repo root so these resolve)
from src.preprocessing.cleaner import load_and_clean_cicids
from src.preprocessing.scaler import fit_scaler, apply_scaler
from src.preprocessing.smote_balancer import apply_smote

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR = "data/raw/MachineLearningCVE"
OUT_DIR = "data/processed"
LABEL_COL = "Label"
TEST_SIZE = 0.20
RANDOM_STATE = 42
SMOTE_STRATEGY = "not majority"
# ──────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("XAI-NIDS  |  Data Processing Pipeline")
    print("=" * 60)

    # ── Step 1: Clean ──────────────────────────────────────────────
    print("\n[1/6] Cleaning raw CSVs ...")
    df = load_and_clean_cicids(RAW_DIR)
    print(f"      Cleaned shape: {df.shape}")

    # ── Step 2: Encode labels ──────────────────────────────────────
    print("\n[2/6] Encoding labels ...")
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)
    print(f"      Classes ({len(le.classes_)}): {list(le.classes_)}")
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
    print("      Saved → data/processed/label_encoder.pkl")

    # ── Step 3: Save feature names ─────────────────────────────────
    print("\n[3/6] Saving feature names ...")
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as fh:
        json.dump(feature_cols, fh, indent=2)
    print(f"      {len(feature_cols)} features → data/processed/feature_names.json")

    X = df[feature_cols].values.astype(np.float32)

    # ── Step 4: Train / test split ─────────────────────────────────
    print("\n[4/6] Splitting (80/20 stratified, random_state=42) ...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"      Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

    # ── Step 5: Scale (fit on train only — no leakage) ─────────────
    print("\n[5/6] Fitting MinMaxScaler on training set ...")
    scaler = fit_scaler(
        X_train_raw,
        save_path=os.path.join(OUT_DIR, "scaler.pkl"),
    )
    X_train_scaled = apply_scaler(X_train_raw, scaler)
    X_test_scaled = apply_scaler(X_test_raw, scaler)
    print("      Saved → data/processed/scaler.pkl")
    print(f"      Scaled range — min: {X_train_scaled.min():.4f}  "
          f"max: {X_train_scaled.max():.4f}")

    # ── Step 6: SMOTE (training set only) ─────────────────────────
    print("\n[6/6] Applying SMOTE to training set ...")
    X_train_final, y_train_final = apply_smote(
        X_train_scaled, y_train,
        strategy=SMOTE_STRATEGY,
        random_state=RANDOM_STATE,
    )

    # ── Save final arrays ─────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train_final.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test_scaled.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train_final.astype(np.int32))
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test.astype(np.int32))

    print("\n" + "=" * 60)
    print("✅  All artifacts saved to data/processed/")
    print("=" * 60)
    print(f"   X_train : {X_train_final.shape}  (SMOTE-balanced)")
    print(f"   X_test  : {X_test_scaled.shape}")
    print(f"   y_train : {y_train_final.shape}")
    print(f"   y_test  : {y_test.shape}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes : {len(le.classes_)}")
    print("\nNext step: python scripts/train_models.py")


if __name__ == "__main__":
    main()
