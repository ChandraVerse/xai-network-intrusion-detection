"""
scripts/bootstrap_artifacts.py
--------------------------------
Run **once** from the repo root to train all three models and write every
processed data / artifact file that the dashboard and notebooks expect.

Usage:
    python scripts/bootstrap_artifacts.py              # trains all three models
    python scripts/bootstrap_artifacts.py --skip-lstm  # skip LSTM (no TF needed)

Outputs
-------
data/processed/
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
    scaler.pkl, label_encoder.pkl
    feature_names.json, label_map.json

models/
    random_forest.pkl   rf_metrics.json   feature_importance_rf.json
    xgboost_model.pkl   xgb_metrics.json
    lstm_model.h5       lstm_metrics.json
    metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── import generate_samples helpers from same repo ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from scripts.generate_samples import FEATURE_NAMES, CLASSES, gen_class_samples  # noqa: E402

warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────────
PROC_DIR  = os.path.join(REPO_ROOT, "data", "processed")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics(model_name: str, y_true, y_pred,
             inf_time: float, train_time: float) -> dict:
    acc = round(float(accuracy_score(y_true, y_pred)), 6)
    f1  = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)
    cm  = confusion_matrix(y_true, y_pred)
    fp  = cm.sum(0) - np.diag(cm)
    tn  = cm.sum() - (fp + (cm.sum(1) - np.diag(cm)) + np.diag(cm))
    fpr = round(float(np.mean(fp / (fp + tn + 1e-9))), 6)
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": f1,
        "mean_fpr": fpr,
        "inference_ms_per_flow": round(inf_time, 4),
        "n_test_samples": int(len(y_true)),
        "train_time_s": round(train_time, 2),
    }


# ── Step 1: Generate & persist data ──────────────────────────────────────────

def build_data(n_train: int = 2000, n_test: int = 400) -> tuple:
    print("\n[1/5]  Generating synthetic CICIDS-2017 samples ...")
    rng = np.random.default_rng(42)
    rows_tr, lbl_tr, rows_te, lbl_te = [], [], [], []
    for cls in CLASSES:
        rows_tr += gen_class_samples(cls, n_train, rng);  lbl_tr += [cls] * n_train
        rows_te += gen_class_samples(cls, n_test,  rng);  lbl_te += [cls] * n_test

    df_tr = pd.DataFrame(rows_tr, columns=FEATURE_NAMES)
    df_tr.insert(0, "Label", lbl_tr)
    df_te = pd.DataFrame(rows_te, columns=FEATURE_NAMES)
    df_te.insert(0, "Label", lbl_te)
    df_tr = df_tr.sample(frac=1, random_state=42).reset_index(drop=True)
    df_te = df_te.sample(frac=1, random_state=42).reset_index(drop=True)

    le = LabelEncoder()
    le.fit(CLASSES)
    y_tr = le.transform(df_tr["Label"].values).astype(np.int32)
    y_te = le.transform(df_te["Label"].values).astype(np.int32)
    X_tr = df_tr[FEATURE_NAMES].values.astype(np.float32)
    X_te = df_te[FEATURE_NAMES].values.astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    X_tr_sc = scaler.transform(X_tr).astype(np.float32)
    X_te_sc = scaler.transform(X_te).astype(np.float32)

    print(f"       Train {X_tr_sc.shape}  |  Test {X_te_sc.shape}")

    print("\n[2/5]  Persisting processed artefacts ...")
    np.save(os.path.join(PROC_DIR, "X_train.npy"), X_tr_sc)
    np.save(os.path.join(PROC_DIR, "X_test.npy"),  X_te_sc)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), y_tr)
    np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_te)
    joblib.dump(scaler, os.path.join(PROC_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(PROC_DIR, "label_encoder.pkl"))
    with open(os.path.join(PROC_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(PROC_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print("       Saved to data/processed/")

    return X_tr_sc, X_te_sc, y_tr, y_te, scaler, le


# ── Step 3: Random Forest ─────────────────────────────────────────────────────

def train_rf(X_tr, X_te, y_tr, y_te) -> dict:
    print("\n[3/5]  Training Random Forest (200 trees) ...")
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    tt  = time.time() - t0
    t0  = time.time()
    p   = clf.predict(X_te)
    it  = (time.time() - t0) / len(y_te) * 1000
    m   = _metrics("RandomForest", y_te, p, it, tt)

    joblib.dump(clf, os.path.join(MODEL_DIR, "random_forest.pkl"), compress=3)
    with open(os.path.join(MODEL_DIR, "rf_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    fi = {
        k: round(v, 8)
        for k, v in sorted(
            zip(FEATURE_NAMES, clf.feature_importances_.tolist()),
            key=lambda x: -x[1],
        )
    }
    with open(os.path.join(MODEL_DIR, "feature_importance_rf.json"), "w") as f:
        json.dump(fi, f, indent=2)
    print(f"       Acc={m['accuracy']}  F1={m['macro_f1']}  -> models/random_forest.pkl")
    return m


# ── Step 4: XGBoost ───────────────────────────────────────────────────────────

def train_xgb(X_tr, X_te, y_tr, y_te) -> dict:
    print("\n[4/5]  Training XGBoost (300 trees) ...")
    import xgboost as xgb  # optional dependency
    t0  = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42,
        eval_metric="mlogloss", verbosity=0,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        early_stopping_rounds=20,
        verbose=False,
    )
    tt = time.time() - t0
    t0 = time.time()
    p  = clf.predict(X_te)
    it = (time.time() - t0) / len(y_te) * 1000
    m  = _metrics("XGBoost", y_te, p, it, tt)

    joblib.dump(clf, os.path.join(MODEL_DIR, "xgboost_model.pkl"), compress=3)
    with open(os.path.join(MODEL_DIR, "xgb_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"       Acc={m['accuracy']}  F1={m['macro_f1']}  -> models/xgboost_model.pkl")
    return m


# ── Step 5: LSTM ──────────────────────────────────────────────────────────────

def train_lstm(X_tr, X_te, y_tr, y_te) -> dict:
    print("\n[5/5]  Training LSTM ...")
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical

    N_CLASSES = len(CLASSES)
    T         = 5  # time-steps: repeat feature vector T times
    X_tr_seq  = np.stack([X_tr] * T, axis=1)   # (n, T, 78)
    X_te_seq  = np.stack([X_te] * T, axis=1)
    y_tr_cat  = to_categorical(y_tr, N_CLASSES)
    y_te_cat  = to_categorical(y_te, N_CLASSES)

    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(T, X_tr.shape[1])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    t0 = time.time()
    model.fit(
        X_tr_seq, y_tr_cat,
        batch_size=512, epochs=30,
        validation_split=0.1, verbose=0,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        ],
    )
    tt = time.time() - t0
    t0 = time.time()
    p  = np.argmax(model.predict(X_te_seq, verbose=0), axis=1)
    it = (time.time() - t0) / len(y_te) * 1000
    m  = _metrics("LSTM", y_te, p, it, tt)

    model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    with open(os.path.join(MODEL_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"       Acc={m['accuracy']}  F1={m['macro_f1']}  -> models/lstm_model.h5")
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap XAI-NIDS model artefacts")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (no TensorFlow required)")
    args = parser.parse_args()

    print("=" * 60)
    print("XAI-NIDS  |  Artefact Bootstrap")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te, _, _ = build_data()
    rf_m  = train_rf(X_tr, X_te, y_tr, y_te)

    try:
        xgb_m = train_xgb(X_tr, X_te, y_tr, y_te)
    except ImportError:
        print("      [WARN] xgboost not installed — skipping.")
        xgb_m = {"model": "XGBoost", "accuracy": None, "macro_f1": None}

    if args.skip_lstm:
        print("\n[5/5]  Skipping LSTM (--skip-lstm flag set).")
        lstm_m = {"model": "LSTM", "accuracy": None, "macro_f1": None}
    else:
        try:
            lstm_m = train_lstm(X_tr, X_te, y_tr, y_te)
        except ImportError:
            print("      [WARN] TensorFlow not installed — skipping LSTM.")
            lstm_m = {"model": "LSTM", "accuracy": None, "macro_f1": None}

    summary = {
        "models": [rf_m, xgb_m, lstm_m],
        "dataset": "CICIDS-2017 (synthetic balanced)",
        "n_classes": len(CLASSES),
        "n_features": len(FEATURE_NAMES),
    }
    with open(os.path.join(MODEL_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("✅  All artefacts saved successfully.")
    print("   Run:  git add models/ data/processed/ && git commit -m 'feat: add trained artefacts'")
    print("=" * 60)


if __name__ == "__main__":
    main()
