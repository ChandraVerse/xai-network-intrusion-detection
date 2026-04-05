"""train_artifacts.py
Runs inside GitHub Actions (ubuntu-latest, Python 3.11).
Trains RF, XGBoost, and a tiny LSTM on synthetic CICIDS-2017 data.
All artifacts are kept small enough to commit directly to the repo:
  - RF  .pkl  (joblib compress=9)  ~  200-400 KB
  - XGB .pkl  (joblib compress=9)  ~   50-150 KB
  - LSTM SavedModel.tar.gz         ~  400-800 KB
  - processed data as .npz         ~   50-200 KB total
"""

import json, os, tarfile, time, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/samples", exist_ok=True)
os.makedirs(".github/scripts", exist_ok=True)

print("=" * 60)
print("XAI-NIDS  |  GitHub Actions Artifact Bootstrap")
print(f"TF version : {tf.__version__}")
print(f"XGB version: {xgb.__version__}")
print("=" * 60)

# ────────────────────────────────────────────────────────────────
# Feature schema  (78 CICIDS-2017 features)
# ────────────────────────────────────────────────────────────────
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
N_CLASSES  = len(CLASSES)
N_FEATURES = len(FEATURE_NAMES)

# ────────────────────────────────────────────────────────────────
# Synthetic data generator
# ────────────────────────────────────────────────────────────────
def gen_class(label, n, rng):
    rows = []
    for _ in range(n):
        f = rng.uniform(0.0, 0.3, N_FEATURES)
        if label == "BENIGN":
            f[0]=rng.uniform(0.3,0.9); f[13]=rng.uniform(0.1,0.5); f[46]=float(rng.choice([0,1],p=[0.3,0.7]))
        elif label == "DDoS":
            f[0]=rng.uniform(0.0,0.1); f[13]=rng.uniform(0.7,1.0); f[14]=rng.uniform(0.8,1.0); f[43]=float(rng.choice([0,1]))
        elif label == "PortScan":
            f[0]=rng.uniform(0.0,0.05); f[43]=1.0; f[14]=rng.uniform(0.6,1.0)
        elif label == "DoS Hulk":
            f[13]=rng.uniform(0.6,1.0); f[5]=rng.uniform(0.5,1.0)
        elif label == "Bot":
            f[0]=rng.uniform(0.2,0.8); f[15]=rng.uniform(0.3,0.7)
        elif label in ("FTP-Patator","SSH-Patator"):
            f[43]=1.0; f[46]=1.0; f[0]=rng.uniform(0.1,0.6)
        elif label in ("DoS Slowloris","DoS Slowhttptest"):
            f[0]=rng.uniform(0.7,1.0); f[13]=rng.uniform(0.0,0.1)
        elif label.startswith("Web Attack"):
            f[5]=rng.uniform(0.3,0.8); f[46]=1.0
        elif label == "DoS GoldenEye":
            f[0]=rng.uniform(0.05,0.3); f[13]=rng.uniform(0.5,0.9)
        rows.append(np.clip(f, 0.0, 1.0))
    return rows

def metrics_dict(name, y_true, y_pred, inf_ms, train_s):
    acc = round(float(accuracy_score(y_true, y_pred)), 6)
    f1  = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6)
    cm  = confusion_matrix(y_true, y_pred)
    fp  = cm.sum(0) - np.diag(cm)
    tn  = cm.sum() - (fp + (cm.sum(1) - np.diag(cm)) + np.diag(cm))
    fpr = round(float(np.mean(fp / (fp + tn + 1e-9))), 6)
    return {"model":name,"accuracy":acc,"macro_f1":f1,"mean_fpr":fpr,
            "inference_ms_per_flow":round(inf_ms,4),
            "n_test_samples":int(len(y_true)),"train_time_s":round(train_s,2)}

# ────────────────────────────────────────────────────────────────
# STEP 1 — Generate data  (800 train / 160 test per class)
# Totals: 11 200 train rows, 2 240 test rows, ~6 MB uncompressed
# ────────────────────────────────────────────────────────────────
print("\n[1/5] Generating synthetic data ...")
rng = np.random.default_rng(42)
N_TR, N_TE = 800, 160
rows_tr, lbl_tr, rows_te, lbl_te = [], [], [], []
for cls in CLASSES:
    rows_tr += gen_class(cls, N_TR, rng); lbl_tr += [cls]*N_TR
    rows_te += gen_class(cls, N_TE, rng);  lbl_te += [cls]*N_TE

df_tr = pd.DataFrame(rows_tr, columns=FEATURE_NAMES); df_tr.insert(0,"Label",lbl_tr)
df_te = pd.DataFrame(rows_te, columns=FEATURE_NAMES); df_te.insert(0,"Label",lbl_te)
df_tr = df_tr.sample(frac=1,random_state=42).reset_index(drop=True)
df_te = df_te.sample(frac=1,random_state=42).reset_index(drop=True)

le = LabelEncoder(); le.fit(CLASSES)
y_tr = le.transform(df_tr["Label"].values).astype(np.int32)
y_te = le.transform(df_te["Label"].values).astype(np.int32)
X_tr = df_tr[FEATURE_NAMES].values.astype(np.float32)
X_te = df_te[FEATURE_NAMES].values.astype(np.float32)

scaler = MinMaxScaler(); scaler.fit(X_tr)
X_tr_sc = scaler.transform(X_tr).astype(np.float32)
X_te_sc = scaler.transform(X_te).astype(np.float32)
print(f"      Train {X_tr_sc.shape}  |  Test {X_te_sc.shape}")

# ────────────────────────────────────────────────────────────────
# STEP 2 — Save processed data as compressed .npz  (~10x smaller)
# ────────────────────────────────────────────────────────────────
print("\n[2/5] Saving compressed processed data ...")
np.savez_compressed("data/processed/X_train.npz", data=X_tr_sc)
np.savez_compressed("data/processed/X_test.npz",  data=X_te_sc)
np.save("data/processed/y_train.npy", y_tr)
np.save("data/processed/y_test.npy",  y_te)
joblib.dump(scaler, "data/processed/scaler.pkl",        compress=("zlib",9))
joblib.dump(le,     "data/processed/label_encoder.pkl", compress=("zlib",9))
with open("data/processed/feature_names.json","w") as f: json.dump(FEATURE_NAMES, f, indent=2)
label_map = {str(int(i)): cls for i,cls in enumerate(le.classes_)}
with open("data/processed/label_map.json","w") as f: json.dump(label_map, f, indent=2)
print("      Saved -> data/processed/")

# Sample CSV for dashboard demo (200 rows, stratified)
df_sample = df_te.groupby("Label", group_keys=False).apply(
    lambda g: g.sample(min(15, len(g)), random_state=42)
).sample(frac=1, random_state=42).reset_index(drop=True)
df_sample["label_encoded"] = le.transform(df_sample["Label"].values)
df_sample.to_csv("data/samples/sample_200rows.csv", index=False)
print(f"      Sample CSV saved ({len(df_sample)} rows) -> data/samples/sample_200rows.csv")

# label_map in models/ too (needed by dashboard)
with open("models/label_map.json","w") as f: json.dump(label_map, f, indent=2)

# ────────────────────────────────────────────────────────────────
# STEP 3 — Random Forest  (100 trees, max_features="sqrt")
# Compressed .pkl  ~200-400 KB
# ────────────────────────────────────────────────────────────────
print("\n[3/5] Training Random Forest ...")
t0=time.time()
rf = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_leaf=3,
    max_features="sqrt", n_jobs=-1, random_state=42, class_weight="balanced"
)
rf.fit(X_tr_sc, y_tr)
tt=time.time()-t0
t0=time.time(); p=rf.predict(X_te_sc); it=(time.time()-t0)/len(y_te)*1000
m=metrics_dict("RandomForest",y_te,p,it,tt)
joblib.dump(rf, "models/random_forest.pkl", compress=("zlib",9))
with open("models/rf_metrics.json","w") as f: json.dump(m,f,indent=2)
fi = dict(sorted(zip(FEATURE_NAMES,rf.feature_importances_.tolist()), key=lambda x:-x[1]))
with open("models/feature_importance_rf.json","w") as f: json.dump({k:round(v,8) for k,v in fi.items()},f,indent=2)
size_rf = Path("models/random_forest.pkl").stat().st_size // 1024
print(f"      Acc={m['accuracy']}  F1={m['macro_f1']}  size={size_rf}KB -> models/random_forest.pkl")

# ────────────────────────────────────────────────────────────────
# STEP 4 — XGBoost  (200 trees, early stopping)
# Compressed .pkl  ~50-150 KB
# ────────────────────────────────────────────────────────────────
print("\n[4/5] Training XGBoost ...")
t0=time.time()
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    tree_method="hist", random_state=42, eval_metric="mlogloss", verbosity=0
)
xgb_clf.fit(
    X_tr_sc, y_tr,
    eval_set=[(X_te_sc, y_te)],
    early_stopping_rounds=15, verbose=False
)
tt=time.time()-t0
t0=time.time(); p=xgb_clf.predict(X_te_sc); it=(time.time()-t0)/len(y_te)*1000
m=metrics_dict("XGBoost",y_te,p,it,tt)
joblib.dump(xgb_clf, "models/xgboost_model.pkl", compress=("zlib",9))
with open("models/xgb_metrics.json","w") as f: json.dump(m,f,indent=2)
size_xgb = Path("models/xgboost_model.pkl").stat().st_size // 1024
print(f"      Acc={m['accuracy']}  F1={m['macro_f1']}  size={size_xgb}KB -> models/xgboost_model.pkl")

# ────────────────────────────────────────────────────────────────
# STEP 5 — LSTM  (tiny architecture, saved as .tar.gz)
# SavedModel tar.gz  ~400-800 KB
# ────────────────────────────────────────────────────────────────
print("\n[5/5] Training LSTM ...")
TIME_STEPS = 5
X_tr_lstm = np.stack([X_tr_sc]*TIME_STEPS, axis=1)  # (n, 5, 78)
X_te_lstm = np.stack([X_te_sc]*TIME_STEPS, axis=1)
y_tr_cat  = to_categorical(y_tr, N_CLASSES)
y_te_cat  = to_categorical(y_te, N_CLASSES)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, N_FEATURES)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(N_CLASSES, activation="softmax"),
], name="lstm_nids")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy", metrics=["accuracy"]
)
t0=time.time()
model.fit(
    X_tr_lstm, y_tr_cat,
    batch_size=256, epochs=20, validation_split=0.1, verbose=0,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)]
)
tt=time.time()-t0
t0=time.time(); p=np.argmax(model.predict(X_te_lstm,verbose=0),axis=1); it=(time.time()-t0)/len(y_te)*1000
m=metrics_dict("LSTM",y_te,p,it,tt)

# Save as SavedModel -> tar.gz for compact git storage
save_dir = "/tmp/lstm_saved_model"
model.save(save_dir)
with tarfile.open("models/lstm_model.tar.gz", "w:gz") as tar:
    tar.add(save_dir, arcname="lstm_saved_model")
with open("models/lstm_metrics.json","w") as f: json.dump(m,f,indent=2)
size_lstm = Path("models/lstm_model.tar.gz").stat().st_size // 1024
print(f"      Acc={m['accuracy']}  F1={m['macro_f1']}  size={size_lstm}KB -> models/lstm_model.tar.gz")

# ────────────────────────────────────────────────────────────────
# metrics_summary.json
# ────────────────────────────────────────────────────────────────
rf_m  = json.load(open("models/rf_metrics.json"))
xgb_m = json.load(open("models/xgb_metrics.json"))
lstm_m= json.load(open("models/lstm_metrics.json"))
summary = {
    "models": [rf_m, xgb_m, lstm_m],
    "dataset": "CICIDS-2017 (synthetic balanced, GitHub Actions generated)",
    "n_classes": N_CLASSES,
    "n_features": N_FEATURES,
    "n_train_samples": int(len(y_tr)),
    "n_test_samples":  int(len(y_te)),
    "generated_by": "train_artifacts.py via GitHub Actions",
}
with open("models/metrics_summary.json","w") as f: json.dump(summary,f,indent=2)

print("\n" + "="*60)
print("All artifacts saved!")
print(f"  RF   : {Path('models/random_forest.pkl').stat().st_size//1024} KB")
print(f"  XGB  : {Path('models/xgboost_model.pkl').stat().st_size//1024} KB")
print(f"  LSTM : {Path('models/lstm_model.tar.gz').stat().st_size//1024} KB")
print(f"  X_train.npz : {Path('data/processed/X_train.npz').stat().st_size//1024} KB")
print(f"  X_test.npz  : {Path('data/processed/X_test.npz').stat().st_size//1024} KB")
print("="*60)
