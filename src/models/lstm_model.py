"""LSTM classifier for network intrusion detection.

Reshapes flat 78-feature flow vectors into 3D sliding-window sequences
(samples, time_steps=5, features=78) and trains a two-layer LSTM with
dropout regularisation.

Usage (CLI):
    python src/models/lstm_model.py \\
        --data data/processed/train_balanced.csv \\
        --test data/processed/test.csv \\
        --out  models/

Outputs:
    models/lstm_model.h5       Keras model weights (HDF5)
    models/lstm_metrics.json   Accuracy, Macro F1, FPR, inference time
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required: pip install tensorflow>=2.12"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TIME_STEPS = 5       # sliding window length
BATCH_SIZE = 512
EPOCHS     = 30
LR         = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int = TIME_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flat 2D array to sliding-window 3D sequences.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label vector of shape (n_samples,).
        time_steps: Number of consecutive flows per sequence window.

    Returns:
        Xs: shape (n_samples - time_steps + 1, time_steps, n_features)
        ys: shape (n_samples - time_steps + 1,)  -- label of the LAST flow
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys)


def build_model(time_steps: int, n_features: int, n_classes: int) -> Sequential:
    """Construct the two-layer LSTM architecture."""
    model = Sequential(
        [
            LSTM(
                128,
                input_shape=(time_steps, n_features),
                return_sequences=True,
                name="lstm_1",
            ),
            Dropout(0.3, name="dropout_1"),
            LSTM(64, return_sequences=False, name="lstm_2"),
            Dropout(0.3, name="dropout_2"),
            Dense(32, activation="relu", name="dense_1"),
            Dense(n_classes, activation="softmax", name="output"),
        ],
        name="xai_nids_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)
    return model


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_split(
    train_path: str | Path,
    test_path: str | Path,
    label_col: str = "label_encoded",
) -> tuple:
    log.info("Loading training data from %s", train_path)
    train = pd.read_csv(train_path)
    log.info("Loading test data from %s", test_path)
    test = pd.read_csv(test_path)

    feature_cols = [c for c in train.columns if c != label_col]
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[label_col].values.astype(int)
    X_test  = test[feature_cols].values.astype(np.float32)
    y_test  = test[label_col].values.astype(int)

    n_classes = len(np.unique(y_train))
    log.info(
        "Train: %s  |  Test: %s  |  Features: %d  |  Classes: %d",
        X_train.shape, X_test.shape, X_train.shape[1], n_classes,
    )
    return X_train, y_train, X_test, y_test, n_classes


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> tuple[Sequential, object]:
    """Create sequences, build model, and fit with early stopping."""
    log.info("Creating sliding-window sequences (time_steps=%d)", TIME_STEPS)
    X_seq, y_seq = create_sequences(X_train, y_train, TIME_STEPS)
    y_cat = to_categorical(y_seq, num_classes=n_classes)

    n_features = X_seq.shape[2]
    model = build_model(TIME_STEPS, n_features, n_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    log.info("Fitting LSTM  epochs=%d  batch_size=%d", EPOCHS, BATCH_SIZE)
    t0 = time.perf_counter()
    history = model.fit(
        X_seq, y_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f s", elapsed)
    return model, history


def evaluate(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> dict:
    log.info("Creating test sequences")
    X_seq, y_seq = create_sequences(X_test, y_test, TIME_STEPS)

    t0 = time.perf_counter()
    y_prob = model.predict(X_seq, batch_size=BATCH_SIZE, verbose=0)
    inference_ms = (time.perf_counter() - t0) / len(X_seq) * 1000

    y_pred = np.argmax(y_prob, axis=1)

    acc      = accuracy_score(y_seq, y_pred)
    macro_f1 = f1_score(y_seq, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_seq, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr_per_class = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
    mean_fpr = float(np.mean(fpr_per_class))

    metrics = {
        "model": "LSTM",
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "false_positive_rate": round(mean_fpr, 6),
        "inference_ms_per_flow": round(inference_ms, 4),
        "time_steps": TIME_STEPS,
        "n_test_samples": int(len(y_seq)),
    }

    log.info("Accuracy : %.4f", acc)
    log.info("Macro F1 : %.4f", macro_f1)
    log.info("Mean FPR : %.4f", mean_fpr)
    log.info("Inference: %.4f ms/flow", inference_ms)
    log.info("\n%s", classification_report(y_seq, y_pred, zero_division=0))

    return metrics


def save(
    model: Sequential,
    metrics: dict,
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "lstm_model.h5"
    model.save(str(model_path))
    log.info("Model saved → %s", model_path)

    metrics_path = out / "lstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM for XAI-NIDS")
    p.add_argument("--data",  default="data/processed/train_balanced.csv")
    p.add_argument("--test",  default="data/processed/test.csv")
    p.add_argument("--out",   default="models/")
    p.add_argument("--label", default="label_encoded")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X_train, y_train, X_test, y_test, n_classes = load_split(
        args.data, args.test, label_col=args.label
    )
    model, _ = train(X_train, y_train, n_classes)
    metrics  = evaluate(model, X_test, y_test, n_classes)
    save(model, metrics, args.out)
    log.info("Done.")


if __name__ == "__main__":
    main()
