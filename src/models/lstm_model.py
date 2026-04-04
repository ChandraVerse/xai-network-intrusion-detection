"""
lstm_model.py
-------------
LSTM (Long Short-Term Memory) model for temporal network flow classification.

Usage:
    python src/models/lstm_model.py --data data/processed/train_balanced.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

MODEL_PATH = "models/lstm_model.h5"
TIME_STEPS  = 5


def reshape_for_lstm(X: np.ndarray, time_steps: int = TIME_STEPS) -> np.ndarray:
    """
    Reshape 2D feature matrix (samples, features) into
    3D tensor (samples, time_steps, features) required by LSTM.

    Truncates the last (samples % time_steps) rows to ensure even slicing.
    """
    n_samples, n_features = X.shape
    n_trimmed = (n_samples // time_steps) * time_steps
    X_trimmed = X[:n_trimmed]
    return X_trimmed.reshape(-1, time_steps, n_features)


def build_lstm(time_steps: int, n_features: int, n_classes: int):
    """
    Build and compile a stacked LSTM architecture.

    Architecture:
        LSTM(128, return_sequences=True)
        Dropout(0.3)
        LSTM(64)
        Dropout(0.3)
        Dense(32, relu)
        Dense(n_classes, softmax)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(128, input_shape=(time_steps, n_features), return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    time_steps: int = TIME_STEPS,
    epochs: int = 10,
    batch_size: int = 512,
    save_path: str = MODEL_PATH,
):
    """
    Reshape data, build LSTM, train, and save model.

    Parameters
    ----------
    X_train    : scaled, SMOTE-balanced training features (2D)
    y_train    : integer label vector
    time_steps : sequence length for temporal reshaping
    epochs     : training epochs
    batch_size : mini-batch size
    save_path  : .h5 path to save trained Keras model

    Returns the trained Keras model.
    """
    n_classes  = len(np.unique(y_train))
    n_features = X_train.shape[1]

    X_3d = reshape_for_lstm(X_train, time_steps)
    # Trim labels to match reshaped samples
    y_trimmed = y_train[:X_3d.shape[0] * time_steps:time_steps]

    print(f"  [LSTM] X_3d shape: {X_3d.shape}  |  n_classes: {n_classes}")

    model = build_lstm(time_steps, n_features, n_classes)
    model.fit(
        X_3d, y_trimmed,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"  [LSTM] Model saved to {save_path}")

    return model


def load_lstm_model(path: str = MODEL_PATH):
    """Load a saved Keras LSTM model from disk."""
    from tensorflow.keras.models import load_model
    return load_model(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM on CICIDS-2017")
    parser.add_argument("--data",       required=True, help="Path to train_balanced.csv")
    parser.add_argument("--label",      default="Label")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X  = df.drop(columns=[args.label]).values.astype(np.float32)
    y  = df[args.label].values.astype(int)

    train_lstm(X, y, epochs=args.epochs, batch_size=args.batch_size)
    print(f"  [LSTM] Training complete. Model saved to {MODEL_PATH}")
