"""
random_forest.py
----------------
Random Forest classifier training, evaluation, and serialisation.

Usage:
    python src/models/random_forest.py --data data/processed/train_balanced.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

MODEL_PATH = "models/random_forest.pkl"


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
    save_path: str = MODEL_PATH,
) -> RandomForestClassifier:
    """
    Train and serialise a Random Forest classifier.

    Parameters
    ----------
    X_train      : scaled, SMOTE-balanced training features
    y_train      : integer label vector
    n_estimators : number of trees
    random_state : reproducibility seed
    save_path    : where to save the fitted model (.pkl)

    Returns the fitted RandomForestClassifier.
    """
    print(f"  [RF] Training RandomForest (n_estimators={n_estimators}) ...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(clf, save_path)
        print(f"  [RF] Model saved to {save_path}")

    return clf


def evaluate(clf, X_test: np.ndarray, y_test: np.ndarray, label_names=None):
    """Print classification report and return macro F1."""
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names))
    return f1_score(y_test, y_pred, average="macro")


def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    """Load a serialised Random Forest from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest on CICIDS-2017")
    parser.add_argument("--data",  required=True, help="Path to train_balanced.csv")
    parser.add_argument("--label", default="Label",  help="Target column name")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X  = df.drop(columns=[args.label]).values
    y  = df[args.label].values

    clf = train_random_forest(X, y, n_estimators=args.n_estimators)
    print(f"  [RF] Training complete. Model saved to {MODEL_PATH}")
