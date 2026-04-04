"""
xgboost_model.py
----------------
XGBoost classifier training, evaluation, and serialisation.

Usage:
    python src/models/xgboost_model.py --data data/processed/train_balanced.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

MODEL_PATH = "models/xgboost_model.pkl"


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    random_state: int = 42,
    save_path: str = MODEL_PATH,
) -> XGBClassifier:
    """
    Train and serialise an XGBoost multiclass classifier.

    Parameters
    ----------
    X_train       : scaled, SMOTE-balanced training features
    y_train       : integer label vector (0-indexed)
    n_estimators  : number of boosting rounds
    max_depth     : maximum tree depth
    learning_rate : step size shrinkage
    random_state  : reproducibility seed
    save_path     : where to save the fitted model (.pkl)

    Returns the fitted XGBClassifier.
    """
    n_classes = len(np.unique(y_train))
    print(f"  [XGB] Training XGBoost (n_estimators={n_estimators}, "
          f"max_depth={max_depth}, n_classes={n_classes}) ...")

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )
    clf.fit(X_train, y_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(clf, save_path)
        print(f"  [XGB] Model saved to {save_path}")

    return clf


def evaluate(clf, X_test: np.ndarray, y_test: np.ndarray, label_names=None):
    """Print classification report and return macro F1."""
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names))
    return f1_score(y_test, y_pred, average="macro")


def load_model(path: str = MODEL_PATH):
    """Load a serialised XGBoost model from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on CICIDS-2017")
    parser.add_argument("--data",  required=True, help="Path to train_balanced.csv")
    parser.add_argument("--label", default="Label",  help="Target column name")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth",    type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X  = df.drop(columns=[args.label]).values
    y  = df[args.label].values

    clf = train_xgboost(X, y,
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth)
    print(f"  [XGB] Training complete. Model saved to {MODEL_PATH}")
