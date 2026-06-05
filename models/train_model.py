"""
Train credit card default prediction model.
Saves two versions: model_v1.pkl (LogisticRegression) and model_v2.pkl (RandomForest).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Download and prepare dataset."""
    print("Loading dataset...")
    df = pd.read_excel(DATA_URL, header=1, index_col=0)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"default payment next month": "default"})

    feature_cols = [c for c in df.columns if c != "default"]
    X = df[feature_cols].values
    y = df["default"].values
    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y, feature_cols


def train_and_save():
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- v1: Logistic Regression ---
    pipe_v1 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    pipe_v1.fit(X_train, y_train)
    y_pred_v1 = pipe_v1.predict(X_test)
    f1_v1 = f1_score(y_test, y_pred_v1)
    print(f"\n=== Model v1 (LogisticRegression) ===")
    print(classification_report(y_test, y_pred_v1))
    print(f"F1 (default class): {f1_v1:.4f}")

    # --- v2: Random Forest ---
    pipe_v2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])
    pipe_v2.fit(X_train, y_train)
    y_pred_v2 = pipe_v2.predict(X_test)
    f1_v2 = f1_score(y_test, y_pred_v2)
    print(f"\n=== Model v2 (RandomForest) ===")
    print(classification_report(y_test, y_pred_v2))
    print(f"F1 (default class): {f1_v2:.4f}")

    # Save models
    artifact = {
        "pipeline": pipe_v1,
        "feature_cols": feature_cols,
        "metrics": {"f1": f1_v1},
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "model_v1.pkl"))

    artifact2 = {
        "pipeline": pipe_v2,
        "feature_cols": feature_cols,
        "metrics": {"f1": f1_v2},
    }
    joblib.dump(artifact2, os.path.join(MODELS_DIR, "model_v2.pkl"))

    print("\nModels saved: model_v1.pkl, model_v2.pkl")


if __name__ == "__main__":
    train_and_save()
