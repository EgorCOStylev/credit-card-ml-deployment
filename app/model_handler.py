"""
Model loading and inference utilities.
"""

import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

_cache = {}


def load_model(version: str = "v1"):
    """Load model by version with caching."""
    if version not in _cache:
        path = os.path.join(MODELS_DIR, f"model_{version}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        _cache[version] = joblib.load(path)
    return _cache[version]


def predict(features: dict, version: str = "v1") -> dict:
    """
    Run inference.

    Args:
        features: dict mapping feature name -> value
        version: model version ('v1' or 'v2')

    Returns:
        dict with prediction, probability, model_version
    """
    artifact = load_model(version)
    pipeline = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    # Build input array in correct column order
    try:
        X = np.array([[features[col] for col in feature_cols]], dtype=float)
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}")

    pred = int(pipeline.predict(X)[0])
    prob = float(pipeline.predict_proba(X)[0][1])

    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "model_version": version,
    }


def get_feature_names(version: str = "v1") -> list:
    artifact = load_model(version)
    return artifact["feature_cols"]
