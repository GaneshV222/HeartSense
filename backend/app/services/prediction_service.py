"""
HeartSense – Prediction Service

Loads the saved best model and selected features from artifacts,
preprocesses a single patient's input, and returns the prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from app.config import MODEL_ARTIFACT_DIR
from app.ml.feature_selection import load_selected_features
from app.ml.preprocessing import preprocess_single_patient


def _load_model(artifact_dir: str = MODEL_ARTIFACT_DIR):
    """Load the best trained model from disk."""
    path = os.path.join(artifact_dir, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best model not found at {path}. Run train.py first."
        )
    return joblib.load(path)


def predict_single(patient_data: dict, artifact_dir: str = MODEL_ARTIFACT_DIR) -> dict:
    """
    Run the full prediction pipeline for one patient.

    Parameters
    ----------
    patient_data : dict  – raw feature values keyed by feature name
    artifact_dir : str   – path to the directory containing model artifacts

    Returns
    -------
    dict with keys: prediction, probability, model_name
    """
    model = _load_model(artifact_dir)
    selected_features = load_selected_features(artifact_dir)
    X = preprocess_single_patient(patient_data, selected_features)

    pred = int(model.predict(X)[0])

    # Probability (not all models support predict_proba)
    try:
        proba = float(model.predict_proba(X)[0][1])
    except AttributeError:
        try:
            dec = float(model.decision_function(X)[0])
            # Sigmoid approximation for SVM decision values
            proba = float(1 / (1 + np.exp(-dec)))
        except AttributeError:
            proba = float(pred)

    # Determine model name from metrics artifact
    model_name = _get_model_name(artifact_dir)

    return {
        "prediction": pred,
        "probability": round(proba, 4),
        "model_name": model_name,
        "label": "High Risk – Disease Predicted" if pred == 1 else "Low Risk – No Disease Predicted",
    }


def _get_model_name(artifact_dir: str = MODEL_ARTIFACT_DIR) -> str:
    """Read the best model name from saved metrics."""
    import json
    path = os.path.join(artifact_dir, "model_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            metrics = json.load(f)
        return metrics.get("model_name", "Unknown")
    return "Unknown"
