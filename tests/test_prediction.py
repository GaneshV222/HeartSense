"""
Tests for the prediction service.
"""

import pytest
import os
from src.config import MODEL_ARTIFACT_DIR


@pytest.fixture
def has_artifacts():
    """Check if model artifacts exist (training must be run first)."""
    required = ["best_model.joblib", "selected_features.json"]
    for f in required:
        if not os.path.exists(os.path.join(MODEL_ARTIFACT_DIR, f)):
            pytest.skip(f"Artifact {f} not found. Run `python train.py` first.")
    return True


class TestPredictionService:
    """Tests for the prediction service."""

    def test_predict_single_low_risk(self, has_artifacts):
        from src.services.prediction_service import predict_single

        # Low-risk patient profile
        patient = {
            "age": 35, "sex": 0, "cp": 0, "trestbps": 110,
            "chol": 180, "fbs": 0, "restecg": 0, "thalach": 175,
            "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0,
        }
        result = predict_single(patient)

        assert "prediction" in result
        assert "probability" in result
        assert "model_name" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_predict_single_high_risk(self, has_artifacts):
        from src.services.prediction_service import predict_single

        # High-risk patient profile
        patient = {
            "age": 65, "sex": 1, "cp": 3, "trestbps": 180,
            "chol": 350, "fbs": 1, "restecg": 2, "thalach": 100,
            "exang": 1, "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 2,
        }
        result = predict_single(patient)

        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_predict_returns_model_name(self, has_artifacts):
        from src.services.prediction_service import predict_single

        patient = {
            "age": 50, "sex": 1, "cp": 1, "trestbps": 130,
            "chol": 220, "fbs": 0, "restecg": 1, "thalach": 140,
            "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 1,
        }
        result = predict_single(patient)
        assert result["model_name"] != "Unknown"
