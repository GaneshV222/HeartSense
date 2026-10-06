"""
HeartSense – Prediction Service

Loads the saved best model and temporal feature vector from artifacts,
computes dynamic cardiovascular risk, and returns the comprehensive prediction.
"""

from app.ml.temporal_pipeline import (
    predict_latest_for_patient,
    insert_manual_visit,
    load_visits_from_db,
    load_mapping,
    generate_temporal_features,
)


def predict_single(patient_data: dict, patient_id: str = "P001") -> dict:
    """
    Run the temporal prediction pipeline for one patient visit.
    """
    return insert_manual_visit(patient_id, patient_data)


def get_patient_temporal_prediction(patient_id: str) -> dict:
    """
    Run temporal prediction for an existing patient based on all their recorded visits in PostgreSQL.
    """
    return predict_latest_for_patient(patient_id)
