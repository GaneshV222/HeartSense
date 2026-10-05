"""
HeartSense – Risk Service

Orchestrates the dynamic / temporal risk assessment by combining
the patient service, prediction service, and temporal feature extraction.
"""

from datetime import datetime
from sqlalchemy.orm import Session

from app.services.patient_service import (
    create_patient,
    create_visit,
    create_prediction,
    get_patient_by_code,
    get_patient_visits,
    get_latest_visit,
    get_previous_visit,
)
from app.services.prediction_service import predict_single
from app.ml.temporal import compute_temporal_changes, summarize_risk_changes
from app.config import FEATURE_NAMES


def assess_patient_risk(
    session: Session,
    patient_code: str,
    clinical_values: dict,
    visit_date: datetime | None = None,
    patient_name: str | None = None,
) -> dict:
    """
    Full dynamic risk assessment workflow:

    1. Create / retrieve patient
    2. Find previous visit (if any)
    3. Create new visit record
    4. Generate ML prediction
    5. Store prediction
    6. Compute temporal changes (if previous visit exists)
    7. Return comprehensive assessment result
    """
    visit_date = visit_date or datetime.utcnow()

    # 1. Patient
    patient = create_patient(session, patient_code, name=patient_name)

    # 2. Previous visit
    prev_visit = get_latest_visit(session, patient.id)

    # 3. New visit
    visit = create_visit(session, patient.id, visit_date, clinical_values)

    # 4. Prediction
    pred_result = predict_single(clinical_values)

    # 5. Store prediction
    create_prediction(
        session,
        patient_id=patient.id,
        visit_id=visit.id,
        prediction=pred_result["prediction"],
        probability=pred_result["probability"],
        model_name=pred_result["model_name"],
    )

    # 6. Temporal comparison
    temporal_changes = None
    risk_summary = None
    if prev_visit:
        temporal_changes = compute_temporal_changes(
            current_values=clinical_values,
            previous_values=prev_visit.to_feature_dict(),
        )
        risk_summary = summarize_risk_changes(temporal_changes)

    # 7. Assemble result
    return {
        "patient": patient,
        "visit": visit,
        "prediction": pred_result,
        "previous_visit": prev_visit,
        "temporal_changes": temporal_changes,
        "risk_summary": risk_summary,
        "is_first_visit": prev_visit is None,
    }


def get_patient_timeline(session: Session, patient_code: str) -> list[dict]:
    """
    Build a chronological timeline of visits + predictions for a patient.
    Returns a list of dicts suitable for display in Streamlit.
    """
    patient = get_patient_by_code(session, patient_code)
    if not patient:
        return []

    visits = get_patient_visits(session, patient.id)
    timeline = []
    for v in visits:
        entry = {
            "visit_id": v.id,
            "timestamp": v.visit_timestamp,
            "features": v.to_feature_dict(),
        }
        if v.prediction:
            entry["prediction"] = v.prediction.prediction
            entry["probability"] = v.prediction.probability
            entry["model_name"] = v.prediction.model_name
        else:
            entry["prediction"] = None
            entry["probability"] = None
            entry["model_name"] = None
        timeline.append(entry)
    return timeline
