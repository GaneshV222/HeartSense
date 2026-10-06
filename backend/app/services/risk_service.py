"""
HeartSense – Risk Service

Orchestrates dynamic / temporal risk assessment by combining
the patient service, prediction service, and temporal feature extraction.
"""

from datetime import datetime
from sqlalchemy.orm import Session

from app.services.patient_service import (
    create_patient,
    create_visit,
    create_prediction,
    get_patient_by_id,
    get_patient_visits,
    get_latest_visit,
    get_previous_visit,
)
from app.services.prediction_service import predict_single
from app.ml.temporal import compute_temporal_changes, summarize_risk_changes
from app.config import BASELINE_FEATURE_COLS


def assess_patient_risk(
    session: Session,
    patient_id: str,
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
    pid = str(patient_id)

    # 1. Patient
    patient = create_patient(
        session,
        patient_id=pid,
        name=patient_name,
        gender=clinical_values.get("gender"),
        age=clinical_values.get("age"),
    )

    # 2. Previous visit
    prev_visit = get_latest_visit(session, pid)

    # 3. New visit
    visit = create_visit(session, pid, visit_date, clinical_values)

    # 4. Prediction
    pred_result = predict_single(clinical_values, patient_id=pid)

    # 5. Store prediction
    create_prediction(
        session,
        patient_id=pid,
        visit_id=visit.id,
        visit_date=visit_date,
        prediction=pred_result["risk_prediction"],
        probability=pred_result["risk_probability"],
        risk_level=pred_result["risk_category"],
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


def get_patient_timeline(session: Session, patient_id: str) -> list[dict]:
    """
    Build a chronological timeline of visits + predictions for a patient.
    """
    patient = get_patient_by_id(session, str(patient_id))
    if not patient:
        return []

    visits = get_patient_visits(session, str(patient_id))
    timeline = []
    for v in visits:
        pred_rec = session.query(Prediction).filter_by(visit_id=v.id).first()
        entry = {
            "visit_id": v.id,
            "timestamp": v.visit_timestamp,
            "visit_date": v.visit_date.isoformat() if v.visit_date else None,
            "features": v.to_feature_dict(),
        }
        if pred_rec:
            entry["prediction"] = pred_rec.prediction
            entry["probability"] = pred_rec.probability
            entry["model_name"] = pred_rec.model_name
            entry["risk_level"] = pred_rec.risk_level
        else:
            entry["prediction"] = None
            entry["probability"] = None
            entry["model_name"] = None
            entry["risk_level"] = None
        timeline.append(entry)
    return timeline
