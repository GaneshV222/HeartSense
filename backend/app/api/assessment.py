"""
HeartSense – Patient Assessment API
"""

import re
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.schemas import AssessmentRequest
from app.ml.temporal_pipeline import (
    insert_manual_visit,
    predict_latest_for_patient,
)
from app.services.temporal_service import TemporalFeatureService
from app.database import get_session

router = APIRouter()


def normalize_patient_id(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^patient\s*id\s*[:#-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^id\s*[:#-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or None


@router.post("/assessment/predict")
def predict_risk(request: AssessmentRequest):
    try:
        pid = normalize_patient_id(request.patientCode) or normalize_patient_id(request.patient_id)
        raw_dict = request.dict(exclude_unset=True) if hasattr(request, "dict") else {}
        if not pid:
            pid = normalize_patient_id(raw_dict.get("patient_code")) or normalize_patient_id(raw_dict.get("patient_id"))
        if not pid:
            raise ValueError("patient_id or patientCode is required.")

        clinical = request.clinicalData or {}
        if not clinical:
            reserved = {"patientCode", "patient_id", "patientName", "patient_name", "visitDate", "visit_date", "clinicalData"}
            clinical = {k: v for k, v in raw_dict.items() if k not in reserved and v is not None}

        v_date_str = request.visitDate or request.visit_date or (clinical.get("visit_date") if isinstance(clinical, dict) else None)
        visit_dt = None
        if v_date_str:
            try:
                visit_dt = datetime.fromisoformat(str(v_date_str).replace('Z', '+00:00')).replace(tzinfo=None)
            except Exception:
                visit_dt = None

        if clinical:
            insert_manual_visit(str(pid), clinical, visit_dt)

        result = predict_latest_for_patient(str(pid))
        pname = request.patientName or request.patient_name or f"Patient {pid}"

        return {
            **result,
            "patient_code": str(pid),
            "patient_name": pname,
            "prediction": {
                "prediction": result["risk_prediction"],
                "probability": result["risk_probability"],
                "model_name": result.get("model_name", "HeartSense ML/DL Model"),
                "label": "Higher Cardiovascular Risk" if result["risk_prediction"] == 1 else "Lower Cardiovascular Risk",
            },
            "is_first_visit": result["is_first_visit"],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/patients/{patient_id}/temporal")
def get_patient_temporal_profile(patient_id: str):
    try:
        cleaned_id = normalize_patient_id(patient_id)
        if not cleaned_id:
            raise ValueError("Invalid patient ID")
        return predict_latest_for_patient(cleaned_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/patients/{patient_id}/visits")
def get_patient_visits_api(patient_id: str):
    cleaned_id = normalize_patient_id(patient_id)
    if not cleaned_id:
        raise HTTPException(status_code=400, detail="Invalid patient ID")

    session = get_session()
    try:
        timeline = TemporalFeatureService.get_patient_temporal_timeline(session, cleaned_id)
        if not timeline:
            raise HTTPException(status_code=404, detail=f"No visits found for patient '{cleaned_id}'")
        return {
            "patient_id": cleaned_id,
            "total_visits": len(timeline),
            "timeline": timeline,
        }
    finally:
        session.close()
