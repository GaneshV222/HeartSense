from fastapi import APIRouter, HTTPException
from app.database import get_session
from app.services.patient_service import get_all_patients, get_patient_by_code
from app.services.risk_service import get_patient_timeline

router = APIRouter()

@router.get("/patients")
def list_patients():
    session = get_session()
    patients = get_all_patients(session)
    session.close()
    return [{"patient_code": p.patient_code, "name": p.name} for p in patients]

@router.get("/history/{patient_code}")
def get_history(patient_code: str):
    session = get_session()
    patient = get_patient_by_code(session, patient_code)
    if not patient:
        session.close()
        raise HTTPException(status_code=404, detail="Patient not found")
        
    timeline = get_patient_timeline(session, patient_code)
    session.close()
    
    for entry in timeline:
        if "timestamp" in entry and entry["timestamp"]:
            entry["timestamp"] = entry["timestamp"].isoformat()
            
    return {
        "patient": {"patient_code": patient.patient_code, "name": patient.name},
        "timeline": timeline
    }
