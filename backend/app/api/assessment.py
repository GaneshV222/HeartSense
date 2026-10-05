from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.schemas import AssessmentRequest
from app.database import get_session
from app.services.risk_service import assess_patient_risk

router = APIRouter()

@router.post("/assessment/predict")
def predict_risk(request: AssessmentRequest):
    try:
        session = get_session()
        
        if request.visitDate:
            try:
                visit_dt = datetime.fromisoformat(request.visitDate.replace('Z', '+00:00'))
                visit_dt = visit_dt.replace(tzinfo=None)
            except ValueError:
                visit_dt = datetime.now()
        else:
            visit_dt = datetime.now()
            
        clinical_dict = request.clinicalData.model_dump() if hasattr(request.clinicalData, 'model_dump') else request.clinicalData.dict()
        
        result = assess_patient_risk(
            session=session,
            patient_code=request.patientCode,
            clinical_values=clinical_dict,
            visit_date=visit_dt,
            patient_name=request.patientName
        )
        session.close()
            
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model artifacts missing: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reports/upload")
def upload_report():
    return {"message": "Uploaded"}
