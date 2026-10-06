from pydantic import BaseModel, Extra
from typing import Optional, Dict, Any

class AssessmentRequest(BaseModel):
    patientCode: Optional[str] = None
    patient_id: Optional[str] = None
    patientName: Optional[str] = None
    patient_name: Optional[str] = None
    visitDate: Optional[str] = None
    visit_date: Optional[str] = None
    clinicalData: Optional[Dict[str, Any]] = None

    class Config:
        extra = Extra.allow
