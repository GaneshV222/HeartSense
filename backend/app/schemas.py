from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class ClinicalData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class AssessmentRequest(BaseModel):
    patientCode: str
    patientName: Optional[str] = None
    visitDate: Optional[str] = None
    clinicalData: ClinicalData
