"""
HeartSense – Pydantic API Schemas
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Extra


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


class CurrentVisit(BaseModel):
    patient_id: str
    visit_id: Optional[int] = None
    visit_date: Optional[str] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    bmi: Optional[float] = None
    chest_pain_type: Optional[str] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    max_heart_rate: Optional[float] = None
    cholesterol: Optional[float] = None
    hdl: Optional[float] = None
    ldl: Optional[float] = None
    fasting_blood_sugar: Optional[str] = None
    hba1c: Optional[float] = None
    diabetes: Optional[str] = None
    resting_ecg: Optional[str] = None
    exercise_angina: Optional[str] = None
    oldpeak: Optional[float] = None
    st_slope: Optional[str] = None
    num_major_vessels: Optional[float] = None
    thalassemia: Optional[str] = None
    smoking: Optional[str] = None
    smoking_status: Optional[str] = None
    family_history: Optional[str] = None
    physical_activity: Optional[str] = None
    stress_level: Optional[float] = None

    class Config:
        extra = Extra.allow


class TemporalFeature(BaseModel):
    feature_name: str
    label: str
    unit: str
    feature_type: str  # 'numerical' or 'categorical'
    previous_value: Optional[Any] = None
    current_value: Optional[Any] = None
    delta_value: Optional[float] = None
    changed: Optional[bool] = None
    status: str  # 'available' or 'no_previous_visit'


class TemporalSnapshot(BaseModel):
    patient_id: str
    visit_id: Optional[int] = None
    assessment_date: Optional[str] = None
    is_first_visit: bool
    has_previous_visit: bool
    delta_systolic_bp: Optional[float] = None
    delta_diastolic_bp: Optional[float] = None
    delta_cholesterol: Optional[float] = None
    delta_ldl: Optional[float] = None
    delta_hdl: Optional[float] = None
    delta_bmi: Optional[float] = None
    delta_hba1c: Optional[float] = None
    delta_resting_heart_rate: Optional[float] = None
    smoking_status_changed: Optional[bool] = None
    physical_activity_changed: Optional[bool] = None
    numerical_details: Optional[Dict[str, Dict[str, Any]]] = None
    categorical_details: Optional[Dict[str, Dict[str, Any]]] = None


class TemporalAnalytics(BaseModel):
    total_patients: int
    total_visits: int
    patients_with_multiple_visits: int
    patients_with_one_visit: int
    average_visits_per_patient: float
    maximum_visits_per_patient: int
    temporal_snapshots: int
    numeric_delta_availability: int
    temporal_variables_count: int = 10
    numerical_delta_features_count: int = 8
    categorical_change_features_count: int = 2


class PredictionResult(BaseModel):
    patient_id: str
    patient_code: Optional[str] = None
    patient_name: Optional[str] = None
    visit_number: int
    number_of_visits: int
    is_first_visit: bool
    has_previous_visit: bool
    risk_prediction: int
    risk_probability: float
    risk_level: str
    model_name: str
    current_values: Dict[str, Any]
    previous_values: Dict[str, Any]
    temporal_features: Dict[str, Any]
    timeline: List[Dict[str, Any]] = []

    class Config:
        extra = Extra.allow
