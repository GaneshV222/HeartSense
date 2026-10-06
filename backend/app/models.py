"""
HeartSense – SQLAlchemy ORM Models

Tables:
  • patients                  – master patient record
  • patient_visits            – stores original/current patient clinical & lifestyle records
  • patient_temporal_features – stores longitudinal temporal snapshot per visit (10 variables: 8 numerical deltas, 2 categorical changes)
  • predictions               – ML risk predictions linked to patient visits
  • model_runs                – metadata and evaluation metrics for training runs
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, JSON, Boolean,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_code = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    visits = relationship("PatientVisit", back_populates="patient", order_by="PatientVisit.visit_date")
    predictions = relationship("Prediction", back_populates="patient")

    def __repr__(self):
        return f"<Patient {self.patient_code}>"


class PatientVisit(Base):
    __tablename__ = "patient_visits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    source_patient_id = Column(String(100), nullable=False, index=True)
    visit_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    visit_date = Column(DateTime, nullable=False, index=True)

    # ── Original Dataset Measurements ──────────────────────────────────────
    age = Column(Float, nullable=True)
    gender = Column(String(50), nullable=True)
    bmi = Column(Float, nullable=True)
    chest_pain_type = Column(String(100), nullable=True)
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    resting_heart_rate = Column(Float, nullable=True)
    max_heart_rate = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    hdl = Column(Float, nullable=True)
    ldl = Column(Float, nullable=True)
    fasting_blood_sugar = Column(String(50), nullable=True)
    hba1c = Column(Float, nullable=True)
    diabetes = Column(String(50), nullable=True)
    resting_ecg = Column(String(100), nullable=True)
    exercise_angina = Column(String(50), nullable=True)
    oldpeak = Column(Float, nullable=True)
    st_slope = Column(String(50), nullable=True)
    num_major_vessels = Column(Float, nullable=True)
    thalassemia = Column(String(100), nullable=True)
    smoking = Column(String(50), nullable=True)
    smoking_status = Column(String(50), nullable=True)
    family_history = Column(String(50), nullable=True)
    physical_activity = Column(String(100), nullable=True)
    stress_level = Column(Float, nullable=True)

    target = Column(Float, nullable=True)
    raw_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="visits")
    prediction = relationship("Prediction", back_populates="visit", uselist=False)
    temporal_feature = relationship("PatientTemporalFeature", back_populates="visit", uselist=False)

    def to_feature_dict(self) -> dict:
        """Return the clinical & lifestyle features as a dictionary."""
        return {
            "patient_id": self.source_patient_id,
            "visit_id": self.id,
            "visit_date": self.visit_date.isoformat() if self.visit_date else None,
            "assessment_date": self.visit_date.isoformat() if self.visit_date else None,
            "age": self.age,
            "gender": self.gender,
            "bmi": self.bmi,
            "chest_pain_type": self.chest_pain_type,
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "resting_heart_rate": self.resting_heart_rate,
            "max_heart_rate": self.max_heart_rate,
            "cholesterol": self.cholesterol,
            "hdl": self.hdl,
            "ldl": self.ldl,
            "fasting_blood_sugar": self.fasting_blood_sugar,
            "hba1c": self.hba1c,
            "diabetes": self.diabetes,
            "resting_ecg": self.resting_ecg,
            "exercise_angina": self.exercise_angina,
            "oldpeak": self.oldpeak,
            "st_slope": self.st_slope,
            "num_major_vessels": self.num_major_vessels,
            "thalassemia": self.thalassemia,
            "smoking": self.smoking_status or self.smoking,
            "smoking_status": self.smoking_status or self.smoking,
            "family_history": self.family_history,
            "physical_activity": self.physical_activity,
            "stress_level": self.stress_level,
            "target": self.target,
        }

    def __repr__(self):
        return f"<PatientVisit {self.source_patient_id} @ {self.visit_date}>"


class PatientTemporalFeature(Base):
    """
    Longitudinal temporal representation for a clinical visit.
    Contains exactly the 10 core temporal clinical variables:
      8 Numerical: Systolic BP, Diastolic BP, Cholesterol, LDL, HDL, BMI, HbA1c, Resting Heart Rate (previous, current, delta)
      2 Categorical: Smoking Status, Physical Activity (previous, current, changed flag)
    """
    __tablename__ = "patient_temporal_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(100), nullable=False, index=True)
    visit_id = Column(Integer, ForeignKey("patient_visits.id"), nullable=True, index=True)
    assessment_date = Column(DateTime, nullable=False, index=True)

    # ── 1. Systolic BP ──────────────────────────────────────────────────────
    previous_systolic_bp = Column(Float, nullable=True)
    current_systolic_bp = Column(Float, nullable=True)
    delta_systolic_bp = Column(Float, nullable=True)

    # ── 2. Diastolic BP ─────────────────────────────────────────────────────
    previous_diastolic_bp = Column(Float, nullable=True)
    current_diastolic_bp = Column(Float, nullable=True)
    delta_diastolic_bp = Column(Float, nullable=True)

    # ── 3. Cholesterol ──────────────────────────────────────────────────────
    previous_cholesterol = Column(Float, nullable=True)
    current_cholesterol = Column(Float, nullable=True)
    delta_cholesterol = Column(Float, nullable=True)

    # ── 4. LDL ──────────────────────────────────────────────────────────────
    previous_ldl = Column(Float, nullable=True)
    current_ldl = Column(Float, nullable=True)
    delta_ldl = Column(Float, nullable=True)

    # ── 5. HDL ──────────────────────────────────────────────────────────────
    previous_hdl = Column(Float, nullable=True)
    current_hdl = Column(Float, nullable=True)
    delta_hdl = Column(Float, nullable=True)

    # ── 6. BMI ──────────────────────────────────────────────────────────────
    previous_bmi = Column(Float, nullable=True)
    current_bmi = Column(Float, nullable=True)
    delta_bmi = Column(Float, nullable=True)

    # ── 7. HbA1c ────────────────────────────────────────────────────────────
    previous_hba1c = Column(Float, nullable=True)
    current_hba1c = Column(Float, nullable=True)
    delta_hba1c = Column(Float, nullable=True)

    # ── 8. Resting Heart Rate ───────────────────────────────────────────────
    previous_resting_heart_rate = Column(Float, nullable=True)
    current_resting_heart_rate = Column(Float, nullable=True)
    delta_resting_heart_rate = Column(Float, nullable=True)

    # ── 9. Smoking Status ───────────────────────────────────────────────────
    previous_smoking_status = Column(String(50), nullable=True)
    current_smoking_status = Column(String(50), nullable=True)
    smoking_status_changed = Column(Boolean, nullable=True)

    # ── 10. Physical Activity ───────────────────────────────────────────────
    previous_physical_activity = Column(String(100), nullable=True)
    current_physical_activity = Column(String(100), nullable=True)
    physical_activity_changed = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    visit = relationship("PatientVisit", back_populates="temporal_feature")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "visit_id": self.visit_id,
            "assessment_date": self.assessment_date.isoformat() if self.assessment_date else None,
            "previous_systolic_bp": self.previous_systolic_bp,
            "current_systolic_bp": self.current_systolic_bp,
            "delta_systolic_bp": self.delta_systolic_bp,
            "previous_diastolic_bp": self.previous_diastolic_bp,
            "current_diastolic_bp": self.current_diastolic_bp,
            "delta_diastolic_bp": self.delta_diastolic_bp,
            "previous_cholesterol": self.previous_cholesterol,
            "current_cholesterol": self.current_cholesterol,
            "delta_cholesterol": self.delta_cholesterol,
            "previous_ldl": self.previous_ldl,
            "current_ldl": self.current_ldl,
            "delta_ldl": self.delta_ldl,
            "previous_hdl": self.previous_hdl,
            "current_hdl": self.current_hdl,
            "delta_hdl": self.delta_hdl,
            "previous_bmi": self.previous_bmi,
            "current_bmi": self.current_bmi,
            "delta_bmi": self.delta_bmi,
            "previous_hba1c": self.previous_hba1c,
            "current_hba1c": self.current_hba1c,
            "delta_hba1c": self.delta_hba1c,
            "previous_resting_heart_rate": self.previous_resting_heart_rate,
            "current_resting_heart_rate": self.current_resting_heart_rate,
            "delta_resting_heart_rate": self.delta_resting_heart_rate,
            "previous_smoking_status": self.previous_smoking_status,
            "current_smoking_status": self.current_smoking_status,
            "smoking_status_changed": self.smoking_status_changed,
            "previous_physical_activity": self.previous_physical_activity,
            "current_physical_activity": self.current_physical_activity,
            "physical_activity_changed": self.physical_activity_changed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<PatientTemporalFeature {self.patient_id} @ {self.assessment_date}>"


# Backwards compatibility alias
TemporalPatientFeature = PatientTemporalFeature


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    patient_code = Column(String(100), nullable=True, index=True)
    visit_id = Column(Integer, ForeignKey("patient_visits.id"), nullable=True)
    visit_date = Column(DateTime, nullable=True)
    prediction = Column(Integer, nullable=False)          # 0 or 1
    probability = Column(Float, nullable=True)            # P(disease)
    risk_level = Column(String(50), nullable=True)
    model_name = Column(String(100), nullable=True)
    model_type = Column(String(50), nullable=True, default="temporal")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")
    visit = relationship("PatientVisit", back_populates="prediction")

    def __repr__(self):
        return f"<Prediction patient={self.patient_code or self.patient_id} pred={self.prediction} prob={self.probability}>"


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=True, default="temporal") # static vs temporal
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    selected_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelRun {self.model_name} ({self.model_type}) @ {self.created_at}>"
