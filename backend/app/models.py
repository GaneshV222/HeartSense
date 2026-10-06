"""
HeartSense – SQLAlchemy ORM Models

Tables:
  • patients                  – master patient record
  • patient_visits            – stores original/current patient clinical & lifestyle records
  • temporal_patient_features – stores critical temporal features, deltas, rates, and current values
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
    family_history = Column(String(50), nullable=True)
    physical_activity = Column(String(100), nullable=True)
    stress_level = Column(Float, nullable=True)

    target = Column(Float, nullable=True)
    raw_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="visits")
    prediction = relationship("Prediction", back_populates="visit", uselist=False)

    def to_feature_dict(self) -> dict:
        """Return the clinical & lifestyle features as a dictionary."""
        return {
            "patient_id": self.source_patient_id,
            "visit_date": self.visit_date.isoformat() if self.visit_date else None,
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
            "smoking": self.smoking,
            "family_history": self.family_history,
            "physical_activity": self.physical_activity,
            "stress_level": self.stress_level,
            "target": self.target,
        }

    def __repr__(self):
        return f"<PatientVisit {self.source_patient_id} @ {self.visit_date}>"


class TemporalPatientFeature(Base):
    __tablename__ = "temporal_patient_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(100), nullable=False, index=True)
    visit_date = Column(DateTime, nullable=False, index=True)
    previous_visit_date = Column(DateTime, nullable=True)
    days_since_previous_visit = Column(Float, nullable=True)
    has_previous_visit = Column(Integer, nullable=False, default=0)
    visit_number = Column(Integer, nullable=True, default=1)

    # ── Current Critical Values ─────────────────────────────────────────────
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)

    # ── Previous Critical Values (NULL for first visit) ─────────────────────
    previous_systolic_bp = Column(Float, nullable=True)
    previous_diastolic_bp = Column(Float, nullable=True)
    previous_cholesterol = Column(Float, nullable=True)
    previous_bmi = Column(Float, nullable=True)

    # ── Temporal Changes (NULL for first visit) ─────────────────────────────
    systolic_bp_change = Column(Float, nullable=True)
    diastolic_bp_change = Column(Float, nullable=True)
    cholesterol_change = Column(Float, nullable=True)
    bmi_change = Column(Float, nullable=True)

    # ── Rates of Change (NULL for first visit) ──────────────────────────────
    systolic_bp_rate = Column(Float, nullable=True)
    diastolic_bp_rate = Column(Float, nullable=True)
    cholesterol_rate = Column(Float, nullable=True)
    bmi_rate = Column(Float, nullable=True)

    # ── Other Features (Current / Latest values only) ───────────────────────
    age = Column(Float, nullable=True)
    gender = Column(String(50), nullable=True)
    chest_pain_type = Column(String(100), nullable=True)
    resting_heart_rate = Column(Float, nullable=True)
    max_heart_rate = Column(Float, nullable=True)
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
    family_history = Column(String(50), nullable=True)
    physical_activity = Column(String(100), nullable=True)
    stress_level = Column(Float, nullable=True)

    target = Column(Float, nullable=True)
    features = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TemporalPatientFeature {self.patient_id} @ {self.visit_date} (has_prev={self.has_previous_visit})>"


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
