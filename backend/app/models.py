"""
HeartSense – SQLAlchemy ORM Models

Tables:
  • patients               – unique master patient records (patient_id PK)
  • patient_visits         – permanent historical visits for all patients
  • temporal_patient_data  – ONLY the latest two visits per patient for temporal deltas and rates
  • predictions            – ML CVD risk predictions linked to patient visits
  • model_runs             – metadata and evaluation metrics for training runs
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, JSON, Boolean,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    """
    Master patient record. Unique per patient_id (no duplicate patient records).
    Stores static and demographic information.
    """
    __tablename__ = "patients"

    patient_id = Column(String(100), primary_key=True, index=True)
    name = Column(String(200), nullable=True)
    gender = Column(String(50), nullable=True)
    age = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    visits = relationship(
        "PatientVisit",
        back_populates="patient",
        order_by="PatientVisit.visit_date",
        cascade="all, delete-orphan",
    )
    temporal_data = relationship(
        "TemporalPatientData",
        back_populates="patient",
        uselist=False,
        cascade="all, delete-orphan",
    )
    predictions = relationship(
        "Prediction",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    @property
    def id(self) -> str:
        return self.patient_id

    @property
    def patient_code(self) -> str:
        return self.patient_id

    def __repr__(self):
        return f"<Patient {self.patient_id}>"


class PatientVisit(Base):
    """
    Permanent visit history table for all patient visits.
    Contains the original measurements and features for each visit.
    Every visit is permanently recorded here and never deleted.
    """
    __tablename__ = "patient_visits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(100), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)
    visit_number = Column(Integer, nullable=True)
    visit_date = Column(DateTime, nullable=False, index=True)
    visit_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # ── Clinical & Lifestyle Features from Dataset ─────────────────────────
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

    prediction = Column(Integer, nullable=True)  # 0 or 1 dataset target
    raw_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="visits")
    prediction_record = relationship("Prediction", back_populates="visit", uselist=False)

    @property
    def source_patient_id(self) -> str:
        return self.patient_id

    @property
    def target(self) -> int | None:
        return self.prediction

    def to_feature_dict(self) -> dict:
        """Return the clinical & lifestyle features as a dictionary."""
        return {
            "patient_id": self.patient_id,
            "patient_code": self.patient_id,
            "visit_id": self.id,
            "visit_number": self.visit_number,
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
            "smoking": self.smoking or self.smoking_status,
            "smoking_status": self.smoking_status or self.smoking,
            "family_history": self.family_history,
            "physical_activity": self.physical_activity,
            "stress_level": self.stress_level,
            "prediction": self.prediction,
            "target": self.prediction,
        }

    def __repr__(self):
        return f"<PatientVisit {self.patient_id} #{self.visit_number} @ {self.visit_date}>"


class TemporalPatientData(Base):
    """
    Temporal representation for a patient.
    Contains ONLY the latest two visits (previous and current) for calculating temporal changes and rates.
    Upserted per patient – exactly one row per patient.
    """
    __tablename__ = "temporal_patient_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(100), ForeignKey("patients.patient_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)

    current_visit_id = Column(Integer, ForeignKey("patient_visits.id", ondelete="SET NULL"), nullable=True)
    previous_visit_id = Column(Integer, ForeignKey("patient_visits.id", ondelete="SET NULL"), nullable=True)
    current_visit_date = Column(DateTime, nullable=True)
    previous_visit_date = Column(DateTime, nullable=True)
    days_between_visits = Column(Float, nullable=True)

    # ── 1. Systolic BP ──────────────────────────────────────────────────────
    current_systolic_bp = Column(Float, nullable=True)
    previous_systolic_bp = Column(Float, nullable=True)
    systolic_bp_change = Column(Float, nullable=True)
    systolic_bp_rate = Column(Float, nullable=True)

    # ── 2. Diastolic BP ─────────────────────────────────────────────────────
    current_diastolic_bp = Column(Float, nullable=True)
    previous_diastolic_bp = Column(Float, nullable=True)
    diastolic_bp_change = Column(Float, nullable=True)
    diastolic_bp_rate = Column(Float, nullable=True)

    # ── 3. Total Cholesterol ────────────────────────────────────────────────
    current_cholesterol = Column(Float, nullable=True)
    previous_cholesterol = Column(Float, nullable=True)
    cholesterol_change = Column(Float, nullable=True)
    cholesterol_rate = Column(Float, nullable=True)

    # ── 4. LDL ──────────────────────────────────────────────────────────────
    current_ldl = Column(Float, nullable=True)
    previous_ldl = Column(Float, nullable=True)
    ldl_change = Column(Float, nullable=True)
    ldl_rate = Column(Float, nullable=True)

    # ── 5. HDL ──────────────────────────────────────────────────────────────
    current_hdl = Column(Float, nullable=True)
    previous_hdl = Column(Float, nullable=True)
    hdl_change = Column(Float, nullable=True)
    hdl_rate = Column(Float, nullable=True)

    # ── 6. BMI ──────────────────────────────────────────────────────────────
    current_bmi = Column(Float, nullable=True)
    previous_bmi = Column(Float, nullable=True)
    bmi_change = Column(Float, nullable=True)
    bmi_rate = Column(Float, nullable=True)

    # ── 7. HbA1c ────────────────────────────────────────────────────────────
    current_hba1c = Column(Float, nullable=True)
    previous_hba1c = Column(Float, nullable=True)
    hba1c_change = Column(Float, nullable=True)
    hba1c_rate = Column(Float, nullable=True)

    # ── 8. Resting Heart Rate ───────────────────────────────────────────────
    current_resting_heart_rate = Column(Float, nullable=True)
    previous_resting_heart_rate = Column(Float, nullable=True)
    resting_heart_rate_change = Column(Float, nullable=True)
    resting_heart_rate_rate = Column(Float, nullable=True)

    # ── 9. Smoking ──────────────────────────────────────────────────────────
    current_smoking = Column(String(50), nullable=True)
    previous_smoking = Column(String(50), nullable=True)
    smoking_changed = Column(Boolean, nullable=True)

    # ── 10. Physical Activity ───────────────────────────────────────────────
    current_physical_activity = Column(String(100), nullable=True)
    previous_physical_activity = Column(String(100), nullable=True)
    physical_activity_changed = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="temporal_data")

    # Compatibility aliases
    @property
    def delta_systolic_bp(self): return self.systolic_bp_change
    @property
    def delta_diastolic_bp(self): return self.diastolic_bp_change
    @property
    def delta_cholesterol(self): return self.cholesterol_change
    @property
    def delta_ldl(self): return self.ldl_change
    @property
    def delta_hdl(self): return self.hdl_change
    @property
    def delta_bmi(self): return self.bmi_change
    @property
    def delta_hba1c(self): return self.hba1c_change
    @property
    def delta_resting_heart_rate(self): return self.resting_heart_rate_change
    @property
    def current_smoking_status(self): return self.current_smoking
    @property
    def previous_smoking_status(self): return self.previous_smoking
    @property
    def smoking_status_changed(self): return self.smoking_changed

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "current_visit_id": self.current_visit_id,
            "previous_visit_id": self.previous_visit_id,
            "current_visit_date": self.current_visit_date.isoformat() if self.current_visit_date else None,
            "previous_visit_date": self.previous_visit_date.isoformat() if self.previous_visit_date else None,
            "days_between_visits": self.days_between_visits,
            # Systolic BP
            "current_systolic_bp": self.current_systolic_bp,
            "previous_systolic_bp": self.previous_systolic_bp,
            "systolic_bp_change": self.systolic_bp_change,
            "systolic_bp_rate": self.systolic_bp_rate,
            "delta_systolic_bp": self.systolic_bp_change,
            # Diastolic BP
            "current_diastolic_bp": self.current_diastolic_bp,
            "previous_diastolic_bp": self.previous_diastolic_bp,
            "diastolic_bp_change": self.diastolic_bp_change,
            "diastolic_bp_rate": self.diastolic_bp_rate,
            "delta_diastolic_bp": self.diastolic_bp_change,
            # Cholesterol
            "current_cholesterol": self.current_cholesterol,
            "previous_cholesterol": self.previous_cholesterol,
            "cholesterol_change": self.cholesterol_change,
            "cholesterol_rate": self.cholesterol_rate,
            "delta_cholesterol": self.cholesterol_change,
            # LDL
            "current_ldl": self.current_ldl,
            "previous_ldl": self.previous_ldl,
            "ldl_change": self.ldl_change,
            "ldl_rate": self.ldl_rate,
            "delta_ldl": self.ldl_change,
            # HDL
            "current_hdl": self.current_hdl,
            "previous_hdl": self.previous_hdl,
            "hdl_change": self.hdl_change,
            "hdl_rate": self.hdl_rate,
            "delta_hdl": self.hdl_change,
            # BMI
            "current_bmi": self.current_bmi,
            "previous_bmi": self.previous_bmi,
            "bmi_change": self.bmi_change,
            "bmi_rate": self.bmi_rate,
            "delta_bmi": self.bmi_change,
            # HbA1c
            "current_hba1c": self.current_hba1c,
            "previous_hba1c": self.previous_hba1c,
            "hba1c_change": self.hba1c_change,
            "hba1c_rate": self.hba1c_rate,
            "delta_hba1c": self.hba1c_change,
            # Resting Heart Rate
            "current_resting_heart_rate": self.current_resting_heart_rate,
            "previous_resting_heart_rate": self.previous_resting_heart_rate,
            "resting_heart_rate_change": self.resting_heart_rate_change,
            "resting_heart_rate_rate": self.resting_heart_rate_rate,
            "delta_resting_heart_rate": self.resting_heart_rate_change,
            # Smoking
            "current_smoking": self.current_smoking,
            "previous_smoking": self.previous_smoking,
            "smoking_changed": self.smoking_changed,
            "current_smoking_status": self.current_smoking,
            "previous_smoking_status": self.previous_smoking,
            "smoking_status_changed": self.smoking_changed,
            # Physical Activity
            "current_physical_activity": self.current_physical_activity,
            "previous_physical_activity": self.previous_physical_activity,
            "physical_activity_changed": self.physical_activity_changed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<TemporalPatientData {self.patient_id}>"


# Backwards compatibility alias
PatientTemporalFeature = TemporalPatientData
TemporalPatientFeature = TemporalPatientData


class Prediction(Base):
    """
    ML CVD risk predictions linked to patient visits.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(100), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False, index=True)
    visit_id = Column(Integer, ForeignKey("patient_visits.id", ondelete="SET NULL"), nullable=True)
    visit_date = Column(DateTime, nullable=True)
    prediction = Column(Integer, nullable=False)          # 0 or 1
    probability = Column(Float, nullable=True)            # P(disease)
    risk_level = Column(String(50), nullable=True)
    model_name = Column(String(100), nullable=True)
    model_type = Column(String(50), nullable=True, default="temporal")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")
    visit = relationship("PatientVisit", back_populates="prediction_record")

    @property
    def patient_code(self) -> str:
        return self.patient_id

    def __repr__(self):
        return f"<Prediction patient={self.patient_id} pred={self.prediction} prob={self.probability}>"


class ModelRun(Base):
    """
    Training run metadata and evaluation metrics.
    """
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=True, default="temporal")
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    selected_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelRun {self.model_name} ({self.model_type}) @ {self.created_at}>"
