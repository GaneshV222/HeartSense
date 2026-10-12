"""
HeartSense â€“ SQLAlchemy ORM Models

Tables:
  â€¢ patients               â€“ unique master patient records (patient_id PK)
  â€¢ patient_visits         â€“ permanent historical visits for all patients
  â€¢ temporal_patient_data  â€“ ONLY the latest two visits per patient for approved temporal changes
  â€¢ smote_training_records â€“ complete resampled training rows generated from SMOTE
  â€¢ predictions            â€“ ML CVD risk predictions linked to patient visits
  â€¢ model_runs             â€“ metadata and evaluation metrics for training runs
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
    patient_code = Column(String(100), nullable=True, index=True)
    name = Column(String(200), nullable=True)
    gender = Column(String(50), nullable=True)
    age = Column(Float, nullable=True)
    source_dataset = Column(String(255), nullable=True)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.patient_code is None and self.patient_id is not None:
            self.patient_code = self.patient_id

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

    # â”€â”€ Clinical & Lifestyle Features from Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    is_synthetic = Column(Boolean, nullable=False, default=False)  # True for SMOTE-generated rows
    source_dataset = Column(String(255), nullable=True)
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

    @property
    def sex(self):
        if self.gender is None:
            return None
        gender = str(self.gender).strip().lower()
        if gender in {"male", "m", "1"}:
            return 1
        if gender in {"female", "f", "0"}:
            return 0
        return None

    @sex.setter
    def sex(self, value):
        if value in (1, "1", "male", "m"):
            self.gender = "Male"
        elif value in (0, "0", "female", "f"):
            self.gender = "Female"
        else:
            self.gender = value

    @property
    def cp(self):
        value = self.chest_pain_type
        if value is None:
            return None
        mapping = {
            "typical angina": 0,
            "typical_angina": 0,
            "atypical angina": 1,
            "atypical_angina": 1,
            "non-anginal pain": 2,
            "non_anginal_pain": 2,
            "asymptomatic": 3,
        }
        return mapping.get(str(value).strip().lower(), value)

    @cp.setter
    def cp(self, value):
        mapping = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
        self.chest_pain_type = mapping.get(value, value)

    @property
    def trestbps(self):
        return self.systolic_bp

    @trestbps.setter
    def trestbps(self, value):
        self.systolic_bp = value

    @property
    def chol(self):
        return self.cholesterol

    @chol.setter
    def chol(self, value):
        self.cholesterol = value

    @property
    def fbs(self):
        value = self.fasting_blood_sugar
        if value is None:
            return None
        if isinstance(value, str):
            return 1 if value.strip().lower() in {"true", "yes", "1", "y"} else 0 if value.strip().lower() in {"false", "no", "0", "n"} else None
        return int(bool(value))

    @fbs.setter
    def fbs(self, value):
        self.fasting_blood_sugar = "true" if bool(value) else "false"

    @property
    def restecg(self):
        value = self.resting_ecg
        if value is None:
            return None
        mapping = {"normal": 0, "horizontal": 1, "st-t wave abnormality": 1, "st_t_wave_abnormality": 1, "left ventricular hypertrophy": 2}
        key = str(value).strip().lower()
        return mapping.get(key, value)

    @restecg.setter
    def restecg(self, value):
        mapping = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
        self.resting_ecg = mapping.get(value, value)

    @property
    def thalach(self):
        return self.max_heart_rate

    @thalach.setter
    def thalach(self, value):
        self.max_heart_rate = value

    @property
    def exang(self):
        value = self.exercise_angina
        if value is None:
            return None
        if isinstance(value, str):
            return 1 if value.strip().lower() in {"true", "yes", "1", "y"} else 0 if value.strip().lower() in {"false", "no", "0", "n"} else None
        return int(bool(value))

    @exang.setter
    def exang(self, value):
        self.exercise_angina = "true" if bool(value) else "false"

    @property
    def slope(self):
        value = self.st_slope
        if value is None:
            return None
        mapping = {"up": 1, "flat": 2, "down": 3}
        return mapping.get(str(value).strip().lower(), value)

    @slope.setter
    def slope(self, value):
        mapping = {1: "Up", 2: "Flat", 3: "Down"}
        self.st_slope = mapping.get(value, value)

    @property
    def ca(self):
        return self.num_major_vessels

    @ca.setter
    def ca(self, value):
        self.num_major_vessels = value

    @property
    def thal(self):
        value = self.thalassemia
        if value is None:
            return None
        mapping = {"normal": 1, "fixed defect": 2, "fixed_defect": 2, "reversible defect": 3, "reversible_defect": 3}
        return mapping.get(str(value).strip().lower(), value)

    @thal.setter
    def thal(self, value):
        mapping = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
        self.thalassemia = mapping.get(value, value)

    def to_feature_dict(self) -> dict:
        """Return the clinical & lifestyle features as a dictionary."""
        data = {
            "patient_id": self.patient_id,
            "patient_code": self.patient_id,
            "visit_id": self.id,
            "visit_number": self.visit_number,
            "visit_date": self.visit_date.isoformat() if self.visit_date else None,
            "assessment_date": self.visit_date.isoformat() if self.visit_date else None,
            "age": self.age,
            "gender": self.gender,
            "sex": self.sex,
            "bmi": self.bmi,
            "chest_pain_type": self.chest_pain_type,
            "cp": self.cp,
            "systolic_bp": self.systolic_bp,
            "trestbps": self.trestbps,
            "diastolic_bp": self.diastolic_bp,
            "resting_heart_rate": self.resting_heart_rate,
            "max_heart_rate": self.max_heart_rate,
            "thalach": self.thalach,
            "cholesterol": self.cholesterol,
            "chol": self.chol,
            "hdl": self.hdl,
            "ldl": self.ldl,
            "fasting_blood_sugar": self.fasting_blood_sugar,
            "fbs": self.fbs,
            "hba1c": self.hba1c,
            "diabetes": self.diabetes,
            "resting_ecg": self.resting_ecg,
            "restecg": self.restecg,
            "exercise_angina": self.exercise_angina,
            "exang": self.exang,
            "oldpeak": self.oldpeak,
            "st_slope": self.st_slope,
            "slope": self.slope,
            "num_major_vessels": self.num_major_vessels,
            "ca": self.ca,
            "thalassemia": self.thalassemia,
            "thal": self.thal,
            "smoking": self.smoking or self.smoking_status,
            "smoking_status": self.smoking_status or self.smoking,
            "family_history": self.family_history,
            "physical_activity": self.physical_activity,
            "stress_level": self.stress_level,
            "prediction": self.prediction,
            "target": self.prediction,
        }
        return data

    def __repr__(self):
        return f"<PatientVisit {self.patient_id} #{self.visit_number} @ {self.visit_date}>"


class TemporalPatientData(Base):
    """
    Temporal representation for a patient.
    Contains ONLY the latest two visits (previous and current) for approved temporal changes.
    Upserted per patient â€“ exactly one row per patient.
    """
    __tablename__ = "temporal_patient_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(100), ForeignKey("patients.patient_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    source_dataset = Column(String(255), nullable=True)

    current_visit_id = Column(Integer, ForeignKey("patient_visits.id", ondelete="SET NULL"), nullable=True)
    previous_visit_id = Column(Integer, ForeignKey("patient_visits.id", ondelete="SET NULL"), nullable=True)
    current_visit_date = Column(DateTime, nullable=True)
    previous_visit_date = Column(DateTime, nullable=True)
    days_between_visits = Column(Float, nullable=True)

    # â”€â”€ 1. Systolic BP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_systolic_bp = Column(Float, nullable=True)
    previous_systolic_bp = Column(Float, nullable=True)
    systolic_bp_change = Column(Float, nullable=True)
    systolic_bp_rate = Column(Float, nullable=True)

    # â”€â”€ 2. Diastolic BP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_diastolic_bp = Column(Float, nullable=True)
    previous_diastolic_bp = Column(Float, nullable=True)
    diastolic_bp_change = Column(Float, nullable=True)
    diastolic_bp_rate = Column(Float, nullable=True)

    # â”€â”€ 3. Total Cholesterol â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_cholesterol = Column(Float, nullable=True)
    previous_cholesterol = Column(Float, nullable=True)
    cholesterol_change = Column(Float, nullable=True)
    cholesterol_rate = Column(Float, nullable=True)

    # â”€â”€ 4. LDL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_ldl = Column(Float, nullable=True)
    previous_ldl = Column(Float, nullable=True)
    ldl_change = Column(Float, nullable=True)
    ldl_rate = Column(Float, nullable=True)

    # â”€â”€ 5. HDL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_hdl = Column(Float, nullable=True)
    previous_hdl = Column(Float, nullable=True)
    hdl_change = Column(Float, nullable=True)
    hdl_rate = Column(Float, nullable=True)

    # â”€â”€ 6. BMI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_bmi = Column(Float, nullable=True)
    previous_bmi = Column(Float, nullable=True)
    bmi_change = Column(Float, nullable=True)
    bmi_rate = Column(Float, nullable=True)

    # â”€â”€ 7. HbA1c â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_hba1c = Column(Float, nullable=True)
    previous_hba1c = Column(Float, nullable=True)
    hba1c_change = Column(Float, nullable=True)
    hba1c_rate = Column(Float, nullable=True)

    # â”€â”€ 8. Resting Heart Rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_resting_heart_rate = Column(Float, nullable=True)
    previous_resting_heart_rate = Column(Float, nullable=True)
    resting_heart_rate_change = Column(Float, nullable=True)
    resting_heart_rate_rate = Column(Float, nullable=True)

    # â”€â”€ 9. Smoking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_smoking = Column(String(50), nullable=True)
    previous_smoking = Column(String(50), nullable=True)
    smoking_changed = Column(Boolean, nullable=True)

    # â”€â”€ 10. Physical Activity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    current_physical_activity = Column(String(100), nullable=True)
    previous_physical_activity = Column(String(100), nullable=True)
    physical_activity_changed = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="temporal_data")



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
