"""
HeartSense – SQLAlchemy ORM Models

Tables:
  • patients        – master patient record
  • patient_visits  – one row per clinical visit (stores all 13 features)
  • predictions     – ML prediction linked to a visit
  • model_runs      – metadata for each training run
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    visits = relationship("PatientVisit", back_populates="patient", order_by="PatientVisit.visit_timestamp")
    predictions = relationship("Prediction", back_populates="patient")

    def __repr__(self):
        return f"<Patient {self.patient_code}>"


class PatientVisit(Base):
    __tablename__ = "patient_visits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    visit_timestamp = Column(DateTime, nullable=False)

    # ── Cleveland dataset clinical features ─────────────────────────────────
    age = Column(Float, nullable=True)
    sex = Column(Float, nullable=True)
    cp = Column(Float, nullable=True)
    trestbps = Column(Float, nullable=True)
    chol = Column(Float, nullable=True)
    fbs = Column(Float, nullable=True)
    restecg = Column(Float, nullable=True)
    thalach = Column(Float, nullable=True)
    exang = Column(Float, nullable=True)
    oldpeak = Column(Float, nullable=True)
    slope = Column(Float, nullable=True)
    ca = Column(Float, nullable=True)
    thal = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="visits")
    prediction = relationship("Prediction", back_populates="visit", uselist=False)

    def to_feature_dict(self) -> dict:
        """Return the clinical features as a dict keyed by feature name."""
        return {
            "age": self.age,
            "sex": self.sex,
            "cp": self.cp,
            "trestbps": self.trestbps,
            "chol": self.chol,
            "fbs": self.fbs,
            "restecg": self.restecg,
            "thalach": self.thalach,
            "exang": self.exang,
            "oldpeak": self.oldpeak,
            "slope": self.slope,
            "ca": self.ca,
            "thal": self.thal,
        }

    def __repr__(self):
        return f"<Visit patient_id={self.patient_id} @ {self.visit_timestamp}>"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    visit_id = Column(Integer, ForeignKey("patient_visits.id"), nullable=False)
    prediction = Column(Integer, nullable=False)          # 0 or 1
    probability = Column(Float, nullable=True)            # P(disease)
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")
    visit = relationship("PatientVisit", back_populates="prediction")

    def __repr__(self):
        return f"<Prediction visit={self.visit_id} pred={self.prediction}>"


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    selected_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelRun {self.model_name} @ {self.created_at}>"
