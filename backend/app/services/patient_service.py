"""
HeartSense – Patient Service

CRUD operations for patients, visits, and predictions using SQLAlchemy.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from app.models import Patient, PatientVisit, Prediction
from app.config import FEATURE_NAMES


# ── Patient CRUD ───────────────────────────────────────────────────────────────

def create_patient(session: Session, patient_code: str, name: str = None) -> Patient:
    """Create a new patient or return existing by patient_code."""
    existing = session.query(Patient).filter_by(patient_code=patient_code).first()
    if existing:
        return existing
    patient = Patient(patient_code=patient_code, name=name)
    session.add(patient)
    session.commit()
    session.refresh(patient)
    return patient


def get_patient_by_code(session: Session, patient_code: str) -> Patient | None:
    """Look up a patient by their code."""
    return session.query(Patient).filter_by(patient_code=patient_code).first()


def get_all_patients(session: Session) -> list[Patient]:
    """Return all patients ordered by creation date."""
    return session.query(Patient).order_by(Patient.created_at.desc()).all()


# ── Visit CRUD ─────────────────────────────────────────────────────────────────

def create_visit(
    session: Session,
    patient_id: int,
    visit_timestamp: datetime,
    clinical_values: dict,
) -> PatientVisit:
    """
    Record a new clinical visit for a patient.

    *clinical_values* should be a dict with feature names as keys.
    """
    visit = PatientVisit(
        patient_id=patient_id,
        visit_timestamp=visit_timestamp,
        **{k: v for k, v in clinical_values.items() if k in FEATURE_NAMES},
    )
    session.add(visit)
    session.commit()
    session.refresh(visit)
    return visit


def get_patient_visits(session: Session, patient_id: int) -> list[PatientVisit]:
    """Return all visits for a patient, ordered chronologically."""
    return (
        session.query(PatientVisit)
        .filter_by(patient_id=patient_id)
        .order_by(PatientVisit.visit_timestamp.asc())
        .all()
    )


def get_latest_visit(session: Session, patient_id: int) -> PatientVisit | None:
    """Return the most recent visit for a patient."""
    return (
        session.query(PatientVisit)
        .filter_by(patient_id=patient_id)
        .order_by(PatientVisit.visit_timestamp.desc())
        .first()
    )


def get_previous_visit(session: Session, patient_id: int, before_visit_id: int) -> PatientVisit | None:
    """Return the visit immediately before *before_visit_id*."""
    current = session.query(PatientVisit).get(before_visit_id)
    if not current:
        return None
    return (
        session.query(PatientVisit)
        .filter(
            PatientVisit.patient_id == patient_id,
            PatientVisit.visit_timestamp < current.visit_timestamp,
        )
        .order_by(PatientVisit.visit_timestamp.desc())
        .first()
    )


# ── Prediction CRUD ───────────────────────────────────────────────────────────

def create_prediction(
    session: Session,
    patient_id: int,
    visit_id: int,
    prediction: int,
    probability: float | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
) -> Prediction:
    """Store a prediction result linked to a visit."""
    pred = Prediction(
        patient_id=patient_id,
        visit_id=visit_id,
        prediction=prediction,
        probability=probability,
        model_name=model_name,
        model_version=model_version,
    )
    session.add(pred)
    session.commit()
    session.refresh(pred)
    return pred


def get_patient_predictions(session: Session, patient_id: int) -> list[Prediction]:
    """Return all predictions for a patient, ordered by visit timestamp."""
    return (
        session.query(Prediction)
        .filter_by(patient_id=patient_id)
        .join(PatientVisit, Prediction.visit_id == PatientVisit.id)
        .order_by(PatientVisit.visit_timestamp.asc())
        .all()
    )
