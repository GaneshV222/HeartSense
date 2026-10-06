"""
HeartSense – Patient Service

CRUD operations for patients, visits, and predictions using SQLAlchemy.
"""

from datetime import datetime
from sqlalchemy.orm import Session
from app.models import Patient, PatientVisit, Prediction


# ── Patient CRUD ───────────────────────────────────────────────────────────────

def create_patient(
    session: Session,
    patient_id: str,
    name: str | None = None,
    gender: str | None = None,
    age: float | None = None,
) -> Patient:
    """Create a new patient or return existing by patient_id."""
    existing = session.query(Patient).filter_by(patient_id=str(patient_id)).first()
    if existing:
        if name and not existing.name:
            existing.name = name
        if gender and not existing.gender:
            existing.gender = gender
        if age is not None and existing.age is None:
            existing.age = age
        session.flush()
        return existing

    patient = Patient(
        patient_id=str(patient_id),
        name=name,
        gender=gender,
        age=age,
        created_at=datetime.utcnow(),
    )
    session.add(patient)
    session.flush()
    return patient


def get_patient_by_id(session: Session, patient_id: str) -> Patient | None:
    """Look up a patient by their unique patient_id."""
    return session.query(Patient).filter_by(patient_id=str(patient_id)).first()


def get_patient_by_code(session: Session, patient_code: str) -> Patient | None:
    """Backward-compatible lookup by patient identifier."""
    return get_patient_by_id(session, patient_code)


def get_all_patients(session: Session) -> list[Patient]:
    """Return all patients ordered by creation date."""
    return session.query(Patient).order_by(Patient.created_at.desc()).all()


# ── Visit CRUD ─────────────────────────────────────────────────────────────────

def create_visit(
    session: Session,
    patient_id: str,
    visit_date: datetime,
    clinical_values: dict,
    visit_number: int | None = None,
) -> PatientVisit:
    """
    Record a new clinical visit for a patient permanently in patient_visits table.
    """
    valid_cols = {c.name for c in PatientVisit.__table__.columns if c.name not in {"id", "created_at", "updated_at"}}
    kwargs = {k: v for k, v in clinical_values.items() if k in valid_cols}

    if visit_number is None:
        count = session.query(PatientVisit).filter_by(patient_id=str(patient_id)).count()
        visit_number = count + 1

    visit = PatientVisit(
        patient_id=str(patient_id),
        visit_number=visit_number,
        visit_date=visit_date,
        visit_timestamp=visit_date,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        **kwargs,
    )
    session.add(visit)
    session.flush()
    return visit


def get_patient_visits(session: Session, patient_id: str) -> list[PatientVisit]:
    """Return all permanent visits for a patient, ordered chronologically."""
    return (
        session.query(PatientVisit)
        .filter_by(patient_id=str(patient_id))
        .order_by(PatientVisit.visit_date.asc(), PatientVisit.id.asc())
        .all()
    )


def get_latest_visit(session: Session, patient_id: str) -> PatientVisit | None:
    """Return the most recent visit for a patient."""
    return (
        session.query(PatientVisit)
        .filter_by(patient_id=str(patient_id))
        .order_by(PatientVisit.visit_date.desc(), PatientVisit.id.desc())
        .first()
    )


def get_previous_visit(session: Session, patient_id: str, before_visit_id: int) -> PatientVisit | None:
    """Return the visit immediately before before_visit_id."""
    current = session.query(PatientVisit).get(before_visit_id)
    if not current:
        return None
    return (
        session.query(PatientVisit)
        .filter(
            PatientVisit.patient_id == str(patient_id),
            (PatientVisit.visit_date < current.visit_date) |
            ((PatientVisit.visit_date == current.visit_date) & (PatientVisit.id < current.id))
        )
        .order_by(PatientVisit.visit_date.desc(), PatientVisit.id.desc())
        .first()
    )


# ── Prediction CRUD ───────────────────────────────────────────────────────────

def create_prediction(
    session: Session,
    patient_id: str,
    visit_id: int | None,
    prediction: int,
    probability: float | None = None,
    risk_level: str | None = None,
    model_name: str | None = None,
    model_type: str | None = "temporal",
    visit_date: datetime | None = None,
) -> Prediction:
    """Store a prediction result linked to a visit and patient."""
    pred = Prediction(
        patient_id=str(patient_id),
        visit_id=visit_id,
        visit_date=visit_date or datetime.utcnow(),
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        model_name=model_name,
        model_type=model_type,
        created_at=datetime.utcnow(),
    )
    session.add(pred)
    session.flush()
    return pred


def get_patient_predictions(session: Session, patient_id: str) -> list[Prediction]:
    """Return all predictions for a patient, ordered by visit timestamp."""
    return (
        session.query(Prediction)
        .filter_by(patient_id=str(patient_id))
        .order_by(Prediction.created_at.asc())
        .all()
    )
