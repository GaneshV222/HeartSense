"""
Tests for database operations (patient, visit, prediction CRUD).

These tests use an in-memory SQLite database to avoid requiring PostgreSQL.
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import Base, Patient, PatientVisit, Prediction
from src.services.patient_service import (
    create_patient,
    get_patient_by_code,
    get_all_patients,
    create_visit,
    get_patient_visits,
    get_latest_visit,
    get_previous_visit,
    create_prediction,
    get_patient_predictions,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestPatientCRUD:
    """Tests for patient creation and retrieval."""

    def test_create_patient(self, db_session):
        patient = create_patient(db_session, "P001", name="Test Patient")
        assert patient.patient_code == "P001"
        assert patient.name == "Test Patient"
        assert patient.id is not None

    def test_create_patient_idempotent(self, db_session):
        p1 = create_patient(db_session, "P002")
        p2 = create_patient(db_session, "P002")
        assert p1.id == p2.id  # Same patient returned

    def test_get_patient_by_code(self, db_session):
        create_patient(db_session, "P003")
        patient = get_patient_by_code(db_session, "P003")
        assert patient is not None
        assert patient.patient_code == "P003"

    def test_get_patient_not_found(self, db_session):
        patient = get_patient_by_code(db_session, "NONEXISTENT")
        assert patient is None

    def test_get_all_patients(self, db_session):
        create_patient(db_session, "PA")
        create_patient(db_session, "PB")
        all_p = get_all_patients(db_session)
        assert len(all_p) == 2


class TestVisitCRUD:
    """Tests for visit creation and retrieval."""

    def test_create_visit(self, db_session):
        patient = create_patient(db_session, "V001")
        visit = create_visit(
            db_session,
            patient.id,
            datetime(2026, 1, 10),
            {"age": 55, "trestbps": 130, "chol": 200},
        )
        assert visit.patient_id == patient.id
        assert visit.age == 55
        assert visit.trestbps == 130

    def test_get_patient_visits_ordered(self, db_session):
        patient = create_patient(db_session, "V002")
        create_visit(db_session, patient.id, datetime(2026, 6, 1), {"age": 50})
        create_visit(db_session, patient.id, datetime(2026, 1, 1), {"age": 50})
        create_visit(db_session, patient.id, datetime(2026, 3, 1), {"age": 50})

        visits = get_patient_visits(db_session, patient.id)
        assert len(visits) == 3
        # Should be chronological (ascending)
        assert visits[0].visit_timestamp < visits[1].visit_timestamp < visits[2].visit_timestamp

    def test_get_latest_visit(self, db_session):
        patient = create_patient(db_session, "V003")
        create_visit(db_session, patient.id, datetime(2026, 1, 1), {"age": 50})
        create_visit(db_session, patient.id, datetime(2026, 6, 1), {"age": 51})

        latest = get_latest_visit(db_session, patient.id)
        assert latest.visit_timestamp == datetime(2026, 6, 1)
        assert latest.age == 51

    def test_get_previous_visit(self, db_session):
        patient = create_patient(db_session, "V004")
        v1 = create_visit(db_session, patient.id, datetime(2026, 1, 1), {"age": 50})
        v2 = create_visit(db_session, patient.id, datetime(2026, 6, 1), {"age": 51})

        prev = get_previous_visit(db_session, patient.id, v2.id)
        assert prev is not None
        assert prev.id == v1.id

    def test_visit_to_feature_dict(self, db_session):
        patient = create_patient(db_session, "V005")
        visit = create_visit(
            db_session,
            patient.id,
            datetime(2026, 1, 1),
            {"age": 55, "sex": 1, "cp": 2, "trestbps": 140},
        )
        feats = visit.to_feature_dict()
        assert feats["age"] == 55
        assert feats["sex"] == 1
        assert feats["cp"] == 2
        assert feats["trestbps"] == 140


class TestPredictionCRUD:
    """Tests for prediction storage and retrieval."""

    def test_create_prediction(self, db_session):
        patient = create_patient(db_session, "PR001")
        visit = create_visit(db_session, patient.id, datetime(2026, 1, 1), {"age": 50})
        pred = create_prediction(
            db_session,
            patient_id=patient.id,
            visit_id=visit.id,
            prediction=1,
            probability=0.85,
            model_name="XGBoost",
        )
        assert pred.prediction == 1
        assert pred.probability == 0.85
        assert pred.model_name == "XGBoost"

    def test_get_patient_predictions(self, db_session):
        patient = create_patient(db_session, "PR002")
        v1 = create_visit(db_session, patient.id, datetime(2026, 1, 1), {"age": 50})
        v2 = create_visit(db_session, patient.id, datetime(2026, 6, 1), {"age": 51})

        create_prediction(db_session, patient.id, v1.id, 0, 0.3, "XGBoost")
        create_prediction(db_session, patient.id, v2.id, 1, 0.8, "XGBoost")

        preds = get_patient_predictions(db_session, patient.id)
        assert len(preds) == 2
