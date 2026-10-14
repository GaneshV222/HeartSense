from pathlib import Path
import sys
import uuid

import pandas as pd
import pytest
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import engine
from app.models import IncrementalLearningBatch, PatientVisit, TemporalPatientData
from app.services import incremental_learning as incremental
from app.config import BASELINE_FEATURE_COLS


@pytest.mark.skipif(engine.dialect.name != "postgresql", reason="Requires PostgreSQL advisory locks and JSON")
def test_confirmed_batch_imports_genuine_visit_once_and_updates_temporal_history(
    monkeypatch, tmp_path
):
    patient_id = f"IL-TEST-{uuid.uuid4().hex}"
    batch_id = f"il-test-{uuid.uuid4().hex}"
    row = {column: 1 for column in BASELINE_FEATURE_COLS}
    row.update({
        "patient_id": patient_id,
        "visit_number": 1,
        "visit_date": "2026-01-10",
        "prediction": 1,
        "age": 54,
        "bmi": 25.0,
        "systolic_bp": 128.0,
        "diastolic_bp": 80.0,
        "resting_heart_rate": 70.0,
        "max_heart_rate": 150.0,
        "cholesterol": 210.0,
        "hdl": 50.0,
        "ldl": 120.0,
        "hba1c": 6.0,
        "oldpeak": 1.0,
        "num_major_vessels": 1.0,
        "stress_level": 5.0,
        "gender": "Female",
        "chest_pain_type": "Asymptomatic",
        "fasting_blood_sugar": "false",
        "diabetes": "false",
        "resting_ecg": "Normal",
        "exercise_angina": "false",
        "st_slope": "Flat",
        "thalassemia": "Normal",
        "smoking": "false",
        "family_history": "false",
        "physical_activity": "Low",
        "source_row_number": 2,
    })
    source_path = tmp_path / "accepted.csv"
    pd.DataFrame([row]).to_csv(source_path, index=False)
    source_hash = incremental._file_sha(source_path.read_bytes())

    with engine.connect() as connection:
        outer = connection.begin()
        session = Session(bind=connection, expire_on_commit=False)
        session.commit = session.flush
        session.close = lambda: None
        monkeypatch.setattr(incremental, "get_session", lambda: session)
        session.add(IncrementalLearningBatch(
            batch_id=batch_id,
            idempotency_key=uuid.uuid4().hex * 2,
            source_filename="accepted.csv",
            source_sha256="a" * 64,
            source_path=str(source_path),
            status="validated",
            validation_report={"accepted_rows_sha256": source_hash},
            accepted_rows=1,
        ))
        session.flush()
        try:
            first = incremental.confirm_batch(batch_id)
            assert first["status"] == "accepted"
            visits = session.query(PatientVisit).filter_by(patient_id=patient_id).all()
            assert len(visits) == 1
            assert visits[0].prediction == 1
            assert visits[0].is_synthetic is False
            temporal = session.query(TemporalPatientData).filter_by(patient_id=patient_id).one()
            assert temporal.current_visit_id == visits[0].id
            assert temporal.previous_visit_id is None

            second = incremental.confirm_batch(batch_id)
            assert second["status"] == "accepted"
            assert session.query(PatientVisit).filter_by(patient_id=patient_id).count() == 1
            assert session.query(TemporalPatientData).filter_by(patient_id=patient_id).count() == 1

            invalid_patient = f"IL-TEST-{uuid.uuid4().hex}"
            invalid_batch_id = f"il-test-{uuid.uuid4().hex}"
            invalid_row = dict(row, patient_id=invalid_patient)
            invalid_path = tmp_path / "partially-invalid.csv"
            pd.DataFrame([invalid_row]).to_csv(invalid_path, index=False)
            session.add(IncrementalLearningBatch(
                batch_id=invalid_batch_id,
                idempotency_key=uuid.uuid4().hex * 2,
                source_filename="partially-invalid.csv",
                source_sha256="b" * 64,
                source_path=str(invalid_path),
                status="validated",
                validation_report={"accepted_rows_sha256": incremental._file_sha(invalid_path.read_bytes())},
                accepted_rows=1,
                rejected_rows=1,
            ))
            session.flush()
            with pytest.raises(ValueError, match="cannot be partially imported"):
                incremental.confirm_batch(invalid_batch_id)
            assert session.query(PatientVisit).filter_by(patient_id=invalid_patient).count() == 0
        finally:
            outer.rollback()
            session.close()

    with engine.connect() as connection:
        assert connection.execute(
            PatientVisit.__table__.select().where(PatientVisit.patient_id == patient_id)
        ).first() is None
