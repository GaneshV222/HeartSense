"""Regression coverage for numeric data stored in the PostgreSQL VARCHAR column."""

from fastapi.testclient import TestClient
from sqlalchemy import String, inspect
from sqlalchemy.orm import Session

from app.database import engine
from app.main import app
from app.models import PatientVisit
from app.services.patient_service import get_patient_visits


PATIENT_ID = "P56393"


def test_patient_visit_stress_level_mapping_matches_postgres_varchar():
    database_column = next(
        column
        for column in inspect(engine).get_columns("patient_visits", schema="public")
        if column["name"] == "stress_level"
    )
    assert isinstance(database_column["type"], String)
    assert isinstance(PatientVisit.__table__.c.stress_level.type.impl, String)


def test_get_patient_visits_converts_varchar_stress_level_to_float():
    with Session(engine) as session:
        visits = get_patient_visits(session, PATIENT_ID)

    assert visits
    assert all(
        visit.stress_level is None or isinstance(visit.stress_level, float)
        for visit in visits
    )


def test_temporal_prediction_endpoint_reads_patient_visits():
    # Avoid the app lifespan startup hook: this regression is read-only and must not
    # perform schema initialization while exercising the prediction route.
    client = TestClient(app)
    try:
        response = client.get(f"/api/patients/{PATIENT_ID}/temporal")
    finally:
        client.close()

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["patient_id"] == PATIENT_ID
    assert result["prediction"]["model_name"]
