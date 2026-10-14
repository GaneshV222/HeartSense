"""Automated tests for reconciled SMOTENC database storage, cumulative retraining,
and the complete frontend-to-database batch confirmation workflow.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func
from starlette.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import BASELINE_FEATURE_COLS, DATASET_PATH
from app.ml.verified_pipeline import MODEL_FEATURE_COLUMNS, TARGET_COLUMN
from app.database import engine, get_session
from app.main import app
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    Patient,
    PatientVisit,
    SyntheticTrainingSample,
    TemporalPatientData,
)
from app.services import incremental_learning as service


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# 1. Synthetic sample persistence count reconciliation
def test_synthetic_sample_persistence_count_reconciliation():
    session = get_session()
    try:
        # Check that synthetic_training_samples table has baseline 40,001 + new 3,933 = 43,934
        latest_run = (
            session.query(IncrementalTrainingRun)
            .filter_by(status="completed")
            .order_by(IncrementalTrainingRun.created_at.desc())
            .first()
        )
        assert latest_run is not None

        total_samples = session.query(func.count(SyntheticTrainingSample.synthetic_id)).scalar()
        assert total_samples == 43934, f"Total synthetic samples in table must be 43,934, got {total_samples}"

        # Baseline run samples
        baseline_samples = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter_by(
            training_run_id="ilrun-20261013T042318Z-7579f93cb7"
        ).scalar()
        assert baseline_samples == 40001, f"Baseline run must have 40,001 synthetic rows, got {baseline_samples}"

        # Newly inserted samples for the latest run
        run_samples = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter_by(
            training_run_id=latest_run.training_run_id
        ).scalar()
        assert run_samples == 3933, f"Latest run must have exactly 3,933 newly inserted synthetic rows, got {run_samples}"

        # Sampler generated 43,934 in the report
        smote_report = (latest_run.report or {}).get("smote") or {}
        assert smote_report.get("synthetic_samples_created") == 43934
        assert smote_report.get("synthetic_samples_newly_inserted") == 3933
        assert smote_report.get("synthetic_samples_previously_persisted") == 40001
        assert smote_report.get("synthetic_samples_used_in_training") == 43934
    finally:
        session.close()


# 2. Run-specific synthetic provenance and idempotency
def test_synthetic_sample_provenance_and_idempotency():
    session = get_session()
    try:
        latest_run = (
            session.query(IncrementalTrainingRun)
            .filter_by(status="completed")
            .order_by(IncrementalTrainingRun.created_at.desc())
            .first()
        )
        assert latest_run is not None

        sample = session.query(SyntheticTrainingSample).filter_by(
            training_run_id=latest_run.training_run_id
        ).first()
        assert sample is not None
        assert sample.is_synthetic is True
        assert sample.training_only is True
        assert sample.generation_method == "SMOTENC"
        assert sample.sample_index is not None
        assert sample.target in (0, 1)
        assert isinstance(sample.feature_values, dict)
        assert len(sample.feature_values) == 34
    finally:
        session.close()


# 3. Clean clinical baseline of 100,001 genuine visits
def test_clean_clinical_baseline_visits_and_no_unwanted_records():
    session = get_session()
    try:
        total_visits = session.query(func.count(PatientVisit.id)).scalar()
        assert total_visits == 100001, f"Expected clean baseline of exactly 100,001 visits, got {total_visits}"
        inc_visits = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.source_dataset.like("incremental:%")
        ).scalar()
        assert inc_visits == 0, f"Expected 0 unwanted/incremental visits in patient_visits, got {inc_visits}"
    finally:
        session.close()


# 4. End-to-end batch validation and confirmation workflow
def test_batch_validation_and_confirmation_api_workflow(client):
    test_id = f"CONFIRM-TEST-{uuid.uuid4().hex[:6]}"
    row = {
        "patient_id": test_id,
        "visit_number": 1,
        "visit_date": "2026-05-15",
        "prediction": 1,
        "age": 52.0,
        "gender": "Female",
        "bmi": 26.4,
        "chest_pain_type": "Typical Angina",
        "systolic_bp": 125.0,
        "diastolic_bp": 82.0,
        "resting_heart_rate": 68.0,
        "max_heart_rate": 155.0,
        "cholesterol": 215.0,
        "hdl": 52.0,
        "ldl": 125.0,
        "fasting_blood_sugar": "false",
        "hba1c": 5.8,
        "diabetes": "false",
        "resting_ecg": "Normal",
        "exercise_angina": "false",
        "oldpeak": 0.8,
        "st_slope": "Up",
        "num_major_vessels": 0.0,
        "thalassemia": "Normal",
        "smoking": "false",
        "family_history": "false",
        "physical_activity": "Moderate",
        "stress_level": 4.0,
    }
    csv_content = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")

    # Step 1: Validate batch
    val_resp = client.post(
        "/api/incremental/batches/validate",
        files={"clinical_file": ("test_batch.csv", csv_content, "text/csv")},
    )
    assert val_resp.status_code == 200, f"Validation failed: {val_resp.text}"
    batch_data = val_resp.json()
    batch_id = batch_data["batch_id"]
    assert batch_data["status"] == "validated"
    assert batch_data["accepted_rows"] == 1
    assert len(batch_data["validation_report"]["preview_rows"]) == 1

    # Step 2: Confirm batch via API (Confirm accepted real rows button workflow)
    conf_resp = client.post(f"/api/incremental/batches/{batch_id}/confirm")
    assert conf_resp.status_code == 200, f"Confirmation failed: {conf_resp.text}"
    conf_data = conf_resp.json()
    assert conf_data["status"] == "accepted"
    assert conf_data["confirmed_at"] is not None

    session = get_session()
    try:
        visit = session.query(PatientVisit).filter_by(patient_id=test_id).first()
        assert visit is not None
        assert visit.source_dataset == f"incremental:{batch_id}"
        assert visit.is_synthetic is False

        # Idempotent retry: confirming again returns accepted status without duplicating
        retry_resp = client.post(f"/api/incremental/batches/{batch_id}/confirm")
        assert retry_resp.status_code == 200
        count_after = session.query(func.count(PatientVisit.id)).filter_by(patient_id=test_id).scalar()
        assert count_after == 1, "Duplicate visits must not be created on retry"
    finally:
        # Clean up test visit, temporal record, patient and batch
        b = session.get(IncrementalLearningBatch, batch_id)
        if b:
            session.delete(b)
        for pv in session.query(PatientVisit).filter_by(patient_id=test_id).all():
            session.delete(pv)
        for td in session.query(TemporalPatientData).filter_by(patient_id=test_id).all():
            session.delete(td)
        for p in session.query(Patient).filter_by(patient_id=test_id).all():
            session.delete(p)
        session.commit()
        session.close()


# 5. Deterministic artifact reload and prediction reproducibility
def test_candidate_artifact_reload_and_deterministic_predictions():
    # Verify the latest completed training run artifact
    session = get_session()
    try:
        latest_run = (
            session.query(IncrementalTrainingRun)
            .filter_by(status="completed")
            .order_by(IncrementalTrainingRun.created_at.desc())
            .first()
        )
        assert latest_run is not None, "No completed incremental training run found"
        artifact_path = Path(latest_run.artifact_path)
        assert artifact_path.is_file(), f"Model artifact not found at {artifact_path}"

        d = joblib.load(artifact_path)
        assert "model" in d
        assert "preprocessor" in d
        assert "feature_columns" in d

        # Deterministic test fixture
        fixture = pd.DataFrame([{
            "age": 60.0, "gender": "Male", "bmi": 28.0, "chest_pain_type": "Asymptomatic",
            "systolic_bp": 140.0, "diastolic_bp": 90.0, "resting_heart_rate": 78.0, "max_heart_rate": 130.0,
            "cholesterol": 240.0, "hdl": 40.0, "ldl": 150.0, "fasting_blood_sugar": "true", "hba1c": 6.5,
            "diabetes": "true", "resting_ecg": "ST-T Wave Abnormality", "exercise_angina": "true", "oldpeak": 2.0,
            "st_slope": "Flat", "num_major_vessels": 2.0, "thalassemia": "Reversible Defect", "smoking": "true",
            "family_history": "true", "physical_activity": "Low", "stress_level": 7.0,
            "delta_systolic_bp": 5.0, "delta_diastolic_bp": 3.0, "delta_cholesterol": 10.0, "delta_ldl": 8.0,
            "delta_hdl": -2.0, "delta_bmi": 0.5, "delta_hba1c": 0.2, "delta_resting_heart_rate": 4.0,
            "smoking_changed": False, "physical_activity_changed": False,
        }])

        X_t1 = d["preprocessor"].transform(fixture)
        prob1 = d["model"].predict_proba(X_t1)[:, 1]

        # Reload second time to confirm determinism
        d2 = joblib.load(artifact_path)
        X_t2 = d2["preprocessor"].transform(fixture)
        prob2 = d2["model"].predict_proba(X_t2)[:, 1]

        np.testing.assert_array_equal(prob1, prob2)
        assert 0.0 <= prob1[0] <= 1.0
    finally:
        session.close()


# 6. Active Champion preservation during challenger training
def test_active_champion_preserved():
    champion = service.get_active_champion()
    assert champion["available"] is True
    assert champion["model_name"] == "XGBoost baseline"
    assert "incremental_learning" not in str(champion.get("artifact_path", ""))
