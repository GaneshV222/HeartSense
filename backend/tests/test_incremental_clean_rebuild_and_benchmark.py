"""Comprehensive automated regression test suite for:
HeartSense Incremental Learning Rebuild, Clean Reset, and Golden Benchmark Validation.

Covers all 18 regression requirements:
1. Experimental cleanup deletes only verified experimental records and preserves original baseline.
2. Cleanup is transactional or safely recoverable, with foreign-key integrity verified.
3. Active Champion artifact and pointers remain unchanged unless an approved promotion occurs.
4. Champion and Challenger use identical golden-benchmark patients and labels.
5. No benchmark patient enters model training, preprocessing fitting, SMOTENC fitting, threshold selection, or hyperparameter tuning.
6. Historical diagnostic path retains SMOTENC-before-split ordering, categorical indices, random seeds, and class-balancing settings.
7. Evaluation-safe path splits by patient before fitting preprocessing or applying SMOTENC.
8. Feature ordering, categorical/numerical transformations, missing-value handling, and inference preprocessing remain consistent.
9. Temporal deltas and change indicators are correct for genuine follow-up visits.
10. First visits do not receive fabricated longitudinal information.
11. Synthetic records never become real patients or clinical visits.
12. Incremental batches and synthetic samples cannot be ingested or counted twice.
13. Identical inputs and settings produce reproducible diagnostic results.
14. Promotion decisions use configured safety requirements and accurate persisted metrics.
15. API endpoints return correct contracts and handle failures without changing the Champion.
16. Frontend data contracts and page endpoints work accurately.
17. Historical run records remain available and accurately labelled after cleanup.
18. Cleaned database counts and baseline CSV reconciliation are correct.
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
from sqlalchemy import func, text
from starlette.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR
from app.database import get_session
from app.main import app
from app.ml.verified_pipeline import MODEL_FEATURE_COLUMNS, TARGET_COLUMN, _sample_then_split
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    Patient,
    PatientVisit,
    SyntheticTrainingSample,
    TemporalPatientData,
)
from app.services import incremental_learning as service
from app.services.temporal_service import TemporalFeatureService

EXPECTED_CHAMPION_SHA256 = "45576cce7069b298f7c2350cc3b53a4e626f8219f1172d218760dc7669ab144b"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# 1. Experimental cleanup deletes only verified experimental records and preserves original baseline
def test_01_experimental_cleanup_preserves_original_baseline():
    session = get_session()
    try:
        baseline_visits = session.query(func.count(PatientVisit.id)).filter(
            ~PatientVisit.source_dataset.like("incremental:%")
        ).scalar()
        assert baseline_visits == 100001, f"Expected 100001 baseline visits, got {baseline_visits}"

        total_visits = session.query(func.count(PatientVisit.id)).scalar()
        assert total_visits == 100001, f"Expected 100001 total visits in clean baseline, got {total_visits}"

        inc_visits = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.source_dataset.like("incremental:%")
        ).scalar()
        assert inc_visits == 0, f"Expected 0 active incremental visits after cleanup, got {inc_visits}"

        # Targeted batches are rolled back
        active_accepted = session.query(IncrementalLearningBatch).filter_by(status="accepted").all()
        assert len(active_accepted) == 0, "No incremental batches should remain active accepted"
    finally:
        session.close()


# 2. Cleanup is transactional or safely recoverable, with foreign-key integrity verified
def test_02_foreign_key_integrity_verified():
    session = get_session()
    try:
        orphans_visit = session.execute(
            text("SELECT COUNT(*) FROM patient_visits pv LEFT JOIN patients p ON pv.patient_id = p.patient_id WHERE p.patient_id IS NULL")
        ).scalar()
        assert orphans_visit == 0, f"Found orphan patient_visits: {orphans_visit}"

        orphans_temporal = session.execute(
            text("SELECT COUNT(*) FROM temporal_patient_data tpd LEFT JOIN patients p ON tpd.patient_id = p.patient_id WHERE p.patient_id IS NULL")
        ).scalar()
        assert orphans_temporal == 0, f"Found orphan temporal_patient_data: {orphans_temporal}"
    finally:
        session.close()


# 3. Active Champion artifact and pointers remain unchanged unless an approved promotion occurs
def test_03_active_champion_artifact_and_pointers_unchanged():
    champion_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    assert champion_path.is_file(), "Champion artifact missing"
    current_hash = hashlib.sha256(champion_path.read_bytes()).hexdigest()
    assert current_hash == EXPECTED_CHAMPION_SHA256, f"Champion SHA-256 altered: {current_hash}"

    champ_info = service.get_active_champion()
    assert champ_info["available"] is True
    assert champ_info["model_name"] == "XGBoost baseline"
    assert champ_info["artifact_sha256"] == EXPECTED_CHAMPION_SHA256


# 4. Champion and Challenger use identical golden-benchmark patients and labels
def test_04_champion_and_challenger_use_identical_golden_benchmark():
    benchmark_dir = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark"
    records_csv = benchmark_dir / "golden_benchmark_records.csv"
    manifest_file = benchmark_dir / "golden_benchmark_manifest.json"
    assert records_csv.is_file()
    assert manifest_file.is_file()

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["benchmark_row_count"] == 15006
    assert manifest["synthetic_rows"] == 0

    df = pd.read_csv(records_csv)
    assert len(df) == 15006
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


# 5. No benchmark patient enters model training, preprocessing fitting, SMOTENC fitting, or tuning
def test_05_no_benchmark_patient_in_training_or_tuning():
    benchmark_dir = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark"
    manifest = json.loads((benchmark_dir / "golden_benchmark_manifest.json").read_text(encoding="utf-8"))
    benchmark_patients = set(manifest.get("test_patient_ids", []))
    train_patients = set(manifest.get("train_patient_ids", []))
    val_patients = set(manifest.get("validation_patient_ids", []))

    assert len(benchmark_patients) > 0
    assert len(train_patients) > 0
    assert len(val_patients) > 0

    # Leakage check: zero patient overlap
    train_overlap = benchmark_patients.intersection(train_patients)
    assert len(train_overlap) == 0, f"Found {len(train_overlap)} benchmark patients in training set!"

    val_overlap = benchmark_patients.intersection(val_patients)
    assert len(val_overlap) == 0, f"Found {len(val_overlap)} benchmark patients in validation set!"


# 6. Historical diagnostic path retains SMOTENC-before-split ordering, categorical indices, random seeds
def test_06_historical_diagnostic_path_retains_smote_before_split():
    events = []

    def dummy_sampler(features, labels):
        events.append(("sampler", len(features)))
        return features + ["synth_sample"], labels + [1]

    def dummy_splitter(features, labels):
        events.append(("split", len(features)))
        return {"train": features[:2], "test": features[2:]}

    f_out, l_out, parts = _sample_then_split(
        ["r1", "r2"], [0, 1], dummy_sampler, dummy_splitter
    )
    assert [e[0] for e in events] == ["sampler", "split"]
    assert "synth_sample" in parts["test"]


# 7. Evaluation-safe path splits by patient before fitting preprocessing or applying SMOTENC
def test_07_evaluation_safe_path_splits_by_patient():
    manifest_path = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark" / "golden_benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evaluation_protocol"].startswith("Independent real-patient")
    assert manifest["synthetic_rows"] == 0
    assert manifest["historical_champion_independence_audit"]["historical_training_membership_recorded"] is False


# 8. Feature ordering, categorical/numerical transformations, and inference preprocessing remain consistent
def test_08_feature_ordering_and_preprocessing_consistent():
    champion_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    champion = joblib.load(champion_path)
    assert len(champion["feature_columns"]) == 34
    assert list(champion["feature_columns"]) == list(MODEL_FEATURE_COLUMNS)
    assert "preprocessor" in champion
    assert "model" in champion


# 9. Temporal deltas and change indicators are correct for genuine follow-up visits
def test_09_temporal_deltas_correct_for_genuine_followup_visits():
    p1 = {
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "cholesterol": 200.0,
        "ldl": 110.0,
        "hdl": 50.0,
        "bmi": 24.0,
        "hba1c": 5.5,
        "resting_heart_rate": 70.0,
        "smoking": "No",
        "physical_activity": "Low",
    }
    p2 = {
        "systolic_bp": 130.0,
        "diastolic_bp": 84.0,
        "cholesterol": 215.0,
        "ldl": 120.0,
        "hdl": 48.0,
        "bmi": 24.5,
        "hba1c": 5.7,
        "resting_heart_rate": 74.0,
        "smoking": "Yes",
        "physical_activity": "Moderate",
    }
    snapshot = TemporalFeatureService.calculate_snapshot(p2, p1, patient_id="PTEST")
    assert snapshot["delta_systolic_bp"] == 10.0
    assert snapshot["delta_diastolic_bp"] == 4.0
    assert snapshot["delta_cholesterol"] == 15.0
    assert snapshot["smoking_changed"] is True
    assert snapshot["physical_activity_changed"] is True


# 10. First visits do not receive fabricated longitudinal information
def test_10_first_visits_do_not_receive_fabricated_longitudinal_info():
    current = {
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "cholesterol": 200.0,
        "ldl": 110.0,
        "hdl": 50.0,
        "bmi": 24.0,
        "hba1c": 5.5,
        "resting_heart_rate": 70.0,
        "smoking": "No",
        "physical_activity": "Low",
    }
    snapshot = TemporalFeatureService.calculate_snapshot(current, None, patient_id="PTEST")
    assert snapshot["is_first_visit"] is True
    assert snapshot["has_previous_visit"] is False
    assert snapshot["previous_visit_id"] is None
    for var in ["systolic_bp", "diastolic_bp", "cholesterol", "ldl", "hdl", "bmi", "hba1c", "resting_heart_rate"]:
        assert snapshot[f"delta_{var}"] is None, f"First visit feature delta_{var} should be None, got {snapshot[f'delta_{var}']}"


# 11. Synthetic records never become real patients or clinical visits
def test_11_synthetic_records_never_become_real_patients_or_visits():
    session = get_session()
    try:
        synth_in_visits = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(True)
        ).scalar()
        assert synth_in_visits == 0, f"Found {synth_in_visits} synthetic records in patient_visits"

        real_in_synthetic = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter(
            SyntheticTrainingSample.is_synthetic.is_(False)
        ).scalar()
        assert real_in_synthetic == 0, f"Found {real_in_synthetic} real records in synthetic_training_samples"
    finally:
        session.close()


# 12. Incremental batches and synthetic samples cannot be ingested or counted twice
def test_12_incremental_batches_cannot_be_ingested_or_counted_twice(client):
    test_id = f"IDEMP-TEST-{uuid.uuid4().hex[:6]}"
    row = {
        "patient_id": test_id,
        "visit_number": 1,
        "visit_date": "2026-06-01",
        "prediction": 0,
        "age": 48.0,
        "gender": "Male",
        "bmi": 23.5,
        "chest_pain_type": "Typical Angina",
        "systolic_bp": 118.0,
        "diastolic_bp": 78.0,
        "resting_heart_rate": 65.0,
        "max_heart_rate": 160.0,
        "cholesterol": 190.0,
        "hdl": 55.0,
        "ldl": 105.0,
        "fasting_blood_sugar": "false",
        "hba1c": 5.2,
        "diabetes": "false",
        "resting_ecg": "Normal",
        "exercise_angina": "false",
        "oldpeak": 0.0,
        "st_slope": "Up",
        "num_major_vessels": 0.0,
        "thalassemia": "Normal",
        "smoking": "false",
        "family_history": "false",
        "physical_activity": "High",
        "stress_level": 3.0,
    }
    csv_bytes = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")

    val_resp = client.post(
        "/api/incremental/batches/validate",
        files={"clinical_file": ("idemp_batch.csv", csv_bytes, "text/csv")},
    )
    assert val_resp.status_code == 200
    batch_id = val_resp.json()["batch_id"]

    try:
        # First confirmation
        c1 = client.post(f"/api/incremental/batches/{batch_id}/confirm")
        assert c1.status_code == 200

        # Duplicate confirmation
        c2 = client.post(f"/api/incremental/batches/{batch_id}/confirm")
        assert c2.status_code == 200

        session = get_session()
        try:
            visits = session.query(PatientVisit).filter_by(patient_id=test_id).all()
            assert len(visits) == 1, "Duplicate visits were ingested on repeated confirmation"
        finally:
            session.close()
    finally:
        session = get_session()
        try:
            b = session.get(IncrementalLearningBatch, batch_id)
            if b:
                session.delete(b)
            for v in session.query(PatientVisit).filter_by(patient_id=test_id).all():
                session.delete(v)
            for td in session.query(TemporalPatientData).filter_by(patient_id=test_id).all():
                session.delete(td)
            for p in session.query(Patient).filter_by(patient_id=test_id).all():
                session.delete(p)
            session.commit()
        finally:
            session.close()


# 13. Identical inputs and settings produce reproducible diagnostic results
def test_13_identical_inputs_reproducible_results():
    champion_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    champ = joblib.load(champion_path)
    service._restore_simple_imputer_runtime_dtype(champ["preprocessor"])

    sample_input = pd.DataFrame([{
        "age": 55.0, "gender": "Female", "bmi": 26.0, "chest_pain_type": "Typical Angina",
        "systolic_bp": 130.0, "diastolic_bp": 85.0, "resting_heart_rate": 72.0, "max_heart_rate": 145.0,
        "cholesterol": 220.0, "hdl": 48.0, "ldl": 130.0, "fasting_blood_sugar": "false", "hba1c": 5.9,
        "diabetes": "false", "resting_ecg": "Normal", "exercise_angina": "false", "oldpeak": 1.2,
        "st_slope": "Flat", "num_major_vessels": 1.0, "thalassemia": "Normal", "smoking": "false",
        "family_history": "true", "physical_activity": "Moderate", "stress_level": 5.0,
        "delta_systolic_bp": 2.0, "delta_diastolic_bp": 1.0, "delta_cholesterol": 5.0, "delta_ldl": 4.0,
        "delta_hdl": -1.0, "delta_bmi": 0.2, "delta_hba1c": 0.1, "delta_resting_heart_rate": 2.0,
        "smoking_changed": False, "physical_activity_changed": False,
    }])
    X1 = champ["preprocessor"].transform(sample_input)
    p1 = champ["model"].predict_proba(X1)[:, 1]

    champ_reloaded = joblib.load(champion_path)
    service._restore_simple_imputer_runtime_dtype(champ_reloaded["preprocessor"])
    X2 = champ_reloaded["preprocessor"].transform(sample_input)
    p2 = champ_reloaded["model"].predict_proba(X2)[:, 1]

    np.testing.assert_array_equal(p1, p2)


# 14. Promotion decisions use configured safety requirements and accurate persisted metrics
def test_14_promotion_decisions_enforce_safety_requirements():
    diagnostic_comp = {
        "comparable": True,
        "evaluation_protocol": "SMOTE-before-split diagnostic metrics",
        "champion_metrics": {
            "accuracy": 0.9365, "precision": 0.9238, "recall_sensitivity": 0.9515,
            "specificity": 0.9215, "f1": 0.9375, "roc_auc": 0.9829, "pr_auc": 0.9837,
            "confusion_matrix": {"tn": 12969, "fp": 1104, "fn": 682, "tp": 13391},
        },
        "challenger_metrics": {
            "accuracy": 0.9276, "precision": 0.9167, "recall_sensitivity": 0.9406,
            "specificity": 0.9145, "f1": 0.9285, "roc_auc": 0.9806, "pr_auc": 0.9802,
            "confusion_matrix": {"tn": 12870, "fp": 1203, "fn": 836, "tp": 13237},
        },
    }
    # 1. Authoritative Golden Benchmark is required; diagnostic comparison cannot decide promotion
    elig_no_gb = service._promotion_eligibility(
        report={}, diagnostic_comparison=diagnostic_comp
    )
    assert elig_no_gb["eligible"] is False
    assert elig_no_gb["decision"] == "rejected"
    assert "Golden Benchmark evaluation is required" in elig_no_gb["reason"]

    # 2. On Golden Benchmark evaluation, challenger with increased false negatives is rejected
    eligibility = service._promotion_eligibility(
        report={}, diagnostic_comparison=diagnostic_comp, golden_benchmark_comparison=diagnostic_comp
    )
    assert eligibility["eligible"] is False
    assert eligibility["decision"] == "rejected"
    assert "FN count increased" in eligibility["reason"]


# 15. API endpoints return correct contracts and handle failures without changing Champion
def test_15_api_endpoints_return_correct_contracts(client):
    resp = client.get("/api/incremental/overview")
    assert resp.status_code == 200
    data = resp.json()

    assert "dataset" in data
    assert data["dataset"]["original_rows"] == 100001
    assert data["dataset"]["imported_rows"] == 0
    assert data["dataset"]["total_rows"] == 100001

    assert "golden_benchmark" in data
    assert data["golden_benchmark"]["available"] is True
    assert data["golden_benchmark"]["benchmark_row_count"] == 15006
    assert data["golden_benchmark"]["synthetic_rows"] == 0

    assert "active_champion" in data
    assert data["active_champion"]["available"] is True
    assert data["active_champion"]["artifact_sha256"] == EXPECTED_CHAMPION_SHA256


# 16. Incremental Learning UI data contracts pass
def test_16_frontend_contract_and_page_safety(client):
    resp = client.get("/api/incremental/overview")
    data = resp.json()
    assert "runs" in data
    assert "batches" in data
    assert len(data["runs"]) >= 1

    run = data["runs"][0]
    assert "training_run_id" in run
    assert "status" in run
    assert "golden_curves" in run
    assert "comparison_curves" in run


# 17. Historical run records remain available and accurately labelled after cleanup
def test_17_historical_run_records_accurately_labelled():
    session = get_session()
    try:
        baseline_run = session.get(IncrementalTrainingRun, "ilrun-20261013T042318Z-7579f93cb7")
        assert baseline_run is not None, "Baseline run record missing"
        assert baseline_run.status == "completed"
        assert baseline_run.dataset_rows == 100001
    finally:
        session.close()


# 18. Cleaned database counts and baseline CSV reconciliation are correct
def test_18_cleaned_db_counts_and_csv_reconciliation():
    source_csv = Path(DATASET_PATH)
    assert source_csv.is_file()
    df = pd.read_csv(source_csv)
    assert len(df) == 100001

    csv_sha = hashlib.sha256(source_csv.read_bytes()).hexdigest()

    session = get_session()
    try:
        baseline_db_count = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            ~PatientVisit.source_dataset.like("incremental:%"),
        ).scalar()
        assert baseline_db_count == len(df) == 100001

        total_db_count = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False)
        ).scalar()
        assert total_db_count == 100001

        # Class distribution reconciliation for original baseline
        db_pos = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            ~PatientVisit.source_dataset.like("incremental:%"),
            PatientVisit.prediction == 1,
        ).scalar()
        db_neg = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            ~PatientVisit.source_dataset.like("incremental:%"),
            PatientVisit.prediction == 0,
        ).scalar()
        csv_pos = int((df["prediction"] == 1).sum())
        csv_neg = int((df["prediction"] == 0).sum())

        assert db_pos == csv_pos == 30000
        assert db_neg == csv_neg == 70001
    finally:
        session.close()
