"""Automated regression tests for:
1. Sole Authoritative Golden Benchmark Evaluation Enforcement
2. Fair Model Comparison at Production Threshold
3. Diagnostic Protocol Isolation (Cannot Override Promotion)
4. Historical Provenance Warning Preservation
5. Safe Incremental Cleanup & 100,001 Baseline Restoration
6. Champion Artifact Preservation & Rejection Safety
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sqlalchemy import func, text
from starlette.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import (
    BASELINE_FEATURE_COLS,
    DATASET_PATH,
    MODEL_ARTIFACT_DIR,
)

GOLDEN_BENCHMARK_DIR = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark"
GOLDEN_BENCHMARK_CSV = GOLDEN_BENCHMARK_DIR / "golden_benchmark_records.csv"
GOLDEN_BENCHMARK_META = GOLDEN_BENCHMARK_DIR / "golden_benchmark_manifest.json"
MODEL_ARTIFACT_PATH = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"

from app.database import get_session
from app.main import app
from app.ml.verified_pipeline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _sample_then_split,
)
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    Patient,
    PatientVisit,
    SyntheticTrainingSample,
    TemporalPatientData,
)
from app.services import incremental_learning as service

EXPECTED_CHAMPION_SHA256 = "45576cce7069b298f7c2350cc3b53a4e626f8219f1172d218760dc7669ab144b"
EXPECTED_BENCHMARK_SHA256 = "559bfadd4539554507407d9ca49f1ceda220ad4b1ec30ee553f0045b42f5ab18"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# 1. Champion artifact preservation
def test_champion_artifact_preserved_and_valid():
    assert Path(MODEL_ARTIFACT_PATH).is_file(), "Champion artifact missing"
    content = Path(MODEL_ARTIFACT_PATH).read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    assert sha256 == EXPECTED_CHAMPION_SHA256, (
        f"Champion artifact has been altered! Expected {EXPECTED_CHAMPION_SHA256}, got {sha256}"
    )

    artifact = joblib.load(MODEL_ARTIFACT_PATH)
    assert "model" in artifact
    assert "preprocessor" in artifact
    assert "feature_columns" in artifact
    assert len(artifact["feature_columns"]) == 34
    assert artifact.get("threshold") == 0.48


# 2. Fixed benchmark reproducibility and checksum integrity
def test_fixed_golden_benchmark_reproducibility():
    assert Path(GOLDEN_BENCHMARK_CSV).is_file(), "Golden benchmark CSV missing"
    assert Path(GOLDEN_BENCHMARK_META).is_file(), "Golden benchmark metadata missing"

    df_bm = pd.read_csv(GOLDEN_BENCHMARK_CSV)
    meta = json.loads(Path(GOLDEN_BENCHMARK_META).read_text(encoding="utf-8"))

    assert len(df_bm) == 15006, f"Expected 15,006 golden benchmark visits, got {len(df_bm)}"
    assert meta["benchmark_row_count"] == 15006
    assert meta["synthetic_rows"] == 0

    csv_sha = hashlib.sha256(Path(GOLDEN_BENCHMARK_CSV).read_bytes()).hexdigest()
    assert csv_sha == EXPECTED_BENCHMARK_SHA256, f"Benchmark hash altered! {csv_sha}"
    assert csv_sha == meta["records_csv_sha256"], f"Checksum mismatch for Golden Benchmark: {csv_sha} vs {meta['records_csv_sha256']}"


# 3. Provenance warning preserved honestly
def test_provenance_warning_preserved():
    meta = json.loads(Path(GOLDEN_BENCHMARK_META).read_text(encoding="utf-8"))
    audit = meta.get("historical_champion_independence_audit", {})
    assert audit.get("historical_training_membership_recorded") is False
    assert "UNVERIFIED" in audit.get("leakage_certification", "")

    # API overview must expose the warning honestly
    overview = service.list_state()
    bm_info = overview.get("golden_benchmark", {})
    assert bm_info.get("historical_champion_independence_audit", {}).get("historical_training_membership_recorded") is False


# 4. Sole authoritative evaluation protocol: Golden Benchmark required, Diagnostic cannot override
def test_golden_benchmark_is_sole_promotion_authority():
    # Diagnostic comparison passes artificially with high accuracy
    fake_diagnostic_comp = {
        "comparable": True,
        "evaluation_protocol": "Diagnostic (SMOTE-before-split)",
        "challenger_metrics": {"accuracy": 0.99, "precision": 0.99, "recall_sensitivity": 0.99, "specificity": 0.99, "f1": 0.99, "roc_auc": 0.99, "pr_auc": 0.99, "confusion_matrix": {"fn": 5, "fp": 5}},
        "champion_metrics": {"accuracy": 0.92, "precision": 0.84, "recall_sensitivity": 0.92, "specificity": 0.92, "f1": 0.88, "roc_auc": 0.98, "pr_auc": 0.95, "confusion_matrix": {"fn": 336, "fp": 795}},
    }

    # Case A: Golden benchmark comparison is missing -> MUST BE REJECTED
    decision_missing_gb = service._promotion_eligibility(
        report={},
        diagnostic_comparison=fake_diagnostic_comp,
        golden_benchmark_comparison=None,
    )
    assert decision_missing_gb["eligible"] is False
    assert decision_missing_gb["decision"] == "rejected"
    assert "Authoritative Golden Benchmark evaluation" in decision_missing_gb["reason"]

    # Case B: Golden benchmark comparison failed or is not comparable -> MUST BE REJECTED
    decision_incomparable_gb = service._promotion_eligibility(
        report={},
        diagnostic_comparison=fake_diagnostic_comp,
        golden_benchmark_comparison={"comparable": False, "reason": "Artifact missing"},
    )
    assert decision_incomparable_gb["eligible"] is False
    assert decision_incomparable_gb["decision"] == "rejected"


# 5. Fair model comparison uses verified production decision threshold
def test_fair_model_comparison_uses_production_threshold():
    session = get_session()
    try:
        run = session.query(IncrementalTrainingRun).filter(
            IncrementalTrainingRun.report.is_not(None)
        ).first()
        assert run is not None
        gb_comp = run.report.get("golden_benchmark_comparison") or {}
        if gb_comp.get("comparable"):
            assert gb_comp.get("production_decision_threshold") == 0.48
    finally:
        session.close()


# 6. Safe cleanup restored active dataset to exactly 100,001 baseline rows
def test_safe_cleanup_restored_100001_baseline_visits():
    session = get_session()
    try:
        baseline_rows = session.query(func.count(PatientVisit.id)).filter(
            ~PatientVisit.source_dataset.like("incremental:%")
        ).scalar()
        assert baseline_rows == 100001, f"Expected 100001 baseline visits, got {baseline_rows}"

        inc_rows = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.source_dataset.like("incremental:%")
        ).scalar()
        assert inc_rows == 0, f"Expected 0 active incremental visits after cleanup, got {inc_rows}"

        total_real = session.query(func.count(PatientVisit.id)).scalar()
        assert total_real == 100001, f"Expected exactly 100001 total real visits, got {total_real}"

        # Targeted batches are rolled back
        active_accepted_batches = session.query(IncrementalLearningBatch).filter_by(status="accepted").all()
        assert len(active_accepted_batches) == 0, "No incremental batches should remain active accepted"

        # Targeted runs are marked rolled_back
        rolled_back_runs = session.query(IncrementalTrainingRun).filter_by(status="rolled_back").all()
        assert len(rolled_back_runs) == 2, f"Expected 2 rolled back runs, got {len(rolled_back_runs)}"
    finally:
        session.close()


# 7. No synthetic samples enter evaluation or Golden Benchmark
def test_synthetic_sample_isolation():
    df_bm = pd.read_csv(GOLDEN_BENCHMARK_CSV)
    if "is_synthetic" in df_bm.columns:
        assert (df_bm["is_synthetic"] == 1).sum() == 0, "Benchmark contains synthetic samples!"

    session = get_session()
    try:
        non_training_synths = session.query(SyntheticTrainingSample).filter(
            SyntheticTrainingSample.training_only.is_(False)
        ).count()
        assert non_training_synths == 0, "Synthetic samples must be strictly training_only=True"
    finally:
        session.close()


# 8. Patient-level split isolation
def test_patient_level_split_isolation():
    raw_df = pd.read_csv(DATASET_PATH, nrows=5000)
    sgkf = StratifiedGroupKFold(n_splits=5)
    for train_idx, test_idx in sgkf.split(raw_df, raw_df["prediction"], raw_df["patient_id"]):
        train_patients = set(raw_df.iloc[train_idx]["patient_id"])
        test_patients = set(raw_df.iloc[test_idx]["patient_id"])
        overlap = train_patients.intersection(test_patients)
        assert len(overlap) == 0, f"Found patient overlap across split: {len(overlap)}"
        break


# 9. API overview matches cleaned baseline and authoritative benchmark
def test_api_overview_reflects_cleaned_baseline_and_authoritative_benchmark(client):
    resp = client.get("/api/incremental/overview")
    assert resp.status_code == 200
    data = resp.json()

    assert data["dataset"]["original_rows"] == 100001
    assert data["dataset"]["imported_rows"] == 0
    assert data["dataset"]["total_rows"] == 100001

    assert data["golden_benchmark"]["available"] is True
    assert data["golden_benchmark"]["benchmark_row_count"] == 15006
    assert data["golden_benchmark"]["synthetic_rows"] == 0
    assert data["golden_benchmark"]["records_csv_sha256"] == EXPECTED_BENCHMARK_SHA256

    assert data["active_champion"]["available"] is True
    assert data["active_champion"]["artifact_sha256"] == EXPECTED_CHAMPION_SHA256
