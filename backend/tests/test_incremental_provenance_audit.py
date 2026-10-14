"""Comprehensive automated test suite for incremental learning provenance,
database counts, model metric forensic audit, and historical run integrity.
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
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import func
from starlette.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR
from app.database import get_session
from app.main import app
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    PatientVisit,
    SyntheticTrainingSample,
)
from app.services import incremental_learning as service


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# 1. Original source row-count provenance
def test_original_source_row_count_provenance():
    source_file = Path(DATASET_PATH)
    assert source_file.is_file(), f"Authoritative source CSV not found at {source_file}"
    content = source_file.read_bytes()
    expected_sha = hashlib.sha256(content).hexdigest()

    df = pd.read_csv(source_file)
    assert len(df) == 100001, f"Authoritative source CSV should have 100001 rows, got {len(df)}"

    session = get_session()
    try:
        orig_db_count = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            ~PatientVisit.source_dataset.like("incremental:%"),
        ).scalar()
        assert orig_db_count == 100001, f"PostgreSQL original real visits should be 100001, got {orig_db_count}"
    finally:
        session.close()

    overview = service.list_state()
    assert overview["original_rows"] == 100001
    assert overview["dataset"]["original_rows"] == 100001
    assert overview["original_dataset"]["sha256"] == expected_sha
    assert overview["dataset"]["source_file"]["rows"] == 100001


# 2. Accepted incremental real-row counts
def test_accepted_incremental_real_row_counts():
    session = get_session()
    try:
        inc_visits_count = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            PatientVisit.source_dataset.like("incremental:%"),
        ).scalar()

        batches = session.query(IncrementalLearningBatch).filter_by(status="accepted").all()
        batch_sum = sum(b.accepted_rows for b in batches)

        assert inc_visits_count == batch_sum, (
            f"Committed incremental visits ({inc_visits_count}) must match accepted batch sum ({batch_sum})"
        )
        assert inc_visits_count == 0
        assert batch_sum == 0
    finally:
        session.close()

    overview = service.list_state()
    assert overview["imported_rows"] == 0
    assert overview["dataset"]["imported_rows"] == 0
    assert overview["cumulative_accepted_rows"] == 0


# 3. Cumulative eligible unique-row counts
def test_cumulative_eligible_unique_row_counts():
    session = get_session()
    try:
        total_real = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
        ).scalar()

        orig = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            ~PatientVisit.source_dataset.like("incremental:%"),
        ).scalar()

        inc = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(False),
            PatientVisit.source_dataset.like("incremental:%"),
        ).scalar()

        assert total_real == orig + inc, f"Total ({total_real}) must equal orig ({orig}) + inc ({inc})"
        assert orig == 100001
        assert inc == 0
        assert total_real == 100001

        # Verify no duplicate clinical visits exist across the combined population
        duplicates = session.query(
            PatientVisit.patient_id,
            PatientVisit.visit_number,
            PatientVisit.visit_date,
            func.count(PatientVisit.id),
        ).filter(
            PatientVisit.is_synthetic.is_(False),
        ).group_by(
            PatientVisit.patient_id,
            PatientVisit.visit_number,
            PatientVisit.visit_date,
        ).having(
            func.count(PatientVisit.id) > 1,
        ).all()
        assert len(duplicates) == 0, f"Found duplicate visit keys in database: {duplicates}"
    finally:
        session.close()

    overview = service.list_state()
    assert overview["total_rows"] == 100001
    assert overview["dataset"]["total_rows"] == 100001


# 4. Separation of real and synthetic records
def test_separation_of_real_and_synthetic_records():
    session = get_session()
    try:
        # Real patient visits table must NEVER contain synthetic samples
        synthetic_in_visits = session.query(func.count(PatientVisit.id)).filter(
            PatientVisit.is_synthetic.is_(True),
        ).scalar()
        assert synthetic_in_visits == 0, f"Synthetic rows found in patient_visits: {synthetic_in_visits}"

        # Synthetic training samples table must contain strictly synthetic rows
        synthetic_in_samples = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter(
            SyntheticTrainingSample.is_synthetic.is_(True),
            SyntheticTrainingSample.training_only.is_(True),
        ).scalar()
        latest_run = (
            session.query(IncrementalTrainingRun)
            .filter_by(status="completed")
            .order_by(IncrementalTrainingRun.created_at.desc())
            .first()
        )
        assert latest_run is not None
        expected_synth = int(((latest_run.report or {}).get("smote") or {}).get("synthetic_samples_created") or 0)
        assert synthetic_in_samples == 0 or synthetic_in_samples == expected_synth

        # Verify active run provenance
        active_samples = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter_by(
            training_run_id=latest_run.training_run_id
        ).scalar()
        assert active_samples in (3933, synthetic_in_samples)

        real_in_samples = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter(
            SyntheticTrainingSample.is_synthetic.is_(False),
        ).scalar()
        assert real_in_samples == 0, f"Real rows found in synthetic_training_samples table: {real_in_samples}"
    finally:
        session.close()


# 5. API responses backed by actual database queries
def test_api_overview_response_matches_database(client):
    response = client.get("/api/incremental/overview")
    assert response.status_code == 200, f"API failed with {response.status_code}: {response.text}"
    data = response.json()

    assert "dataset" in data
    assert data["dataset"]["original_rows"] == 100001
    assert data["dataset"]["imported_rows"] == 0
    assert data["dataset"]["total_rows"] == 100001

    assert data["original_rows"] == 100001
    assert data["imported_rows"] == 0
    assert data["total_rows"] == 100001

    assert "active_champion" in data
    assert data["active_champion"]["available"] is True
    assert data["active_champion"]["model_name"] == "XGBoost baseline"

    assert "runs" in data
    assert len(data["runs"]) >= 1
    rolled_back_batch = next(
        batch for batch in data["batches"]
        if batch["batch_id"] == "il-d27e301814880c273656"
    )
    assert rolled_back_batch["status"] == "rolled_back"
    assert rolled_back_batch["validation_report"]["rollback"]["removed_real_visits"] == 9875


# 6. Database query failures returning honest errors
def test_database_query_failure_returns_honest_error(monkeypatch):
    class BrokenSession:
        def query(self, *args, **kwargs):
            raise RuntimeError("Simulated PostgreSQL connection failure")
        def close(self):
            pass

    monkeypatch.setattr(service, "get_session", lambda: BrokenSession())

    # list_state should gracefully capture error or raise honest unavailable exception
    with pytest.raises(Exception) as exc_info:
        service.list_state()
    assert "Simulated PostgreSQL connection failure" in str(exc_info.value)


# 7. Frontend rendering data contract (structure matches frontend expectation)
def test_frontend_data_contract_structure(client):
    response = client.get("/api/incremental/overview")
    assert response.status_code == 200
    data = response.json()

    # Frontend checks: overview?.dataset?.original_rows ?? overview?.original_rows
    assert (data.get("dataset", {}).get("original_rows") is not None or data.get("original_rows") is not None)
    # Frontend checks: overview?.dataset?.imported_rows ?? overview?.imported_rows
    assert (data.get("dataset", {}).get("imported_rows") is not None or data.get("imported_rows") is not None)
    # Frontend checks: overview?.dataset?.total_rows ?? overview?.total_rows
    assert (data.get("dataset", {}).get("total_rows") is not None or data.get("total_rows") is not None)
    # Frontend checks: overview?.active_champion?.version
    assert data.get("active_champion", {}).get("version") is not None


# 8. Detection of fabricated frontend constants or unsupported fallbacks
def test_no_fabricated_metrics_in_code():
    frontend_path = BACKEND_ROOT.parent / "frontend" / "src"
    ts_files = list(frontend_path.glob("**/*.tsx")) + list(frontend_path.glob("**/*.ts"))

    forbidden_patterns = ["93.65%", "92.76%", "98.29%", "92.85%"]
    for path in ts_files:
        content = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in content, f"Hardcoded metric string '{pattern}' found in {path.name}"


# 9. Accuracy recomputation from actual evaluation predictions and labels
def test_accuracy_recomputation_from_persisted_predictions():
    # Test Run 2: ilrun-20261013T045727Z-79194db3e4
    pred_path_2 = (
        BACKEND_ROOT
        / "artifacts"
        / "incremental_learning"
        / "runs"
        / "ilrun-20261013T045727Z-79194db3e4"
        / "champion_comparison_predictions.csv"
    )
    assert pred_path_2.is_file()
    df2 = pd.read_csv(pred_path_2)

    y_true = df2["actual_label"]
    y_pred_champ = df2["champion_predicted_label"]
    y_pred_chal = df2["challenger_predicted_label"]

    champ_acc = accuracy_score(y_true, y_pred_champ)
    chal_acc = accuracy_score(y_true, y_pred_chal)

    # 93.6545% -> 93.65%
    assert round(champ_acc * 100, 2) == 93.65
    assert np.isclose(champ_acc, 0.9365451573935906)

    # 92.7556% -> 92.76%
    assert round(chal_acc * 100, 2) == 92.76
    assert np.isclose(chal_acc, 0.9275563135081362)


# 10. Confusion matrix reconciliation
def test_confusion_matrix_reconciliation():
    # Reconcile Run 2 (ilrun-20261013T045727Z-79194db3e4)
    run_dir = BACKEND_ROOT / "artifacts" / "incremental_learning" / "runs" / "ilrun-20261013T045727Z-79194db3e4"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    comp = manifest["champion_comparison"]
    eval_rows = comp["evaluation_rows"]
    assert eval_rows == 28146

    # Champion confusion matrix
    cm_champ = comp["champion_metrics"]["confusion_matrix"]
    tn_c, fp_c, fn_c, tp_c = cm_champ["tn"], cm_champ["fp"], cm_champ["fn"], cm_champ["tp"]
    assert tn_c + fp_c + fn_c + tp_c == eval_rows == 28146
    assert tn_c == 12969 and fp_c == 1104 and fn_c == 682 and tp_c == 13391

    # Challenger confusion matrix
    cm_chal = comp["challenger_metrics"]["confusion_matrix"]
    tn_ch, fp_ch, fn_ch, tp_ch = cm_chal["tn"], cm_chal["fp"], cm_chal["fn"], cm_chal["tp"]
    assert tn_ch + fp_ch + fn_ch + tp_ch == eval_rows == 28146
    assert tn_ch == 12870 and fp_ch == 1203 and fn_ch == 836 and tp_ch == 13237


# 11. Champion-Challenger comparison compatibility
def test_champion_challenger_comparison_compatibility():
    run_dir = BACKEND_ROOT / "artifacts" / "incremental_learning" / "runs" / "ilrun-20261013T045727Z-79194db3e4"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    comp = manifest["champion_comparison"]

    assert comp["comparable"] is True
    # Protocol must match
    assert comp["evaluation_protocol"] == "SMOTE-before-split diagnostic metrics"
    # Deltas must accurately reflect Challenger minus Champion
    acc_delta = comp["challenger_metrics"]["accuracy"] - comp["champion_metrics"]["accuracy"]
    assert np.isclose(comp["metric_deltas"]["accuracy"], acc_delta)
    # Negative delta of ~ -0.90 percentage points
    assert np.isclose(round(acc_delta * 100, 2), -0.90)


# 12. Run evaluation synthetic sample transparency
def test_synthetic_sample_transparency_in_test_partition():
    run_dir = BACKEND_ROOT / "artifacts" / "incremental_learning" / "runs" / "ilrun-20261013T045727Z-79194db3e4"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    comp = manifest["champion_comparison"]
    synthetic_test_rows = comp["synthetic_rows_in_test"]
    total_eval_rows = comp["evaluation_rows"]
    real_test_rows = total_eval_rows - synthetic_test_rows

    assert synthetic_test_rows == 8120
    assert real_test_rows == 20026
    assert total_eval_rows == 28146

    # Verify predictions file has exact same count of synthetic flags
    df = pd.read_csv(run_dir / "champion_comparison_predictions.csv")
    assert df["is_synthetic"].sum() == 8120
    assert (~df["is_synthetic"]).sum() == 20026


# 13. Dataset fingerprint and run-manifest consistency
def test_dataset_fingerprint_and_manifest_consistency():
    for run_id in ["ilrun-20261013T042318Z-7579f93cb7", "ilrun-20261013T045727Z-79194db3e4"]:
        run_dir = BACKEND_ROOT / "artifacts" / "incremental_learning" / "runs" / run_id
        manifest_path = run_dir / "run_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        training_input = run_dir / "cumulative_real_training_input.csv"
        assert training_input.is_file()
        actual_sha = hashlib.sha256(training_input.read_bytes()).hexdigest()

        assert manifest["dataset_provenance"]["sha256"] == actual_sha
        assert manifest["training_run_id"] == run_id


# 14. Historical run values remaining tied to their original snapshots
def test_historical_run_values_tied_to_original_snapshots():
    overview = service.list_state()
    runs = {r["training_run_id"]: r for r in overview["runs"]}

    # Run 1: baseline historical run created with 100001 rows
    assert "ilrun-20261013T042318Z-7579f93cb7" in runs
    run1 = runs["ilrun-20261013T042318Z-7579f93cb7"]
    assert run1["dataset_rows"] == 100001
    assert run1["decision"] in ("pending", "promoted", "rejected")

    # Preserved historical run manifest on disk retains its original snapshot rows
    run2_manifest_file = (
        BACKEND_ROOT
        / "artifacts"
        / "incremental_learning"
        / "runs"
        / "ilrun-20261013T045727Z-79194db3e4"
        / "run_manifest.json"
    )
    assert run2_manifest_file.is_file()
    run2_manifest = json.loads(run2_manifest_file.read_text(encoding="utf-8"))
    assert run2_manifest["dataset_provenance"]["real_source_rows"] == 100501

    # Historical snapshots stay tied to each run while PostgreSQL reflects current rows.
    assert run1["dataset_rows"] == 100001
    assert overview["total_rows"] == 100001
    # A run artifact without a matching PostgreSQL run row is not exposed as a run.
    assert "ilrun-20261014T102052Z-0ada41b939" not in runs
    rolled_back_batch = next(b for b in overview["batches"] if b["batch_id"] == "il-d27e301814880c273656")
    assert rolled_back_batch["status"] == "rolled_back"
    assert rolled_back_batch["validation_report"]["rollback"]["removed_real_visits"] == 9875


def test_run_counts_without_lineage_are_unavailable_not_inferred(monkeypatch):
    class FakeRun:
        training_run_id = "run-without-lineage"
        status = "completed"
        decision = "pending"
        decision_reason = None
        dataset_rows = None
        dataset_sha256 = None
        artifact_path = None
        manifest_path = None
        created_at = None
        report = {"dataset_provenance": {"real_source_rows": 5}, "smote": {}}

    payload = service._run_summary_from_persisted_record(
        FakeRun(), synthetic_counts_by_run={}, total_stored_synthetic_count=0
    )
    assert payload["dataset_rows"] == 5
    assert payload["new_records_added"] is None
    assert payload["unique_real_patients"] is None
