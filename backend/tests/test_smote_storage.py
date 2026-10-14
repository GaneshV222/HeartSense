from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import sys
import uuid

import pandas as pd
import pytest
from sqlalchemy import event, text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import engine, init_db
from app.ml.verified_pipeline import (
    _dataset_provenance,
    _persist_smote_training_samples,
    _smote_report_fields,
)


class _TransactionBoundEngine:
    def __init__(self, connection):
        self.connection = connection

    @contextmanager
    def begin(self):
        with self.connection.begin_nested():
            yield self.connection


def _canonical_counts(connection):
    return {
        "patients": connection.execute(text("SELECT COUNT(*) FROM patients")).scalar_one(),
        "patient_visits": connection.execute(
            text("SELECT COUNT(*) FROM patient_visits")
        ).scalar_one(),
        "temporal_patient_data": connection.execute(
            text("SELECT COUNT(*) FROM temporal_patient_data")
        ).scalar_one(),
    }


def _persist_in_outer_transaction(monkeypatch, connection, **kwargs):
    import app.database as database

    monkeypatch.setattr(database, "engine", _TransactionBoundEngine(connection))
    monkeypatch.setattr(database, "init_db", lambda: None)
    return _persist_smote_training_samples(**kwargs)


def test_dataset_fingerprint_uses_resolved_source_file(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("patient_id,prediction\nP1,0\n", encoding="utf-8")

    provenance = _dataset_provenance(source)

    assert provenance["resolved_path"] == str(source.resolve())
    assert provenance["sha256"] == sha256(source.read_bytes()).hexdigest()


def test_training_report_distinguishes_generated_and_saved_counts():
    disabled = _smote_report_fields(
        training_run_id="run-disabled",
        real_source_rows=10,
        synthetic_rows_generated=4,
        synthetic_rows_saved_to_csv=4,
        synthetic_rows_saved_to_database=0,
        sampler_configuration={"random_state": 42, "k_neighbors": 5},
        csv_path="/tmp/smote.csv",
        database_enabled=False,
        dataset_sha256="a" * 64,
    )
    enabled = _smote_report_fields(
        training_run_id="run-enabled",
        real_source_rows=10,
        synthetic_rows_generated=4,
        synthetic_rows_saved_to_csv=4,
        synthetic_rows_saved_to_database=4,
        sampler_configuration={"random_state": 42, "k_neighbors": 5},
        csv_path="/tmp/smote.csv",
        database_enabled=True,
        dataset_sha256="b" * 64,
    )

    assert disabled["synthetic_rows_generated"] == disabled["synthetic_rows_saved_to_csv"] == 4
    assert disabled["synthetic_rows_saved_to_database"] == 0
    assert disabled["database_table"] is None
    assert enabled["synthetic_rows_saved_to_database"] == 4
    assert enabled["database_table"] == "synthetic_training_samples"


def test_synthetic_persistence_is_idempotent_and_never_touches_clinical_tables(monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Synthetic persistence requires the configured PostgreSQL database")
    init_db()
    run_id = f"test-{uuid.uuid4()}"
    features = pd.DataFrame([{"age": 54.0, "gender": "Female"}, {"age": 63.0, "gender": "Male"}])
    targets = pd.Series([1, 1])
    targets.attrs["__training_patient_split"] = {"patient_overlap": 0}
    protected_artifacts = [
        BACKEND_ROOT / "artifacts" / "models" / "active_model.json",
        BACKEND_ROOT / "artifacts" / "models" / "temporal" / "active_temporal_model.json",
        BACKEND_ROOT / "artifacts" / "temporal_model_artifact.joblib",
    ]
    artifacts_before = {
        path: sha256(path.read_bytes()).hexdigest()
        for path in protected_artifacts
        if path.exists()
    }

    with engine.connect() as connection:
        transaction = connection.begin()
        before = _canonical_counts(connection)
        try:
            options = {
                "synthetic_features": features,
                "synthetic_labels": targets,
                "training_run_id": run_id,
                "dataset_path": str((BACKEND_ROOT / "data" / "heart_disease_prediction_2026.csv").resolve()),
                "dataset_sha256": "c" * 64,
                "expected_database": connection.execute(
                    text("SELECT current_database()")
                ).scalar_one(),
            }
            first = _persist_in_outer_transaction(monkeypatch, connection, **options)
            second = _persist_in_outer_transaction(monkeypatch, connection, **options)

            assert first["generated_rows"] == first["verified_persisted_rows"] == 2
            assert first["inserted_rows"] == 2
            assert second["inserted_rows"] == 0
            assert second["existing_rows_on_retry"] == 2
            assert second["verified_persisted_rows"] == 2
            assert _canonical_counts(connection) == before
            assert connection.execute(
                text("SELECT COUNT(*) FROM synthetic_training_samples WHERE training_run_id=:run"),
                {"run": run_id},
            ).scalar_one() == 2
            assert connection.execute(
                text("SELECT COUNT(*) FROM patients WHERE patient_id LIKE 'SMOTE_%'")
            ).scalar_one() == 0
            assert connection.execute(
                text("SELECT COUNT(*) FROM patient_visits WHERE is_synthetic=TRUE")
            ).scalar_one() == 0
            assert connection.execute(
                text(
                    "SELECT COUNT(*) FROM temporal_patient_data "
                    "WHERE patient_id LIKE 'SMOTE_%' OR patient_id LIKE 'SYNTHETIC_%'"
                )
            ).scalar_one() == 0
            saved_metadata = connection.execute(
                text(
                    "SELECT generation_method, is_synthetic, training_only, "
                    "dataset_sha256 FROM synthetic_training_samples "
                    "WHERE training_run_id=:run ORDER BY sample_index"
                ),
                {"run": run_id},
            ).all()
            assert saved_metadata == [("SMOTENC", True, True, "c" * 64)] * 2
            assert artifacts_before == {
                path: sha256(path.read_bytes()).hexdigest()
                for path in protected_artifacts
                if path.exists()
            }
        finally:
            transaction.rollback()


def test_failed_synthetic_insert_rolls_back(monkeypatch):
    if engine.dialect.name != "postgresql":
        pytest.skip("Synthetic persistence requires the configured PostgreSQL database")
    init_db()
    run_id = f"test-rollback-{uuid.uuid4()}"
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            def fail_insert(conn, cursor, statement, parameters, context, executemany):
                if statement.lstrip().lower().startswith("insert into synthetic_training_samples"):
                    raise RuntimeError("simulated insert failure")

            fail_labels = pd.Series([1])
            fail_labels.attrs["__training_patient_split"] = {"patient_overlap": 0}
            event.listen(connection, "before_cursor_execute", fail_insert)
            with pytest.raises(RuntimeError, match="simulated insert failure"):
                _persist_in_outer_transaction(
                    monkeypatch,
                    connection,
                    synthetic_features=pd.DataFrame([{"age": 47.0}]),
                    synthetic_labels=fail_labels,
                    training_run_id=run_id,
                    dataset_path="/source.csv",
                    dataset_sha256="d" * 64,
                    expected_database=connection.execute(
                        text("SELECT current_database()")
                    ).scalar_one(),
                )
            event.remove(connection, "before_cursor_execute", fail_insert)
            assert connection.execute(
                text("SELECT COUNT(*) FROM synthetic_training_samples WHERE training_run_id=:run"),
                {"run": run_id},
            ).scalar_one() == 0
        finally:
            transaction.rollback()
