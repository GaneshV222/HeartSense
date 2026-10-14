from types import SimpleNamespace
import hashlib

import pandas as pd

import pytest

from app.services.incremental_learning import (
    IdempotencyKeyConflict,
    _find_idempotent_training_run,
)
from app.services import incremental_learning as service


class _Query:
    def __init__(self, row):
        self.row = row
        self.key = None

    def filter_by(self, **kwargs):
        self.key = kwargs["idempotency_key"]
        return self

    def one_or_none(self):
        if self.row and self.row.idempotency_key == self.key:
            return self.row
        return None


class _Session:
    def __init__(self, row):
        self.row = row

    def query(self, _model):
        return _Query(self.row)


def test_matching_retry_returns_existing_run_without_creating_another():
    existing = SimpleNamespace(
        training_run_id="ilrun-existing",
        idempotency_key="request-key-00000001",
        dataset_sha256="dataset-sha",
        status="completed",
    )

    replay = _find_idempotent_training_run(
        _Session(existing), "request-key-00000001", "dataset-sha"
    )

    assert replay is existing


def test_reusing_key_for_different_input_is_rejected():
    existing = SimpleNamespace(
        training_run_id="ilrun-existing",
        idempotency_key="request-key-00000001",
        dataset_sha256="dataset-sha-original",
        status="completed",
    )

    with pytest.raises(IdempotencyKeyConflict, match="different cumulative dataset"):
        _find_idempotent_training_run(
            _Session(existing), "request-key-00000001", "different-dataset-sha"
        )


def test_new_key_has_no_prior_run():
    assert _find_idempotent_training_run(
        _Session(None), "request-key-00000002", "dataset-sha"
    ) is None


def test_training_service_replays_existing_run_without_training(monkeypatch, tmp_path):
    dataset = tmp_path / "source.csv"
    frame = pd.DataFrame({"patient_id": ["P1"], "prediction": [0]})
    dataset.write_text(frame.to_csv(index=False), encoding="utf-8")
    fingerprint = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    existing = SimpleNamespace(
        training_run_id="ilrun-existing",
        idempotency_key="request-key-00000001",
        dataset_sha256=fingerprint,
        status="completed",
        decision="rejected",
        report={"training_run_id": "ilrun-existing", "stored": True},
    )

    class LockConnection:
        def execute(self, *_args, **_kwargs):
            return None

        def commit(self):
            return None

        def close(self):
            return None

    class FakeEngine:
        def connect(self):
            return LockConnection()

    class FakeSession(_Session):
        def close(self):
            return None

        def rollback(self):
            return None

    monkeypatch.setattr(service, "get_session", lambda: FakeSession(existing))
    monkeypatch.setattr(service, "_current_batches", lambda _session: [])
    monkeypatch.setattr(service, "DATASET_PATH", str(dataset))
    monkeypatch.setattr("app.database.engine", FakeEngine())
    monkeypatch.setattr(service, "RUN_ROOT", tmp_path / "runs")

    result = service.run_cumulative_training(idempotency_key="request-key-00000001")

    assert result == {
        "training_run_id": "ilrun-existing",
        "status": "completed",
        "decision": "rejected",
        "report": existing.report,
        "idempotent_replay": True,
    }
    assert not (tmp_path / "runs").exists()


def test_training_api_requires_a_retry_idempotency_key(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(
        service,
        "run_cumulative_training",
        lambda **_kwargs: pytest.fail("Training must not start without an idempotency key"),
    )
    with TestClient(app) as client:
        response = client.post("/api/incremental/training-runs")

    assert response.status_code == 422
