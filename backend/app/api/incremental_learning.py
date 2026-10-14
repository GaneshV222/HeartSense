"""Manual batch validation, import, and Challenger lifecycle endpoints."""

import logging
import uuid

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services import incremental_learning as service
from app.database import get_session
from app.models import IncrementalTrainingRun

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class DecisionRequest(BaseModel):
    reason: str = Field(min_length=5, max_length=1000)


@router.get("/incremental/overview")
def incremental_overview():
    try:
        return service.list_state()
    except Exception as exc:
        logger.exception("Incremental learning overview query failed")
        raise HTTPException(status_code=503, detail=f"Incremental learning data is unavailable ({type(exc).__name__})") from exc


@router.post("/incremental/batches/validate")
async def validate_batch(
    clinical_file: UploadFile = File(...),
    temporal_file: UploadFile | None = File(default=None),
):
    try:
        if clinical_file.size is not None and clinical_file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Clinical CSV exceeds the 10 MB upload limit")
        if temporal_file is not None and temporal_file.size is not None and temporal_file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Temporal workbook exceeds the 10 MB upload limit")
        source_bytes = await clinical_file.read()
        temporal_bytes = await temporal_file.read() if temporal_file is not None else None
        result = service.validate_batch(
            source_bytes,
            clinical_file.filename or "clinical.csv",
            temporal_bytes,
            temporal_file.filename if temporal_file is not None else None,
        )
        return service.save_validation(result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Batch validation failed: {exc}") from exc


@router.post("/incremental/batches/{batch_id}/confirm")
def confirm_batch(batch_id: str, strict: bool = False):
    try:
        return service.confirm_batch(batch_id, strict=strict)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Batch import failed: {exc}") from exc


@router.get("/incremental/training-runs/{training_run_id}/synthetic-count")
def get_training_run_synthetic_count(training_run_id: str):
    try:
        return service.get_run_synthetic_count(training_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Synthetic count query failed for training run %s", training_run_id)
        raise HTTPException(status_code=503, detail=f"Failed to query synthetic count ({type(exc).__name__})") from exc


@router.post("/incremental/training-runs")
def train_challenger(idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=16, max_length=100)):
    try:
        return service.run_cumulative_training(idempotency_key=idempotency_key)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cumulative Challenger training failed: {exc}") from exc


@router.post("/incremental/training-runs/{training_run_id}/promote")
def promote_challenger(training_run_id: str):
    try:
        return service.promote_run(training_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {exc}") from exc


@router.post("/incremental/training-runs/{training_run_id}/reject")
def reject_challenger(training_run_id: str, request: DecisionRequest):
    session = get_session()
    try:
        run = session.get(IncrementalTrainingRun, training_run_id)
        if run is None or run.status != "completed":
            raise HTTPException(status_code=404, detail="Completed training run was not found")
        if run.decision == "promoted":
            raise HTTPException(status_code=409, detail="A promoted run cannot be marked rejected")
        run.decision = "rejected"
        run.decision_reason = request.reason
        session.commit()
        return {"training_run_id": training_run_id, "decision": run.decision,
                "reason": run.decision_reason}
    finally:
        session.close()


@router.post("/incremental/champion/rollback")
def rollback_champion():
    try:
        return service.rollback_champion()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Champion rollback failed: {exc}") from exc
