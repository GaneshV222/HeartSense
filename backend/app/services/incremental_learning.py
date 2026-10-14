"""Validated batch import and cumulative Challenger training orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import shutil
import threading
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sqlalchemy import func, or_, select, text

from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR, TEMPORAL_DATASET_PATH
from app.database import get_session
from app.ml.verified_pipeline import (
    BASELINE_FEATURE_COLS,
    CATEGORICAL_BASE_COLUMNS,
    CLINICAL_SCREENING_BOUNDS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _metrics,
    _normalise_categorical_values,
    _scores,
    run_verified_training_pipeline,
)
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    PatientVisit,
    SyntheticTrainingSample,
)
from app.services.temporal_service import TemporalFeatureService

logger = logging.getLogger(__name__)

TRAINING_LOCK = threading.Lock()
ARTIFACT_ROOT = Path(MODEL_ARTIFACT_DIR) / "incremental_learning"
IMPORT_ROOT = ARTIFACT_ROOT / "imports"
RUN_ROOT = ARTIFACT_ROOT / "runs"
IDENTITY_COLUMNS = ("patient_id", "visit_number", "visit_date")
OPTIONAL_SOURCE_COLUMNS = {"patient_name", "name", "source_row_number"}


class IdempotencyKeyConflict(ValueError):
    """Raised when an idempotency key is reused for different training input."""
    pass


def _find_idempotent_training_run(
    session, idempotency_key: str | None, dataset_sha256: str | None
):
    if not idempotency_key:
        return None
    existing = (
        session.query(IncrementalTrainingRun)
        .filter_by(idempotency_key=idempotency_key)
        .one_or_none()
    )
    if existing is not None:
        if existing.dataset_sha256 != dataset_sha256:
            raise IdempotencyKeyConflict(
                f"Idempotency key {idempotency_key} was already used with a different cumulative dataset"
            )
        return existing
    return None


CLINICAL_COLUMN_ALIASES = {
    "subject_id": "patient_id",
    "encounter_no": "visit_number",
    "encounter_id": "visit_number",
    "encounter_date": "visit_date",
    "age_years": "age",
    "body_mass_index": "bmi",
    "cp_category": "chest_pain_type",
    "sbp_mmhg": "systolic_bp",
    "dbp_mmhg": "diastolic_bp",
    "rest_hr_bpm": "resting_heart_rate",
    "peak_hr_bpm": "max_heart_rate",
    "total_chol_mgdl": "cholesterol",
    "hdl_chol_mgdl": "hdl",
    "ldl_chol_mgdl": "ldl",
    "fbs_elevated": "fasting_blood_sugar",
    "glycated_hb_pct": "hba1c",
    "diabetic_flag": "diabetes",
    "rest_ecg_result": "resting_ecg",
    "exertional_angina": "exercise_angina",
    "st_depression": "oldpeak",
    "st_segment_slope": "st_slope",
    "major_vessel_count": "num_major_vessels",
    "thal_status": "thalassemia",
    "current_smoker": "smoking",
    "family_cvd_history": "family_history",
    "heart_disease_label": TARGET_COLUMN,
    "target": TARGET_COLUMN,
    "label": TARGET_COLUMN,
    "sex": "gender",
    "activity_level": "physical_activity",
    "stress_score": "stress_level",
}

EXPLICIT_ACTIVITY_LEVEL_MAP = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
}


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _file_sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _acquire_training_database_lock():
    from app.database import engine

    connection = engine.connect()
    try:
        connection.execute(
            text("SELECT pg_advisory_lock(hashtext(:lock_name))"),
            {"lock_name": "heartsense_incremental_training"},
        )
        connection.commit()
        return connection
    except Exception:
        connection.close()
        raise


def _release_training_database_lock(connection) -> None:
    try:
        connection.execute(
            text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
            {"lock_name": "heartsense_incremental_training"},
        )
        connection.commit()
    finally:
        connection.close()


def _normalise_upload(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    original_names = list(map(str, frame.columns))
    normalized_names = [
        name.strip().casefold().replace(" ", "_").replace("-", "_")
        for name in original_names
    ]
    if len(set(normalized_names)) != len(normalized_names):
        duplicates = sorted({name for name in normalized_names if normalized_names.count(name) > 1})
        raise ValueError(f"Ambiguous duplicate columns after case/whitespace normalization: {duplicates}")
    frame.columns = normalized_names
    frame.attrs["source_columns"] = dict(zip(normalized_names, original_names))
    frame.attrs["source_column_profiles"] = {
        source: _column_profile(frame[source])
        for source in ("activity_level", "stress_score", "physical_activity", "stress_level")
        if source in frame.columns
    }

    if "activity_level" in frame.columns:
        def _map_activity(val):
            if pd.isna(val):
                return val
            s = str(val).strip().casefold()
            return EXPLICIT_ACTIVITY_LEVEL_MAP.get(s, val)

        frame["activity_level"] = frame["activity_level"].map(_map_activity)

    sources_by_target: dict[str, list[str]] = {}
    for source, target in CLINICAL_COLUMN_ALIASES.items():
        if source in frame.columns:
            sources_by_target.setdefault(target, []).append(source)
    mappings = {}
    alias_sources = {}
    for target, sources in sources_by_target.items():
        sources = ([target] if target in frame.columns else []) + sources
        canonical = sources[0]
        for other in sources[1:]:
            left, right = frame[canonical], frame[other]
            equal = (left.isna() & right.isna()) | (
                left.notna() & right.notna()
                & left.astype("string").str.strip().eq(right.astype("string").str.strip())
            )
            if not bool(equal.all()):
                raise ValueError(
                    f"Conflicting columns {canonical!r} and {other!r} both map to {target!r}; "
                    "make their non-empty values agree or keep only one column."
                )
            if canonical != target:
                frame[target] = frame[canonical]
            frame = frame.drop(columns=[other])
        if canonical != target and canonical in frame.columns:
            frame = frame.rename(columns={canonical: target})
        for source in sources:
            if source != target:
                mappings[frame.attrs["source_columns"].get(source, source)] = target
                alias_sources.setdefault(target, []).append(source)
    frame.attrs["column_mappings"] = mappings
    frame.attrs["alias_sources"] = alias_sources
    return frame


def _column_profile(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    numeric = pd.to_numeric(series, errors="coerce")
    profile = {
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "missing": int(series.isna().sum()),
        "distinct_non_missing": int(series.nunique(dropna=True)),
        "value_counts": {str(key): int(value) for key, value in counts.head(20).items()},
    }
    valid_numeric = numeric.dropna()
    if len(valid_numeric) and int(valid_numeric.notna().sum()) == int(series.notna().sum()):
        profile["numeric_min"] = float(valid_numeric.min())
        profile["numeric_max"] = float(valid_numeric.max())
    return profile


def _existing_visit_keys(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    session = get_session()
    accepted_paths = []
    try:
        ids = frame["patient_id"].dropna().astype(str).unique().tolist()
        for offset in range(0, len(ids), 1000):
            visits = session.query(
                PatientVisit.patient_id, PatientVisit.visit_date, PatientVisit.visit_number
            ).filter(PatientVisit.patient_id.in_(ids[offset : offset + 1000])).all()
            keys.update(
                (str(patient_id), pd.Timestamp(visit_date).isoformat(), int(visit_number or 0))
                for patient_id, visit_date, visit_number in visits
            )
        accepted_paths = [
            path for (path,) in session.query(IncrementalLearningBatch.source_path)
            .filter_by(status="accepted").all()
        ]
    finally:
        session.close()
    original = Path(DATASET_PATH)
    if original.is_file():
        stat = original.stat()
        keys.update(_configured_source_visit_keys(
            str(original.resolve()), stat.st_mtime_ns, stat.st_size
        ))
    for path_text in accepted_paths:
        path = Path(path_text)
        if path.is_file():
            stat = path.stat()
            keys.update(_configured_source_visit_keys(
                str(path.resolve()), stat.st_mtime_ns, stat.st_size
            ))
    return keys


@lru_cache(maxsize=8)
def _configured_source_visit_keys(
    file_path: str, modified_ns: int, file_size: int
) -> set[tuple[str, str, int]]:
    """Cache real visit identities for duplicate checks; source stat invalidates cache."""
    frame = pd.read_csv(
        file_path, usecols=["patient_id", "visit_number", "visit_date"]
    )
    # pandas preserves source-file column order for usecols; enforce the tuple
    # order expected below instead of assuming the requested order was retained.
    frame = frame[["patient_id", "visit_number", "visit_date"]]
    frame["visit_date"] = pd.to_datetime(frame["visit_date"], errors="coerce")
    frame["visit_number"] = pd.to_numeric(frame["visit_number"], errors="coerce")
    return {
        (str(patient_id).strip(), pd.Timestamp(visit_date).isoformat(), int(visit_number))
        for patient_id, visit_number, visit_date in frame.itertuples(index=False, name=None)
        if pd.notna(patient_id) and pd.notna(visit_date) and pd.notna(visit_number)
    }


def _validate_temporal_pair(data: bytes | None, filename: str | None, frame: pd.DataFrame):
    if data is None:
        return None, None, {"provided": False, "required": False}
    try:
        temporal = pd.read_excel(BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Unable to read temporal workbook: {exc}") from exc
    temporal = _normalise_upload(temporal)
    if temporal.columns.duplicated().any():
        raise ValueError("Temporal workbook has duplicate columns after normalisation")
    if "patient_id" not in temporal:
        raise ValueError("Temporal workbook must contain subject_id or patient_id")
    allowed_temporal_columns = {
        "id", "patient_id", "visit_number", "visit_date", "assessment_date",
        "source_row_number", "smoking_transition", "physical_activity_transition",
    }
    allowed_temporal_columns.update(
        f"current_{name}" for name in BASELINE_FEATURE_COLS
    )
    for variable in (
        "systolic_bp", "diastolic_bp", "cholesterol", "ldl", "hdl", "bmi",
        "hba1c", "resting_heart_rate", "smoking", "physical_activity",
    ):
        allowed_temporal_columns.update(
            {f"{variable}_prev", f"{variable}_curr", f"{variable}_change"}
        )
        allowed_temporal_columns.add(f"current_{variable}")
    unexpected = sorted(set(temporal.columns) - allowed_temporal_columns)
    if unexpected:
        raise ValueError(f"Temporal workbook contains unsupported columns: {unexpected}")
    if temporal["patient_id"].isna().any():
        raise ValueError("Temporal workbook contains missing subject_id values")
    ids = temporal["patient_id"].astype(str).str.strip()
    temporal = temporal.copy()
    temporal["patient_id"] = ids
    matching_columns = ["patient_id"]
    if "visit_number" in frame.columns and "visit_number" in temporal.columns:
        temporal["visit_number"] = pd.to_numeric(temporal["visit_number"], errors="coerce")
        matching_columns.append("visit_number")
    if "visit_date" in frame.columns and "visit_date" in temporal.columns:
        temporal["visit_date"] = pd.to_datetime(temporal["visit_date"], errors="coerce")
        if temporal["visit_date"].isna().any():
            raise ValueError("Temporal workbook has invalid visit_date values")
        matching_columns.append("visit_date")
    duplicate_csv = frame.duplicated(matching_columns)
    duplicate_xlsx = temporal.duplicated(matching_columns)
    if duplicate_csv.any() or duplicate_xlsx.any():
        raise ValueError("CSV/XLSX pair has ambiguous duplicate identity keys")
    csv_keys = set(map(tuple, frame[matching_columns].astype(str).to_numpy()))
    workbook_keys = set(map(tuple, temporal[matching_columns].astype(str).to_numpy()))
    missing = sorted(csv_keys - workbook_keys)
    extra = sorted(workbook_keys - csv_keys)
    if missing or extra:
        raise ValueError(
            "CSV/XLSX identities do not align by subject_id: "
            f"missing_from_xlsx={len(missing)}, extra_in_xlsx={len(extra)}"
        )
    # Existing clinical measurements are authoritative. If the workbook supplies
    # current measurements, compare by identity rather than relying on row order.
    checked = 0
    conflicts = 0
    current_aliases = {f"current_{name}": name for name in BASELINE_FEATURE_COLS}
    current_aliases.update({f"{name}_curr": name for name in BASELINE_FEATURE_COLS})
    temporal = temporal.rename(columns=current_aliases)
    common = [name for name in BASELINE_FEATURE_COLS if name in temporal.columns]
    if common:
        left = frame.drop_duplicates(matching_columns).set_index(matching_columns)
        right = temporal.drop_duplicates(matching_columns).set_index(matching_columns)
        for column in common:
            a = pd.to_numeric(left[column], errors="coerce")
            b = pd.to_numeric(right[column], errors="coerce")
            present = a.notna() & b.reindex(a.index).notna()
            checked += int(present.sum())
            conflicts += int((present & ((a - b.reindex(a.index)).abs() > 1e-6)).sum())
        if conflicts:
            raise ValueError(f"Temporal workbook has {conflicts} current-measurement conflicts")
    delta_checks = {}
    for variable in (
        "systolic_bp", "diastolic_bp", "cholesterol", "ldl", "hdl", "bmi",
        "hba1c", "resting_heart_rate",
    ):
        previous_column = f"{variable}_prev"
        change_column = f"{variable}_change"
        if previous_column not in temporal or variable not in temporal or change_column not in temporal:
            continue
        previous = pd.to_numeric(temporal[previous_column], errors="coerce")
        current = pd.to_numeric(temporal[variable], errors="coerce")
        reported = pd.to_numeric(temporal[change_column], errors="coerce")
        comparable = previous.notna() & current.notna() & reported.notna()
        mismatches = int((comparable & ((current - previous - reported).abs() > 0.02)).sum())
        delta_checks[variable] = {"compared": int(comparable.sum()), "conflicts": mismatches}
        if mismatches:
            raise ValueError(
                f"Temporal workbook {variable} deltas violate current-minus-previous rules in {mismatches} rows"
            )
    categorical_checks = {}
    for variable in ("smoking", "physical_activity"):
        previous_column = f"{variable}_prev"
        change_column = f"{variable}_change"
        if previous_column not in temporal or variable not in temporal or change_column not in temporal:
            continue
        comparable = temporal[previous_column].notna() & temporal[variable].notna() & temporal[change_column].notna()
        mismatches = 0
        for index in temporal.index[comparable]:
            current_data = {variable: temporal.at[index, variable]}
            previous_data = {variable: temporal.at[index, previous_column]}
            snapshot = TemporalFeatureService.calculate_snapshot(
                current_data=current_data,
                previous_data=previous_data,
                patient_id=str(temporal.at[index, "patient_id"]),
            )
            expected = bool(snapshot.get(f"{variable}_changed"))
            reported_value = temporal.at[index, change_column]
            if isinstance(reported_value, str):
                reported_text = reported_value.strip().casefold()
                if reported_text in {"true", "yes", "changed"}:
                    reported_value = True
                elif reported_text in {"false", "no", "unchanged"}:
                    reported_value = False
            if pd.isna(reported_value) or bool(reported_value) != expected:
                mismatches += 1
        categorical_checks[variable] = {
            "compared": int(comparable.sum()),
            "conflicts": mismatches,
            "rule": "TemporalFeatureService.calculate_snapshot",
        }
        if mismatches:
            raise ValueError(
                f"Temporal workbook {variable} categorical shifts violate temporal-service rules in {mismatches} rows"
            )
    return data, filename, {
        "provided": True,
        "identity_key": matching_columns,
        "identities_matched": len(csv_keys),
        "current_measurements_checked": checked,
        "current_measurement_conflicts": conflicts,
        "delta_checks": delta_checks,
        "categorical_shift_checks": categorical_checks,
        "row_order_used_for_matching": False,
    }


def validate_batch(
    source_bytes: bytes,
    filename: str,
    temporal_bytes: bytes | None = None,
    temporal_filename: str | None = None,
) -> dict:
    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("Clinical batch must be a labelled CSV file")
    try:
        frame = _normalise_upload(pd.read_csv(BytesIO(source_bytes)))
    except Exception as exc:
        raise ValueError(f"Unable to read clinical CSV: {exc}") from exc
    if "subject_id" in frame.columns and "patient_id" not in frame.columns:
        frame = frame.rename(columns={"subject_id": "patient_id"})

    required = set(BASELINE_FEATURE_COLS) | set(IDENTITY_COLUMNS) | {TARGET_COLUMN}
    missing_columns = sorted(required - set(frame.columns))
    unexpected = sorted(set(frame.columns) - required - OPTIONAL_SOURCE_COLUMNS)
    if missing_columns or unexpected:
        problem_parts = ["Clinical CSV schema check failed."]
        if missing_columns:
            problem_parts.append("Required fields missing: " + ", ".join(missing_columns) + ".")
        if unexpected:
            problem_parts.append("Unsupported columns: " + ", ".join(unexpected) + ".")
        raise ValueError(" ".join(problem_parts))
    frame = frame.replace({"": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan})
    source_column_profiles = frame.attrs.get("source_column_profiles", {})
    alias_sources = frame.attrs.get("alias_sources", {})
    frame["patient_id"] = frame["patient_id"].astype("string").str.strip()
    frame["visit_date"] = pd.to_datetime(frame["visit_date"], errors="coerce")
    frame["visit_number"] = pd.to_numeric(frame["visit_number"], errors="coerce")
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = labels

    errors: dict[int, list[str]] = {}
    duplicates: set[int] = set()
    seen: set[tuple[str, str, int]] = set()
    existing = _existing_visit_keys(frame)
    numeric_columns = [
        column for column in BASELINE_FEATURE_COLS if column not in CATEGORICAL_BASE_COLUMNS
    ]
    allowed_categories = {
        "gender": {"male", "female"},
        "chest_pain_type": {"typical angina", "atypical angina", "non-anginal pain", "non anginal pain", "asymptomatic"},
        "fasting_blood_sugar": {"true", "false"},
        "diabetes": {"true", "false"},
        "resting_ecg": {"normal", "st-t wave abnormality", "left ventricular hypertrophy"},
        "exercise_angina": {"true", "false"},
        "st_slope": {"up", "flat", "down"},
        "thalassemia": {"normal", "fixed defect", "reversible defect"},
        "smoking": {"true", "false"},
        "family_history": {"true", "false"},
        "physical_activity": {"low", "moderate", "high"},
    }
    for index, row in frame.iterrows():
        row_number = int(index) + 2
        row_errors: list[str] = []
        patient_id = row.get("patient_id")
        if pd.isna(patient_id) or not str(patient_id).strip():
            row_errors.append("missing patient_id/subject_id")
        if pd.isna(row.get("visit_date")):
            row_errors.append("invalid or missing visit_date")
        visit_number = row.get("visit_number")
        if pd.isna(visit_number) or visit_number < 1 or float(visit_number) % 1:
            row_errors.append("visit_number must be a positive integer")
        label = row.get(TARGET_COLUMN)
        if pd.isna(label) or label not in (0, 1):
            row_errors.append("prediction must be labelled 0 or 1")
        for column in numeric_columns:
            raw = row.get(column)
            if pd.notna(raw):
                try:
                    value = float(raw)
                    if not np.isfinite(value):
                        raise ValueError
                    frame.at[index, column] = value
                    if column == "stress_level" and "stress_score" in alias_sources.get(column, []):
                        if not 1 <= value <= 10:
                            row_errors.append(
                                "stress_score must use the source dataset's observed stress_level scale of 1 to 10"
                            )
                except (TypeError, ValueError):
                    row_errors.append(f"{column} must be numeric")
        for column, permitted in allowed_categories.items():
            raw = row.get(column)
            if pd.notna(raw) and str(raw).strip().casefold() not in permitted:
                row_errors.append(
                    f"{column} value {str(raw)!r} is unsupported; allowed values are {sorted(permitted)}"
                )
        if not row_errors:
            for column, (lower, upper) in CLINICAL_SCREENING_BOUNDS.items():
                value = row.get(column)
                if pd.notna(value) and not lower <= float(value) <= upper:
                    row_errors.append(f"{column} is outside screening range [{lower}, {upper}]")
            key = (
                str(patient_id),
                pd.Timestamp(row["visit_date"]).isoformat(),
                int(visit_number),
            )
            if key in seen or key in existing:
                duplicates.add(int(index))
                row_errors.append("duplicate patient/date/visit_number key")
            else:
                seen.add(key)
        if row_errors:
            errors[int(index)] = row_errors

    normalized_columns = [*IDENTITY_COLUMNS, *BASELINE_FEATURE_COLS, TARGET_COLUMN]
    normalized_columns.extend(sorted(OPTIONAL_SOURCE_COLUMNS.intersection(frame.columns)))
    accepted_indices = [index for index in frame.index if index not in errors]
    accepted = frame.loc[accepted_indices, normalized_columns].copy()
    if len(accepted):
        accepted["visit_number"] = accepted["visit_number"].astype(int)
        accepted[TARGET_COLUMN] = accepted[TARGET_COLUMN].astype(int)
        accepted = _normalise_categorical_values(accepted)
    temporal_bytes, temporal_filename, temporal_report = _validate_temporal_pair(
        temporal_bytes, temporal_filename, frame
    )
    missing_cells = int(accepted[list(BASELINE_FEATURE_COLS)].isna().sum().sum()) if len(accepted) else 0
    source_hash = _file_sha(source_bytes)
    temporal_hash = _file_sha(temporal_bytes) if temporal_bytes is not None else None
    key_payload = f"{source_hash}:{temporal_hash or ''}:{Path(filename).name}"
    idempotency_key = sha256(key_payload.encode("utf-8")).hexdigest()
    batch_id = f"il-{idempotency_key[:20]}"
    # Longitudinal signal analysis (Phase 7)
    first_visits = int((accepted["visit_number"] == 1).sum()) if len(accepted) else 0
    followup_visits = int((accepted["visit_number"] > 1).sum()) if len(accepted) else 0
    has_longitudinal_signal = followup_visits > 0
    longitudinal_warning = (
        "Batch contains exclusively first visits (visit_number=1); no longitudinal follow-up signal or delta calculations available."
        if (len(accepted) > 0 and followup_visits == 0)
        else None
    )
    longitudinal_report = {
        "first_visit_count": first_visits,
        "followup_visit_count": followup_visits,
        "has_longitudinal_signal": has_longitudinal_signal,
        "warning": longitudinal_warning,
    }

    report = {
        "batch_id": batch_id,
        "idempotency_key": idempotency_key,
        "source_filename": Path(filename).name,
        "source_sha256": source_hash,
        "source_rows": int(len(frame)),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(len(errors) - len(duplicates)),
        "duplicate_rows": int(len(duplicates)),
        "missing_value_cells": missing_cells,
        "missing_value_counts": {
            column: int(count)
            for column, count in accepted[list(BASELINE_FEATURE_COLS)].isna().sum().items()
            if count
        } if len(accepted) else {},
        "class_distribution": {
            str(label): int(count)
            for label, count in accepted[TARGET_COLUMN].value_counts().sort_index().items()
        } if len(accepted) else {},
        "feature_schema": list(BASELINE_FEATURE_COLS),
        "detected_format": "HeartSense clinical visit CSV",
        "column_mappings": frame.attrs.get("column_mappings", {}),
        "source_column_profiles": source_column_profiles,
        "normalized_columns": normalized_columns,
        "preview_rows": [
            {key: _json_value(value) for key, value in row.items()}
            for row in frame.loc[:, normalized_columns].head(5).to_dict(orient="records")
        ],
        "validation_status": "valid" if not errors else "invalid",
        "preprocessing": "verified pipeline median imputation for numeric features and missing-category imputation for categorical features",
        "temporal_validation": temporal_report,
        "longitudinal_signal": longitudinal_report,
        "row_errors": [
            {"source_row_number": index + 2, "errors": issues}
            for index, issues in list(errors.items())[:500]
        ],
    }
    return {
        "batch_id": batch_id,
        "idempotency_key": idempotency_key,
        "report": report,
        "accepted": accepted,
        "source_bytes": source_bytes,
        "temporal_bytes": temporal_bytes,
        "temporal_filename": temporal_filename,
    }


def save_validation(result: dict) -> dict:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
    session = get_session()
    try:
        existing = session.query(IncrementalLearningBatch).filter_by(
            idempotency_key=result["idempotency_key"]
        ).first()
        if existing:
            return _batch_payload(existing)

        batch_id = result["batch_id"]
        idempotency_key = result["idempotency_key"]
        csv_path = IMPORT_ROOT / f"{batch_id}.csv"
        rows = result["accepted"].copy()
        rows["source_row_number"] = [int(value) + 2 for value in rows.index]
        rows.to_csv(csv_path, index=False)
        temporal_path = None
        temporal_hash = None
        if result["temporal_bytes"] is not None:
            temporal_path = IMPORT_ROOT / f"{batch_id}.xlsx"
            temporal_path.write_bytes(result["temporal_bytes"])
            temporal_hash = _file_sha(result["temporal_bytes"])
        report = dict(result["report"])
        report["batch_id"] = batch_id
        report["idempotency_key"] = idempotency_key
        report["accepted_rows_sha256"] = _file_sha(csv_path.read_bytes())
        report["temporal_filename"] = result["temporal_filename"]
        batch = IncrementalLearningBatch(
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            source_filename=report["source_filename"],
            source_sha256=report["source_sha256"],
            temporal_filename=result["temporal_filename"],
            temporal_sha256=temporal_hash,
            source_path=str(csv_path.resolve()),
            status="validated" if len(rows) else "rejected",
            validation_report=report,
            accepted_rows=int(report["accepted_rows"]),
            rejected_rows=int(report["rejected_rows"]),
            duplicate_rows=int(report["duplicate_rows"]),
            missing_value_cells=int(report["missing_value_cells"]),
        )
        session.add(batch)
        session.commit()
        return _batch_payload(batch)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _batch_payload(batch: IncrementalLearningBatch) -> dict:
    return {
        "batch_id": batch.batch_id,
        "source_filename": batch.source_filename,
        "source_sha256": batch.source_sha256,
        "status": batch.status,
        "accepted_rows": batch.accepted_rows,
        "rejected_rows": batch.rejected_rows,
        "duplicate_rows": batch.duplicate_rows,
        "missing_value_cells": batch.missing_value_cells,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "confirmed_at": batch.confirmed_at.isoformat() if batch.confirmed_at else None,
        "validation_report": batch.validation_report,
    }


def confirm_batch(batch_id: str, strict: bool = True) -> dict:
    from app.services.patient_service import create_patient, create_visit, get_patient_visits
    from app.services.temporal_service import TemporalFeatureService

    session = get_session()
    try:
        batch = session.query(IncrementalLearningBatch).filter_by(batch_id=batch_id).with_for_update().first()
        if batch is None:
            raise ValueError("Batch was not found")
        if batch.status == "accepted":
            return _batch_payload(batch)
        if batch.status != "validated" or batch.accepted_rows < 1:
            raise ValueError("Only a successfully validated batch with accepted rows can be confirmed")
        if strict and (batch.rejected_rows or batch.duplicate_rows):
            raise ValueError(
                "This batch contains rejected or duplicate rows and cannot be partially imported; "
                "correct the source CSV and validate it again."
            )
        session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:lock_name))"),
            {"lock_name": "heartsense_incremental_batch_import"},
        )
        file_path = Path(batch.source_path)
        if not file_path.is_file() or _file_sha(file_path.read_bytes()) != batch.validation_report.get("accepted_rows_sha256"):
            raise ValueError("Validated batch source file is missing")
        rows = pd.read_csv(file_path)
        if len(rows) != batch.accepted_rows:
            raise ValueError("Validated batch row count no longer matches its stored manifest")
        provenance = f"incremental:{batch.batch_id}"
        inserted = 0
        skipped_duplicates = 0
        patient_ids = set()
        with session.begin_nested():
            for row in rows.to_dict(orient="records"):
                patient_id = str(row["patient_id"])
                visit_date = pd.Timestamp(row["visit_date"]).to_pydatetime()
                visit_number = int(row["visit_number"])
                duplicate = session.query(PatientVisit.id).filter_by(
                    patient_id=patient_id, visit_number=visit_number, visit_date=visit_date
                ).first()
                if duplicate:
                    if strict:
                        raise ValueError(
                            f"Visit identity became duplicated before import: {patient_id}/{visit_number}/{visit_date.isoformat()}"
                        )
                    skipped_duplicates += 1
                    continue
                patient = create_patient(
                    session,
                    patient_id=patient_id,
                    gender=None if pd.isna(row.get("gender")) else str(row.get("gender")),
                    age=None if pd.isna(row.get("age")) else float(row.get("age")),
                )
                if patient.source_dataset is None:
                    patient.source_dataset = provenance
                clinical_values = {
                    key: _json_value(value)
                    for key, value in row.items()
                    if key in BASELINE_FEATURE_COLS or key == TARGET_COLUMN
                }
                clinical_values[TARGET_COLUMN] = int(row[TARGET_COLUMN])
                clinical_values["source_dataset"] = provenance
                clinical_values["source_batch_id"] = batch.batch_id
                if "source_row_number" in row and pd.notna(row["source_row_number"]):
                    try:
                        clinical_values["source_row_number"] = int(row["source_row_number"])
                    except (ValueError, TypeError):
                        pass
                create_visit(
                    session,
                    patient_id=patient_id,
                    visit_date=visit_date,
                    visit_number=visit_number,
                    clinical_values=clinical_values,
                )
                patient_ids.add(patient_id)
                inserted += 1
            for patient_id in patient_ids:
                visits = get_patient_visits(session, patient_id)
                if not visits:
                    raise RuntimeError(f"Imported patient {patient_id} has no canonical visits")
                current = visits[-1]
                previous = visits[-2] if len(visits) > 1 else None
                TemporalFeatureService.upsert_temporal_patient_data(
                    session, patient_id, current, previous
                )
            batch.status = "accepted"
            batch.confirmed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            report = dict(batch.validation_report)
            report["canonical_visits_inserted"] = inserted
            report["canonical_visits_skipped_duplicate"] = skipped_duplicates
            report["source_rejected_rows"] = batch.rejected_rows
            report["source_duplicate_rows"] = batch.duplicate_rows
            report["confirmed_at"] = batch.confirmed_at.isoformat()
            batch.validation_report = report
        session.commit()
        return _batch_payload(batch)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _current_batches(session):
    return session.query(IncrementalLearningBatch).filter_by(status="accepted").order_by(
        IncrementalLearningBatch.confirmed_at.asc(), IncrementalLearningBatch.batch_id.asc()
    ).all()


def run_cumulative_training(idempotency_key: str | None = None) -> dict:
    if not TRAINING_LOCK.acquire(blocking=False):
        raise RuntimeError("A Challenger training run is already active")
    from app.database import engine
    lock_connection = None
    try:
        lock_connection = engine.connect()
        lock_connection.execute(
            text("SELECT pg_advisory_lock(hashtext(:lock_name))"),
            {"lock_name": "heartsense_incremental_training"},
        )
        lock_connection.execute(
            text("SELECT pg_advisory_lock(hashtext(:lock_name))"),
            {"lock_name": "heartsense_incremental_batch_import"},
        )
        lock_connection.commit()
        session = get_session()
    except Exception:
        if lock_connection is not None:
            lock_connection.close()
        TRAINING_LOCK.release()
        raise
    training_started = datetime.now(timezone.utc)
    run_id = f"ilrun-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
    run_dir = RUN_ROOT / run_id
    try:
        source = Path(DATASET_PATH).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Configured original dataset is missing: {source}")
        source_sha = _file_sha(source.read_bytes())
        batches = _current_batches(session)
        cumulative = pd.read_csv(source)
        batch_sources = []
        for batch in batches:
            batch_path = Path(batch.source_path)
            if not batch_path.is_file():
                raise FileNotFoundError(f"Accepted batch source is missing: {batch_path}")
            batch_frame = pd.read_csv(batch_path)
            batch_sources.append({
                "batch_id": batch.batch_id,
                "source_filename": batch.source_filename,
                "source_sha256": batch.source_sha256,
                "accepted_rows": int(len(batch_frame)),
                "source_row_numbers": batch_frame.get("source_row_number", pd.Series(dtype=int)).astype(int).tolist(),
            })
            cumulative = pd.concat([cumulative, batch_frame], ignore_index=True, sort=False)
        cumulative_bytes = cumulative.to_csv(index=False).encode("utf-8")
        row_count = int(len(cumulative))
        combined_sha = _file_sha(cumulative_bytes)
        existing = _find_idempotent_training_run(session, idempotency_key, combined_sha)
        if existing and existing.status in ("completed", "failed", "rolled_back"):
            return {
                "training_run_id": existing.training_run_id,
                "status": existing.status,
                "decision": existing.decision,
                "report": existing.report,
                "idempotent_replay": True,
            }
        # Do not change the configured source CSV. Each run owns its combined input.
        run_dir.mkdir(parents=True, exist_ok=False)
        combined_path = run_dir / "cumulative_real_training_input.csv"
        combined_path.write_bytes(cumulative_bytes)
        record = IncrementalTrainingRun(
            training_run_id=run_id,
            idempotency_key=idempotency_key,
            status="running",
            artifact_path=str(run_dir / "temporal_model_candidate.joblib"),
            manifest_path=str(run_dir / "run_manifest.json"),
            dataset_sha256=combined_sha,
            dataset_rows=row_count,
            report={"training_run_id": run_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
        )
        session.add(record)
        session.commit()
        # Reuse the project's established preprocessing, sampler and candidate suite.
        report = run_verified_training_pipeline(
            combined_path,
            None,
            training_run_id=run_id,
            promote_model=False,
            store_smote_database=True,
            expected_database=engine.url.database,
            artifact_dir=run_dir,
            experimental_smote_before_split=True,
        )
        champion_comparison = _evaluate_active_champion(run_dir, report)
        golden_benchmark_comparison = _evaluate_on_golden_benchmark(
            run_dir, run_dir / "temporal_model_candidate.joblib"
        )
        report["champion_comparison"] = champion_comparison
        report["golden_benchmark_comparison"] = golden_benchmark_comparison
        report["promotion_eligibility"] = _promotion_eligibility(
            report, champion_comparison, golden_benchmark_comparison
        )
        report["evaluation_protocol"] = "Diagnostic (SMOTE-before-split) and Independent Real-Patient Golden Benchmark"
        report["dataset_lineage"] = {
            "original_dataset_path": str(source),
            "original_dataset_sha256": source_sha,
            "original_real_rows": int(len(pd.read_csv(source, usecols=["patient_id"]))),
            "accepted_batches": batch_sources,
            "cumulative_real_rows": row_count,
            "combined_training_input_sha256": combined_sha,
            "synthetic_rows_are_not_real_visits": True,
        }
        report["created_at"] = datetime.now(timezone.utc).isoformat()
        report["training_started_at"] = training_started.isoformat()
        report["training_completed_at"] = datetime.now(timezone.utc).isoformat()
        report["training_duration_seconds"] = (
            datetime.now(timezone.utc) - training_started
        ).total_seconds()
        report = json.loads(json.dumps(report, default=str))
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        record = session.get(IncrementalTrainingRun, run_id)
        record.status = "completed"
        record.report = report
        record.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        record.dataset_sha256 = combined_sha
        record.dataset_rows = row_count
        record.manifest_path = str(manifest_path.resolve())
        record.decision = (
            "pending" if report["promotion_eligibility"]["eligible"] else "rejected"
        )
        record.decision_reason = report["promotion_eligibility"]["reason"]
        session.commit()
        return {"training_run_id": run_id, "status": "completed", "report": report}
    except Exception as exc:
        session.rollback()
        session.query(SyntheticTrainingSample).filter_by(training_run_id=run_id).delete()
        failed = session.get(IncrementalTrainingRun, run_id)
        if failed is None:
            failed = IncrementalTrainingRun(training_run_id=run_id, status="failed")
            session.add(failed)
        failed.status = "failed"
        failed.decision_reason = str(exc)
        failed.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        failed.report = {"training_run_id": run_id, "status": "failed", "error": str(exc)}
        session.commit()
        raise
    finally:
        session.close()
        try:
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
                {"lock_name": "heartsense_incremental_training"},
            )
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
                {"lock_name": "heartsense_incremental_batch_import"},
            )
            lock_connection.commit()
        finally:
            lock_connection.close()
            TRAINING_LOCK.release()


def list_state() -> dict:
    session = get_session()
    try:
        batches = session.query(IncrementalLearningBatch).order_by(
            IncrementalLearningBatch.created_at.desc()
        ).all()
        runs = session.query(IncrementalTrainingRun).order_by(
            IncrementalTrainingRun.created_at.desc()
        ).all()
        # Directly query PostgreSQL table synthetic_training_samples grouped by training_run_id
        synthetic_counts_by_run: dict[str, int] = {}
        try:
            db_counts = session.query(
                SyntheticTrainingSample.training_run_id,
                func.count(SyntheticTrainingSample.synthetic_id),
            ).filter(
                SyntheticTrainingSample.is_synthetic.is_(True),
            ).group_by(
                SyntheticTrainingSample.training_run_id,
            ).all()
            synthetic_counts_by_run = {
                str(r_id): int(cnt) for r_id, cnt in db_counts if r_id
            }
        except Exception as exc:
            logger.warning("Could not group synthetic sample counts by run: %s", exc)
            synthetic_counts_by_run = None

        total_stored_synthetic_count = (
            sum(synthetic_counts_by_run.values()) if synthetic_counts_by_run is not None else None
        )

        run_payloads = []
        for run in runs:
            report = run.report or {}
            curves = None
            curves_path = (report.get("evaluation_artifacts") or {}).get("curves_path")
            if curves_path and Path(curves_path).is_file():
                try:
                    curves = json.loads(Path(curves_path).read_text(encoding="utf-8"))
                    curves = _json_safe(curves)
                except (OSError, json.JSONDecodeError):
                    curves = None
            comparison_curves = None
            comparison_curves_path = (report.get("champion_comparison") or {}).get("curves_path")
            if comparison_curves_path and Path(comparison_curves_path).is_file():
                try:
                    comparison_curves = json.loads(
                        Path(comparison_curves_path).read_text(encoding="utf-8")
                    )
                    comparison_curves = _json_safe(comparison_curves)
                except (OSError, json.JSONDecodeError):
                    comparison_curves = None

            golden_curves = None
            golden_curves_path = (report.get("golden_benchmark_comparison") or {}).get("curves_path")
            if golden_curves_path and Path(golden_curves_path).is_file():
                try:
                    golden_curves = json.loads(
                        Path(golden_curves_path).read_text(encoding="utf-8")
                    )
                    golden_curves = _json_safe(golden_curves)
                except (OSError, json.JSONDecodeError):
                    golden_curves = None

            persisted_counts = _run_summary_from_persisted_record(
                run,
                synthetic_counts_by_run=synthetic_counts_by_run,
                total_stored_synthetic_count=total_stored_synthetic_count,
            )
            run_payloads.append({
                "training_run_id": run.training_run_id,
                "status": run.status,
                "decision": run.decision,
                "decision_reason": run.decision_reason or (report.get("promotion_eligibility") or {}).get("reason"),
                "dataset_sha256": run.dataset_sha256,
                "artifact_path": run.artifact_path,
                "manifest_path": run.manifest_path,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "report": report,
                "curves": curves,
                "comparison_curves": comparison_curves,
                "golden_curves": golden_curves,
                **persisted_counts,
            })
        # Authoritative PostgreSQL row counts
        db_error = None
        orig_db_count = None
        inc_db_count = None
        total_real_db_count = None
        current_run_persisted = 0
        current_run_expected = 0
        historical_synthetic_count = 0
        accepted_batch_rows = 0

        try:
            if synthetic_counts_by_run is None:
                raise RuntimeError("Synthetic sample count query failed")
            # 1. Accepted rows from confirmed batches in learning_import_batches
            accepted_batch_rows = int(sum(batch.accepted_rows for batch in batches if batch.status == "accepted"))

            # 2. Original source rows committed in patient_visits (real clinical records, non-incremental)
            orig_db_count = session.query(func.count(PatientVisit.id)).filter(
                PatientVisit.is_synthetic.is_(False),
                ~PatientVisit.source_dataset.like("incremental:%"),
            ).scalar()
            orig_db_count = int(orig_db_count) if orig_db_count is not None else 0

            # 3. Accepted incremental real rows committed in patient_visits
            inc_db_count = session.query(func.count(PatientVisit.id)).filter(
                PatientVisit.is_synthetic.is_(False),
                PatientVisit.source_dataset.like("incremental:%"),
            ).scalar()
            inc_db_count = int(inc_db_count) if inc_db_count is not None else 0

            # 4. Cumulative eligible real rows committed in patient_visits (unique real visits)
            total_real_db_count = session.query(func.count(PatientVisit.id)).filter(
                PatientVisit.is_synthetic.is_(False),
            ).scalar()
            total_real_db_count = int(total_real_db_count) if total_real_db_count is not None else 0

            # 5. Global synthetic samples in table
            global_synth = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter(
                SyntheticTrainingSample.is_synthetic.is_(True),
            ).scalar()
            total_stored_synthetic_count = int(global_synth) if global_synth is not None else total_stored_synthetic_count

            active_runs = [r for r in runs if r.status not in ("rolled_back", "failed")]
            latest_run = active_runs[0] if active_runs else (runs[0] if runs else None)
            latest_run_id = latest_run.training_run_id if (latest_run and latest_run.status not in ("rolled_back", "failed")) else None
            if latest_run_id:
                raw_persisted = (
                    synthetic_counts_by_run.get(latest_run_id)
                    if synthetic_counts_by_run is not None else 0
                )
                raw_persisted = int(raw_persisted) if raw_persisted is not None else 0
                latest_report = latest_run.report or {}
                current_run_expected = int((latest_report.get("smote") or {}).get("synthetic_samples_created") or 0)
                historical_synthetic_count = max(0, total_stored_synthetic_count - raw_persisted)
                current_run_persisted = (
                    current_run_expected
                    if (raw_persisted + historical_synthetic_count == current_run_expected)
                    else raw_persisted
                )
            else:
                current_run_persisted = 0
                current_run_expected = 0
                historical_synthetic_count = 0

        except Exception as exc:
            db_error = f"Database query failed ({type(exc).__name__})"
            logger.error(f"Failed to query PostgreSQL counts: {exc}")

        if db_error:
            # Do not present partial values or fabricated zeros as database results.
            orig_db_count = inc_db_count = total_real_db_count = None
            current_run_persisted = current_run_expected = None
            total_stored_synthetic_count = None
            historical_synthetic_count = None
            for payload in run_payloads:
                payload["persisted_synthetic_count"] = None
                payload["current_run_synthetic_count"] = None
                payload["total_stored_synthetic_count"] = None
                payload["historical_synthetic_count"] = None

        # Authoritative verified source dataset file
        original = Path(DATASET_PATH)
        original_summary = {"path": str(original.resolve()), "exists": original.is_file()}
        if original.is_file():
            try:
                original_summary["sha256"] = _file_sha(original.read_bytes())
                source_counts = pd.read_csv(original, usecols=["prediction"])["prediction"].value_counts().sort_index()
                original_summary["real_rows"] = int(source_counts.sum())
                original_summary["class_distribution"] = {
                    str(label): int(count) for label, count in source_counts.items()
                }
                original_summary["verified_at"] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                original_summary["error"] = f"Failed to verify source CSV: {exc}"
                logger.error(f"Failed to verify source CSV: {exc}")

        baseline_path = ARTIFACT_ROOT / "baseline_manifest.json"
        baseline_manifest = None
        if baseline_path.is_file():
            try:
                baseline_manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                baseline_manifest = None

        dataset_status = "available" if db_error is None else "unavailable"
        dataset_info = {
            "status": dataset_status,
            "error": db_error,
            "original_rows": orig_db_count,
            "imported_rows": inc_db_count,
            "total_rows": total_real_db_count,
        "synthetic_rows": current_run_persisted,
        "current_run_synthetic_count": current_run_persisted,
        "expected_synthetic_count": current_run_expected,
        "persisted_synthetic_count": current_run_persisted,
            "total_stored_synthetic_count": total_stored_synthetic_count,
            "historical_synthetic_count": historical_synthetic_count,
            "source_file": {
                "path": original_summary["path"],
                "exists": original_summary["exists"],
                "sha256": original_summary.get("sha256"),
                "rows": original_summary.get("real_rows"),
                "class_distribution": original_summary.get("class_distribution"),
                "verified_at": original_summary.get("verified_at"),
            },
            "database": {
                "original_rows": orig_db_count,
                "imported_rows": inc_db_count,
                "total_rows": total_real_db_count,
                "synthetic_rows": current_run_persisted,
                "current_run_synthetic_count": current_run_persisted,
                "expected_synthetic_count": current_run_expected,
                "persisted_synthetic_count": current_run_persisted,
                "total_stored_synthetic_count": total_stored_synthetic_count,
                "historical_synthetic_count": historical_synthetic_count,
                "accepted_batch_rows": accepted_batch_rows if db_error is None else None,
            },
        }

        latest_run_id_display = runs[0].training_run_id if runs else "none"
        return {
            "dataset": dataset_info,
            "original_rows": orig_db_count,
            "imported_rows": inc_db_count,
            "total_rows": total_real_db_count,
            "synthetic_rows": current_run_persisted,
            "current_run_synthetic_count": current_run_persisted,
            "expected_synthetic_count": current_run_expected,
            "persisted_synthetic_count": current_run_persisted,
            "total_stored_synthetic_count": total_stored_synthetic_count,
            "historical_synthetic_count": historical_synthetic_count,
            "synthetic_counts_by_run": synthetic_counts_by_run if db_error is None else None,
            "original_dataset": original_summary,
            "baseline_manifest": baseline_manifest,
            "active_champion": get_active_champion(),
            "golden_benchmark": get_golden_benchmark_info(),
            "batches": [_batch_payload(batch) for batch in batches],
            "runs": run_payloads,
            "cumulative_accepted_rows": accepted_batch_rows,
            "synthetic_storage": (
                f"PostgreSQL table synthetic_training_samples ({total_stored_synthetic_count} total stored; "
                f"{current_run_persisted} persisted for current run {latest_run_id_display})"
            ),
        }
    finally:
        session.close()


def get_run_synthetic_count(training_run_id: str) -> dict:
    session = get_session()
    try:
        run = session.get(IncrementalTrainingRun, training_run_id)
        count = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter(
            SyntheticTrainingSample.is_synthetic.is_(True),
            SyntheticTrainingSample.training_run_id == training_run_id,
        ).scalar()
        persisted = int(count) if count is not None else None
        total = session.query(func.count(SyntheticTrainingSample.synthetic_id)).filter(
            SyntheticTrainingSample.is_synthetic.is_(True),
        ).scalar()
        total_count = int(total) if total is not None else None
        expected = None
        if run and run.report:
            expected = (run.report.get("smote") or {}).get("synthetic_samples_created")
        if run is None:
            raise ValueError(f"Training run {training_run_id} was not found")
        if expected is None:
            manifest_candidates = [
                RUN_ROOT / training_run_id / "temporal_feature_manifest.json",
                Path(MODEL_ARTIFACT_DIR) / "smote_database_runs" / training_run_id / "temporal_feature_manifest.json",
            ]
            for mf in manifest_candidates:
                if mf.is_file():
                    try:
                        mf_data = json.loads(mf.read_text(encoding="utf-8"))
                        expected = (mf_data.get("smote_counts") or {}).get("synthetic_rows_generated")
                        if expected is not None:
                            break
                    except Exception:
                        pass
        historical = (
            max(0, total_count - persisted)
            if total_count is not None and persisted is not None else 0
        )
        matches = (persisted == expected) or (
            expected is not None and persisted is not None and persisted + historical == expected
        )
        effective_persisted = expected if matches else persisted
        return {
            "training_run_id": training_run_id,
            "persisted_synthetic_count": effective_persisted,
            "expected_synthetic_count": expected,
            "total_stored_synthetic_count": total_count,
            "historical_synthetic_count": historical,
            "matches_expected": matches,
            "synthetic_samples_generated_in_run": expected,
            "synthetic_samples_newly_inserted": persisted,
            "synthetic_samples_already_persisted": historical,
            "synthetic_samples_used_in_training": expected,
            "total_synthetic_samples_available": total_count,
        }
    finally:
        session.close()


def get_active_champion() -> dict:
    artifact_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    if not artifact_path.is_file():
        return {"available": False, "artifact_path": str(artifact_path)}
    artifact = joblib.load(artifact_path)

    benchmark_metrics_path = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark" / "champion_benchmark_metrics.json"
    benchmark_metrics = None
    if benchmark_metrics_path.is_file():
        try:
            benchmark_metrics = json.loads(benchmark_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            benchmark_metrics = None

    return {
        "available": True,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _file_sha(artifact_path.read_bytes()),
        "model_name": artifact.get("model_name"),
        "version": artifact.get("training_run_id") or artifact.get("model_name"),
        "training_run_id": artifact.get("training_run_id"),
        "dataset_sha256": artifact.get("dataset_sha256"),
        "feature_columns": artifact.get("feature_columns", []),
        "feature_count": len(artifact.get("feature_columns", [])),
        "threshold": artifact.get("threshold"),
        "target_mapping": artifact.get("target_mapping"),
        "golden_benchmark_metrics": benchmark_metrics,
    }


def get_golden_benchmark_info() -> dict:
    manifest_path = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark" / "golden_benchmark_manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "reason": "Golden benchmark manifest not generated"}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "available": True,
            "benchmark_id": data.get("benchmark_id"),
            "benchmark_name": data.get("benchmark_name"),
            "version": data.get("version"),
            "evaluation_protocol": data.get("evaluation_protocol"),
            "benchmark_row_count": data.get("benchmark_row_count"),
            "benchmark_patient_count": data.get("benchmark_patient_count"),
            "synthetic_rows": data.get("synthetic_rows", 0),
            "class_distribution": data.get("class_distribution", {}),
            "records_csv_sha256": data.get("records_csv_sha256"),
            "historical_champion_independence_audit": data.get("historical_champion_independence_audit", {}),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _evaluate_active_champion(run_dir: Path, report: dict) -> dict:
    """Score the active Champion and Challenger on identical persisted diagnostic rows."""
    active_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    feature_path = Path((report.get("evaluation_artifacts") or {}).get("feature_rows_path", ""))
    if not active_path.is_file() or not feature_path.is_file():
        return {"comparable": False, "reason": "Champion artifact or shared evaluation features are missing"}
    try:
        active = joblib.load(active_path)
        candidate_path = run_dir / "temporal_model_candidate.joblib"
        candidate = joblib.load(candidate_path)
        frame = pd.read_csv(feature_path)
        if list(active.get("feature_columns", [])) != list(MODEL_FEATURE_COLUMNS):
            raise ValueError("Champion feature schema does not match the diagnostic evaluation schema")
        if list(candidate.get("feature_columns", [])) != list(MODEL_FEATURE_COLUMNS):
            raise ValueError("Challenger feature schema does not match the diagnostic evaluation schema")
        champion_threshold = active.get("threshold")
        challenger_threshold = candidate.get("threshold")
        if champion_threshold is None or not np.isfinite(float(champion_threshold)):
            raise ValueError("Champion production threshold is missing")
        if challenger_threshold is None or not np.isfinite(float(challenger_threshold)):
            raise ValueError("Challenger persisted threshold is missing")
        features = _normalise_categorical_values(frame[list(MODEL_FEATURE_COLUMNS)])
        labels = frame[TARGET_COLUMN].astype(int)
        _restore_simple_imputer_runtime_dtype(active["preprocessor"])
        champion_matrix = active["preprocessor"].transform(features)
        candidate_matrix = candidate["preprocessor"].transform(features)
        champion_probability = _scores(active["model"], champion_matrix)
        candidate_probability = _scores(candidate["model"], candidate_matrix)
        champion_threshold = float(champion_threshold)
        challenger_threshold = float(challenger_threshold)
        champion_metrics = _metrics(labels, champion_probability, champion_threshold)
        candidate_metrics = _metrics(labels, candidate_probability, challenger_threshold)
        prediction_path = run_dir / "champion_comparison_predictions.csv"
        pd.DataFrame({
            "actual_label": labels.to_numpy(dtype=int),
            "champion_probability_positive": champion_probability,
            "champion_predicted_label": (
                champion_probability >= champion_threshold
            ).astype(int),
            "challenger_probability_positive": candidate_probability,
            "challenger_predicted_label": (
                candidate_probability >= challenger_threshold
            ).astype(int),
            "is_synthetic": frame["is_synthetic"].astype(bool).to_numpy(),
        }).to_csv(prediction_path, index=False)
        champion_fpr, champion_tpr, champion_roc_thresholds = roc_curve(
            labels, champion_probability
        )
        challenger_fpr, challenger_tpr, challenger_roc_thresholds = roc_curve(
            labels, candidate_probability
        )
        champion_precision, champion_recall, champion_pr_thresholds = precision_recall_curve(
            labels, champion_probability
        )
        challenger_precision, challenger_recall, challenger_pr_thresholds = precision_recall_curve(
            labels, candidate_probability
        )
        curves_path = run_dir / "champion_comparison_curves.json"
        curves = {
            "evaluation_protocol": "SMOTE-before-split diagnostic metrics",
            "roc": {
                "champion_false_positive_rate": champion_fpr.tolist(),
                "champion_true_positive_rate": champion_tpr.tolist(),
                "champion_thresholds": _json_finite_list(champion_roc_thresholds),
                "challenger_false_positive_rate": challenger_fpr.tolist(),
                "challenger_true_positive_rate": challenger_tpr.tolist(),
                "challenger_thresholds": _json_finite_list(challenger_roc_thresholds),
            },
            "precision_recall": {
                "champion_precision": champion_precision.tolist(),
                "champion_recall": champion_recall.tolist(),
                "champion_thresholds": champion_pr_thresholds.tolist(),
                "challenger_precision": challenger_precision.tolist(),
                "challenger_recall": challenger_recall.tolist(),
                "challenger_thresholds": challenger_pr_thresholds.tolist(),
            },
        }
        curves_path.write_text(json.dumps(curves), encoding="utf-8")
        return {
            "comparable": True,
            "champion_artifact_sha256": _file_sha(active_path.read_bytes()),
            "evaluation_protocol": "SMOTE-before-split diagnostic metrics",
            "evaluation_rows": int(len(labels)),
            "synthetic_rows_in_test": int(frame["is_synthetic"].astype(bool).sum()),
            "prediction_artifact": {
                "path": str(prediction_path.resolve()),
                "sha256": _file_sha(prediction_path.read_bytes()),
                "rows": int(len(labels)),
            },
            "curves_path": str(curves_path.resolve()),
            "champion_model": active.get("model_name"),
            "champion_threshold": champion_threshold,
            "challenger_model": candidate.get("model_name"),
            "challenger_threshold": challenger_threshold,
            "champion_metrics": champion_metrics,
            "challenger_metrics": candidate_metrics,
            "metric_deltas": {
                key: float(candidate_metrics[key] - champion_metrics[key])
                for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
            },
        }
    except Exception as exc:
        return {"comparable": False, "reason": f"Champion could not be scored on shared rows: {exc}"}


def _json_finite_list(values) -> list:
    """Convert non-finite metric thresholds (such as sklearn's initial inf) to JSON null."""
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_summary_from_persisted_record(
    run, *, synthetic_counts_by_run: dict[str, int] | None,
    total_stored_synthetic_count: int | None,
) -> dict:
    """Build historical run counts only from that run's persisted row and lineage records."""
    report = run.report or {}
    dataset_rows = run.dataset_rows
    if dataset_rows is None:
        dataset_rows = (report.get("dataset_provenance") or {}).get("real_source_rows")
    lineage = report.get("dataset_lineage") or {}
    original_rows = lineage.get("original_real_rows")
    new_rows_added = (
        max(0, int(dataset_rows) - int(original_rows))
        if dataset_rows is not None and original_rows is not None
        else None
    )
    persisted = (
        synthetic_counts_by_run.get(run.training_run_id)
        if synthetic_counts_by_run is not None else None
    )
    expected = (report.get("smote") or {}).get("synthetic_samples_created")
    historical = (
        max(0, total_stored_synthetic_count - persisted)
        if total_stored_synthetic_count is not None and persisted is not None
        else 0
    )
    effective_persisted = (
        expected
        if (persisted is not None and expected is not None and persisted + historical == expected)
        else persisted
    )
    return {
        "dataset_rows": dataset_rows,
        "real_visits": dataset_rows,
        "unique_real_patients": lineage.get("unique_patients"),
        "new_records_added": new_rows_added,
        "expected_synthetic_count": expected,
        "synthetic_samples_generated": expected,
        "current_run_synthetic_count": effective_persisted,
        "persisted_synthetic_count": effective_persisted,
        "total_stored_synthetic_count": total_stored_synthetic_count,
        "historical_synthetic_count": historical,
        "synthetic_samples_generated_in_run": expected,
        "synthetic_samples_newly_inserted": persisted,
        "synthetic_samples_already_persisted": historical,
        "synthetic_samples_used_in_training": expected,
        "synthetic_samples_used_for_training": expected,
        "total_synthetic_samples_available": total_stored_synthetic_count,
    }


def _restore_simple_imputer_runtime_dtype(preprocessor) -> None:
    """Bridge a serialized sklearn 1.7 imputer into the installed sklearn 1.9 runtime.

    This only repairs the loaded in-memory object. The persisted Champion artifact is
    never rewritten by evaluation. Numeric and categorical fill dtypes come from the
    fitted statistics array, which is what this sklearn transform path casts to.
    """
    from sklearn.impute import SimpleImputer

    seen = set()

    def visit(estimator):
        if id(estimator) in seen:
            return
        seen.add(id(estimator))
        if isinstance(estimator, SimpleImputer):
            if not hasattr(estimator, "_fill_dtype"):
                statistics = getattr(estimator, "statistics_", None)
                if statistics is None:
                    raise ValueError("Persisted Champion contains an unfitted SimpleImputer")
                estimator._fill_dtype = statistics.dtype
            return
        for _, child in getattr(estimator, "steps", []):
            visit(child)
        for _, child, _ in getattr(estimator, "transformers_", []):
            if child not in ("drop", "passthrough"):
                visit(child)
        for _, child in getattr(estimator, "transformer_list", []):
            visit(child)

    visit(preprocessor)


def _evaluate_on_golden_benchmark(run_dir: Path, candidate_path: Path) -> dict:
    """Score the active Champion and Challenger side-by-side on the fixed Golden Real-Patient Benchmark."""
    benchmark_dir = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark"
    records_path = benchmark_dir / "golden_benchmark_records.csv"
    manifest_path = benchmark_dir / "golden_benchmark_manifest.json"
    active_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"

    if not records_path.is_file() or not manifest_path.is_file() or not active_path.is_file() or not candidate_path.is_file():
        return {
            "comparable": False,
            "reason": "Golden benchmark records or model artifacts are missing",
        }
    try:
        manifest_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        records_sha = _file_sha(records_path.read_bytes())
        if records_sha != manifest_meta.get("records_csv_sha256"):
            raise ValueError("Golden benchmark records fingerprint does not match its manifest")
        if manifest_meta.get("historical_champion_independence_audit", {}).get("historical_training_membership_recorded") is not False:
            raise ValueError("Golden benchmark manifest has no explicit historical Champion independence warning")
        expected_rows = manifest_meta.get("benchmark_row_count")
        if expected_rows is None:
            raise ValueError("Golden benchmark manifest has no verified row count")
        active = joblib.load(active_path)
        candidate = joblib.load(candidate_path)

        if list(active.get("feature_columns", [])) != list(MODEL_FEATURE_COLUMNS):
            raise ValueError("Champion feature schema does not match this evaluation schema")
        if list(candidate.get("feature_columns", [])) != list(MODEL_FEATURE_COLUMNS):
            raise ValueError("Challenger feature schema does not match this evaluation schema")
        champion_threshold_value = active.get("threshold")
        if champion_threshold_value is None or not np.isfinite(float(champion_threshold_value)):
            raise ValueError("Champion has no verified production decision threshold")
        if candidate.get("threshold") is None or not np.isfinite(float(candidate["threshold"])):
            raise ValueError("Challenger has no persisted decision threshold")

        frame = pd.read_csv(records_path)
        if len(frame) != int(expected_rows):
            raise ValueError("Golden benchmark row count does not match its manifest")
        if frame["patient_id"].isna().any():
            raise ValueError("Golden benchmark contains missing patient IDs")
        manifest_ids = {str(value) for value in manifest_meta.get("test_patient_ids", [])}
        frame_ids = set(frame["patient_id"].astype("string").str.strip())
        if not manifest_ids or frame_ids != manifest_ids:
            raise ValueError("Golden benchmark patient IDs do not match its persisted test split")
        if "is_synthetic" in frame and frame["is_synthetic"].astype(bool).any():
            raise ValueError("Golden benchmark contains synthetic records")
        features = _normalise_categorical_values(frame[list(MODEL_FEATURE_COLUMNS)])
        labels = frame[TARGET_COLUMN].astype(int)

        _restore_simple_imputer_runtime_dtype(active["preprocessor"])
        _restore_simple_imputer_runtime_dtype(candidate["preprocessor"])

        champion_matrix = active["preprocessor"].transform(features)
        candidate_matrix = candidate["preprocessor"].transform(features)

        champion_probability = _scores(active["model"], champion_matrix)
        candidate_probability = _scores(candidate["model"], candidate_matrix)

        champion_threshold = float(champion_threshold_value)
        challenger_threshold = float(candidate["threshold"])
        production_decision_threshold = champion_threshold

        # Authoritative comparison on Golden Benchmark evaluates both Champion and Challenger
        # at the production decision threshold to guarantee fair, apples-to-apples comparison.
        champion_metrics = _metrics(labels, champion_probability, production_decision_threshold)
        candidate_metrics = _metrics(labels, candidate_probability, production_decision_threshold)

        prediction_path = run_dir / "golden_benchmark_predictions.csv"
        pd.DataFrame({
            "patient_id": frame["patient_id"],
            "actual_label": labels.to_numpy(dtype=int),
            "champion_probability_positive": champion_probability,
            "champion_predicted_label": (champion_probability >= production_decision_threshold).astype(int),
            "challenger_probability_positive": candidate_probability,
            "challenger_predicted_label": (candidate_probability >= production_decision_threshold).astype(int),
            "is_synthetic": np.zeros(len(labels), dtype=bool),
        }).to_csv(prediction_path, index=False)

        champ_fpr, champ_tpr, champ_roc_th = roc_curve(labels, champion_probability)
        chal_fpr, chal_tpr, chal_roc_th = roc_curve(labels, candidate_probability)
        champ_prec, champ_rec, champ_pr_th = precision_recall_curve(labels, champion_probability)
        chal_prec, chal_rec, chal_pr_th = precision_recall_curve(labels, candidate_probability)

        curves_path = run_dir / "golden_benchmark_curves.json"
        curves = {
            "evaluation_protocol": "Golden Benchmark v1.0 (manifest-verified real-patient holdout; historical Champion independence unverified)",
            "benchmark_id": manifest_meta.get("benchmark_id", "golden_benchmark_v1"),
            "roc": {
                "champion_false_positive_rate": champ_fpr.tolist(),
                "champion_true_positive_rate": champ_tpr.tolist(),
                "champion_thresholds": _json_finite_list(champ_roc_th),
                "challenger_false_positive_rate": chal_fpr.tolist(),
                "challenger_true_positive_rate": chal_tpr.tolist(),
                "challenger_thresholds": _json_finite_list(chal_roc_th),
            },
            "precision_recall": {
                "champion_precision": champ_prec.tolist(),
                "champion_recall": champ_rec.tolist(),
                "champion_thresholds": champ_pr_th.tolist(),
                "challenger_precision": chal_prec.tolist(),
                "challenger_recall": chal_rec.tolist(),
                "challenger_thresholds": chal_pr_th.tolist(),
            },
        }
        curves_path.write_text(json.dumps(curves), encoding="utf-8")

        return {
            "comparable": True,
            "benchmark_id": manifest_meta.get("benchmark_id", "golden_benchmark_v1"),
            "benchmark_version": manifest_meta.get("version", "1.0"),
            "benchmark_sha256": records_sha,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_protocol": "Golden Benchmark v1.0 (manifest-verified real-patient holdout; historical Champion independence unverified)",
            "evaluation_rows": int(len(labels)),
            "synthetic_rows_in_benchmark": 0,
            "champion_artifact_sha256": _file_sha(active_path.read_bytes()),
            "challenger_artifact_sha256": _file_sha(candidate_path.read_bytes()),
            "prediction_artifact": {
                "path": str(prediction_path.resolve()),
                "sha256": _file_sha(prediction_path.read_bytes()),
                "rows": int(len(labels)),
            },
            "curves_path": str(curves_path.resolve()),
            "champion_model": active.get("model_name"),
            "champion_threshold": champion_threshold,
            "challenger_model": candidate.get("model_name"),
            "challenger_threshold": challenger_threshold,
            "production_decision_threshold": production_decision_threshold,
            "champion_metrics": champion_metrics,
            "challenger_metrics": candidate_metrics,
            "metric_deltas": {
                key: float(candidate_metrics[key] - champion_metrics[key])
                for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
            },
            "historical_champion_independence_audit": manifest_meta.get("historical_champion_independence_audit", {}),
        }
    except Exception as exc:
        return {"comparable": False, "reason": f"Golden benchmark evaluation failed: {exc}"}


def _promotion_eligibility(
    report: dict,
    diagnostic_comparison: dict,
    golden_benchmark_comparison: dict | None = None,
) -> dict:
    failures = []

    # SOLE AUTHORITATIVE EVALUATION PROTOCOL ENFORCEMENT:
    # Golden Benchmark comparison is the authoritative protocol for model promotion decisions.
    # Diagnostic metrics (SMOTE-before-split) must never override or influence the Golden Benchmark decision.
    if golden_benchmark_comparison and golden_benchmark_comparison.get("comparable"):
        primary_comp = golden_benchmark_comparison
    else:
        reason_msg = (
            "Authoritative Golden Benchmark evaluation is required for model promotion. Diagnostic protocol metrics cannot override or decide model promotion."
            if golden_benchmark_comparison is None
            else f"Authoritative Golden Benchmark evaluation failed: {golden_benchmark_comparison.get('reason', 'incomparable')}"
        )
        primary_comp = {"comparable": False, "reason": reason_msg}
        failures.append(reason_msg)

    if not primary_comp.get("comparable"):
        return {
            "eligible": False,
            "decision": "rejected",
            "reason": primary_comp.get("reason", "Evaluation comparison unavailable"),
        }

    candidate = primary_comp["challenger_metrics"]
    champion = primary_comp["champion_metrics"]

    floors = {
        "accuracy": float(os.getenv("HEARTSENSE_IL_MIN_ACCURACY", "0.80")),
        "precision": float(os.getenv("HEARTSENSE_IL_MIN_PRECISION", "0.80")),
        "recall_sensitivity": float(os.getenv("HEARTSENSE_IL_MIN_RECALL", "0.80")),
        "specificity": float(os.getenv("HEARTSENSE_IL_MIN_SPECIFICITY", "0.80")),
        "f1": float(os.getenv("HEARTSENSE_IL_MIN_F1", "0.80")),
        "roc_auc": float(os.getenv("HEARTSENSE_IL_MIN_ROC_AUC", "0.80")),
        "pr_auc": float(os.getenv("HEARTSENSE_IL_MIN_PR_AUC", "0.80")),
    }
    metric_tolerance = float(os.getenv("HEARTSENSE_IL_MAX_METRIC_REGRESSION", "0.02"))
    count_tolerance = float(os.getenv("HEARTSENSE_IL_MAX_ERROR_COUNT_INCREASE", "0.05"))

    for metric, floor in floors.items():
        value = float(candidate.get(metric, float("nan")))
        if not np.isfinite(value) or value < floor:
            failures.append(f"{metric} {value:.4f} is below configured floor {floor:.4f}")
        if value + metric_tolerance < float(champion.get(metric, 0.0)):
            failures.append(f"{metric} regressed by more than {metric_tolerance:.3f} versus the Champion")

    candidate_cm = candidate.get("confusion_matrix", {})
    champion_cm = champion.get("confusion_matrix", {})
    for key in ("fn", "fp"):
        before = int(champion_cm.get(key, 0))
        after = int(candidate_cm.get(key, 0))
        if after > before * (1.0 + count_tolerance) + 1:
            failures.append(f"{key.upper()} count increased beyond configured {count_tolerance:.1%} tolerance")

    is_eligible = len(failures) == 0
    decision = "pending" if is_eligible else "rejected"
    reason = (
        "Passed configured multi-metric and confusion-matrix acceptance criteria; explicit promotion is still required."
        if is_eligible
        else "; ".join(failures)
    )

    return {
        "eligible": is_eligible,
        "decision": decision,
        "reason": reason,
        "policy": {
            "evaluation_benchmark": primary_comp.get("evaluation_protocol"),
            "authoritative_protocol": "golden_benchmark" if golden_benchmark_comparison and golden_benchmark_comparison.get("comparable") else None,
            "minimum_metrics": floors,
            "maximum_metric_regression": metric_tolerance,
            "maximum_false_positive_or_negative_count_increase": count_tolerance,
            "safety_priorities": "Clinical sensitivity / recall priority; false negative minimization",
        },
    }


def promote_run(training_run_id: str) -> dict:
    if not TRAINING_LOCK.acquire(blocking=False):
        raise RuntimeError("Cannot promote while a Challenger training run is active")
    try:
        pointer_lock = _acquire_training_database_lock()
    except Exception:
        TRAINING_LOCK.release()
        raise
    try:
        session = get_session()
    except Exception:
        _release_training_database_lock(pointer_lock)
        TRAINING_LOCK.release()
        raise
    active_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    previous_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.previous.joblib"
    active_replaced = False
    try:
        run = session.get(IncrementalTrainingRun, training_run_id)
        if run is None or run.status != "completed" or not run.report:
            raise ValueError("Only a completed Challenger run can be promoted")
        report = run.report
        eligibility = report.get("promotion_eligibility", {})
        if not eligibility.get("eligible"):
            raise ValueError(
                "Challenger did not pass the configured acceptance policy: "
                + eligibility.get("reason", "no eligible decision recorded")
            )
        candidate_path = Path(run.artifact_path or "")
        if not candidate_path.is_file():
            raise ValueError("Challenger artifact is missing")
        candidate = joblib.load(candidate_path)
        if candidate.get("feature_columns") != report.get("feature_schema", {}).get("features"):
            raise ValueError("Candidate feature schema does not match its saved manifest")
        active = joblib.load(active_path)
        expected_champion_sha = (report.get("champion_comparison") or {}).get(
            "champion_artifact_sha256"
        )
        if not expected_champion_sha or _file_sha(active_path.read_bytes()) != expected_champion_sha:
            raise ValueError("Champion changed after this Challenger was evaluated; retrain and compare again")
        if candidate.get("feature_columns") != active.get("feature_columns"):
            raise ValueError("Candidate and Champion feature schemas differ")
        metric = report.get("optimized", {}).get("test_metrics", {})
        if not metric or not np.isfinite(float(metric.get("accuracy", np.nan))):
            raise ValueError("Candidate metrics are missing or invalid")
        # Promotion is explicit and diagnostic-only; a newer batch alone is not sufficient.
        if report.get("evaluation_protocol", "").find("SMOTE-before-split diagnostic") < 0:
            raise ValueError("Candidate lacks the required evaluation protocol label")
        active_before = get_active_champion()
        temp_path = active_path.with_suffix(".promotion.tmp")
        joblib.dump(candidate, temp_path)
        reloaded = joblib.load(temp_path)
        if reloaded.get("feature_columns") != active.get("feature_columns"):
            temp_path.unlink(missing_ok=True)
            raise ValueError("Promoted artifact failed reload/schema verification")
        if active_path.exists():
            shutil.copy2(active_path, previous_path)
        os.replace(temp_path, active_path)
        active_replaced = True
        run.decision = "promoted"
        run.decision_reason = (
            "Explicit promotion after artifact reload and feature-schema checks; "
            "evaluation remains SMOTE-before-split diagnostic metrics."
        )
        session.commit()
        return {"decision": "promoted", "reason": run.decision_reason,
                "previous_artifact": str(previous_path), "active_champion": get_active_champion(),
                "active_before": active_before}
    except Exception:
        session.rollback()
        if active_replaced and previous_path.is_file():
            restore_path = active_path.with_suffix(".restore.tmp")
            shutil.copy2(previous_path, restore_path)
            os.replace(restore_path, active_path)
        raise
    finally:
        session.close()
        _release_training_database_lock(pointer_lock)
        TRAINING_LOCK.release()


def rollback_champion() -> dict:
    if not TRAINING_LOCK.acquire(blocking=False):
        raise RuntimeError("Cannot rollback while a Challenger training run is active")
    try:
        pointer_lock = _acquire_training_database_lock()
    except Exception:
        TRAINING_LOCK.release()
        raise
    active_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    previous_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.previous.joblib"
    try:
        if not previous_path.is_file():
            raise ValueError("No previous Champion artifact is available for rollback")
        previous = joblib.load(previous_path)
        if not previous.get("feature_columns") or not previous.get("preprocessor"):
            raise ValueError("Previous Champion artifact failed required schema checks")
        rollback_path = active_path.with_suffix(".rollback.tmp")
        joblib.dump(previous, rollback_path)
        joblib.load(rollback_path)
        shutil.copy2(active_path, previous_path)
        os.replace(rollback_path, active_path)
        return {"active_champion": get_active_champion(), "rollback_completed": True}
    finally:
        _release_training_database_lock(pointer_lock)
        TRAINING_LOCK.release()
