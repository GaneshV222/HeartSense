"""Transactional, source-scoped ingestion of the configured visit dataset."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import DATASET_PATH
from app.database import engine
from app.services.temporal_service import (
    CATEGORICAL_VARIABLES,
    NUMERICAL_VARIABLES,
    TemporalFeatureService,
)
from app.ml.verified_pipeline import load_and_validate_dataset

SOURCE_FEATURES = (
    "age", "gender", "bmi", "chest_pain_type", "systolic_bp", "diastolic_bp",
    "resting_heart_rate", "max_heart_rate", "cholesterol", "hdl", "ldl",
    "fasting_blood_sugar", "hba1c", "diabetes", "resting_ecg", "exercise_angina",
    "oldpeak", "st_slope", "num_major_vessels", "thalassemia", "smoking",
    "family_history", "physical_activity", "stress_level",
)
VISIT_COLUMNS = (
    "patient_id", "visit_number", "visit_date", "visit_timestamp", *SOURCE_FEATURES,
    "smoking_status", "prediction", "is_synthetic", "source_dataset",
    "source_row_number", "raw_data", "created_at", "updated_at",
)


def _clean(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def inspect_dataset(path: str = DATASET_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the complete source using the project's canonical validator."""
    frame, report = load_and_validate_dataset(path)
    stats = {
        "path": report["path"],
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_patient_ids": int(frame["patient_id"].nunique()),
        "duplicate_visit_keys": int(frame.duplicated(["patient_id", "visit_date", "visit_number"]).sum()),
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
        "target_distribution": {
            str(k): int(v) for k, v in frame["prediction"].value_counts().sort_index().items()
        },
        "validation": report,
    }
    return frame, stats


def _chunks(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def ingest_csv(path: str = DATASET_PATH, batch_size: int = 2000) -> dict[str, Any]:
    """Replace only this source's genuine patient, visit, and temporal rows atomically.

    Ownership and foreign-key checks run inside the transaction before any delete.
    If rows cannot be attributed safely, the transaction is rolled back.
    """
    frame, report = inspect_dataset(path)
    source = Path(path).name
    now = datetime.utcnow()
    patient_ids = frame["patient_id"].astype(str).drop_duplicates().tolist()
    if not patient_ids:
        raise ValueError("Refusing to replace dataset records with an empty source file.")

    counts = Counter()
    with engine.begin() as conn:
        # Refuse cross-source identity collisions; those patients are not ours to overwrite.
        collisions = conn.execute(text("""
            SELECT patient_id FROM patients
            WHERE patient_id = ANY(:ids) AND source_dataset IS DISTINCT FROM :source
            LIMIT 5
        """), {"ids": patient_ids, "source": source}).scalars().all()
        if collisions:
            raise RuntimeError(f"Dataset patient IDs collide with unowned patients: {collisions}")

        # A visit replacement changes visit primary keys. Protect every existing prediction
        # reference and preserve any independently created visits or temporal rows.
        linked_predictions = conn.execute(text("""
            SELECT COUNT(*) FROM predictions p
            JOIN patient_visits v ON v.id = p.visit_id
            WHERE v.source_dataset = :source
        """), {"source": source}).scalar_one()
        if linked_predictions:
            raise RuntimeError(
                f"Cannot replace {source}: {linked_predictions} predictions reference its visits."
            )
        unowned_visits = conn.execute(text("""
            SELECT COUNT(*) FROM patient_visits
            WHERE patient_id = ANY(:ids) AND source_dataset IS DISTINCT FROM :source
        """), {"ids": patient_ids, "source": source}).scalar_one()
        unowned_temporal = conn.execute(text("""
            SELECT COUNT(*) FROM temporal_patient_data
            WHERE patient_id = ANY(:ids) AND source_dataset IS DISTINCT FROM :source
        """), {"ids": patient_ids, "source": source}).scalar_one()
        if unowned_visits or unowned_temporal:
            raise RuntimeError(
                "Cannot safely refresh dataset patients that also have independently owned "
                f"history (visits={unowned_visits}, temporal={unowned_temporal})."
            )

        existing_source_patients = conn.execute(text(
            "SELECT patient_id FROM patients WHERE source_dataset = :source"
        ), {"source": source}).scalars().all()
        stale_ids = sorted(set(existing_source_patients) - set(patient_ids))
        if stale_ids:
            stale_predictions = conn.execute(text(
                "SELECT COUNT(*) FROM predictions WHERE patient_id = ANY(:ids)"
            ), {"ids": stale_ids}).scalar_one()
            stale_unowned_visits = conn.execute(text("""
                SELECT COUNT(*) FROM patient_visits
                WHERE patient_id = ANY(:ids) AND source_dataset IS DISTINCT FROM :source
            """), {"ids": stale_ids, "source": source}).scalar_one()
            stale_unowned_temporal = conn.execute(text("""
                SELECT COUNT(*) FROM temporal_patient_data
                WHERE patient_id = ANY(:ids) AND source_dataset IS DISTINCT FROM :source
            """), {"ids": stale_ids, "source": source}).scalar_one()
            if stale_predictions or stale_unowned_visits or stale_unowned_temporal:
                raise RuntimeError(
                    "Cannot safely remove stale source patients because unrelated records "
                    f"reference them (predictions={stale_predictions}, visits={stale_unowned_visits}, "
                    f"temporal={stale_unowned_temporal})."
                )

        synthetic_source_rows = conn.execute(text("""
            SELECT COUNT(*) FROM patient_visits
            WHERE source_dataset = :source AND is_synthetic IS TRUE
        """), {"source": source}).scalar_one()
        if synthetic_source_rows:
            raise RuntimeError(
                f"Refusing to replace {synthetic_source_rows} synthetic rows marked as source data."
            )

        counts["patients_before"] = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar_one()
        counts["visits_before"] = conn.execute(text("SELECT COUNT(*) FROM patient_visits")).scalar_one()
        counts["temporal_before"] = conn.execute(text("SELECT COUNT(*) FROM temporal_patient_data")).scalar_one()
        existing_target = set(conn.execute(text(
            "SELECT patient_id FROM patients WHERE patient_id = ANY(:ids)"
        ), {"ids": patient_ids}).scalars())
        source_temporal_count = conn.execute(text(
            "SELECT COUNT(*) FROM temporal_patient_data WHERE source_dataset = :source"
        ), {"source": source}).scalar_one()

        patient_rows = []
        for patient_id, group in frame.groupby("patient_id", sort=False):
            first = group.iloc[0]
            patient_rows.append({
                "patient_id": str(patient_id), "gender": _clean(first.get("gender")),
                "age": _clean(first.get("age")), "source_dataset": source,
                "created_at": now,
            })
        for batch in _chunks(patient_rows, batch_size):
            conn.execute(text("""
                INSERT INTO patients (patient_id, gender, age, source_dataset, created_at)
                VALUES (:patient_id, :gender, :age, :source_dataset, :created_at)
                ON CONFLICT (patient_id) DO UPDATE SET
                    gender = EXCLUDED.gender, age = EXCLUDED.age,
                    source_dataset = EXCLUDED.source_dataset
            """), batch)

        visit_rows = []
        for source_row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
            visit_date = _clean(row["visit_date"])
            visit = {column: _clean(row.get(column)) for column in SOURCE_FEATURES}
            # stress_level is VARCHAR in this live schema; retain the source value as text.
            if visit["stress_level"] is not None:
                visit["stress_level"] = str(visit["stress_level"])
            visit_rows.append({
                "patient_id": str(row["patient_id"]),
                "visit_number": int(row["visit_number"]),
                "visit_date": visit_date,
                "visit_timestamp": visit_date,
                **visit,
                "smoking_status": visit.get("smoking"),
                "prediction": int(row["prediction"]),
                "is_synthetic": False,
                "source_dataset": source,
                "source_row_number": source_row_number,
                "raw_data": json.dumps({str(k): _clean(v) for k, v in row.items()}, default=str),
                "created_at": now,
                "updated_at": now,
            })

        incoming_identities = {
            (record["patient_id"], record["visit_number"], record["visit_date"])
            for record in visit_rows
        }
        prior_source_rows = conn.execute(text("""
            SELECT id, patient_id, visit_number, visit_date
            FROM patient_visits WHERE source_dataset = :source
        """), {"source": source}).mappings().all()
        prior_identities = {
            (row["patient_id"], row["visit_number"], row["visit_date"])
            for row in prior_source_rows
        }
        stale_visit_ids = [
            row["id"] for row in prior_source_rows
            if (row["patient_id"], row["visit_number"], row["visit_date"]) not in incoming_identities
        ]
        # Visits removed from the CSV are source-owned, but temporal foreign keys still
        # need to be cleared before deletion. Add short-lived lookup indexes only when
        # there are stale visits; they are dropped again before this transaction commits.
        if stale_visit_ids:
            conn.execute(text(
                "CREATE INDEX ix_refresh_temporal_current_fk ON temporal_patient_data (current_visit_id)"
            ))
            conn.execute(text(
                "CREATE INDEX ix_refresh_temporal_previous_fk ON temporal_patient_data (previous_visit_id)"
            ))
            if stale_ids:
                conn.execute(text("DELETE FROM temporal_patient_data WHERE patient_id = ANY(:ids)"), {"ids": stale_ids})
            conn.execute(text("""
                UPDATE temporal_patient_data SET current_visit_id = NULL
                WHERE current_visit_id = ANY(:visit_ids)
            """), {"visit_ids": stale_visit_ids})
            conn.execute(text("""
                UPDATE temporal_patient_data SET previous_visit_id = NULL
                WHERE previous_visit_id = ANY(:visit_ids)
            """), {"visit_ids": stale_visit_ids})
            conn.execute(text("DELETE FROM patient_visits WHERE id = ANY(:ids)"), {"ids": stale_visit_ids})
            conn.execute(text("DROP INDEX ix_refresh_temporal_current_fk"))
            conn.execute(text("DROP INDEX ix_refresh_temporal_previous_fk"))
        if stale_ids:
            conn.execute(text("DELETE FROM temporal_patient_data WHERE patient_id = ANY(:ids)"), {"ids": stale_ids})
            conn.execute(text("DELETE FROM patients WHERE patient_id = ANY(:ids)"), {"ids": stale_ids})

        visit_column_sql = ", ".join(VISIT_COLUMNS)
        visit_bind_sql = ", ".join(f":{column}" for column in VISIT_COLUMNS)
        insert_visits = text(
            f"INSERT INTO patient_visits ({visit_column_sql}) VALUES ({visit_bind_sql}) "
            "ON CONFLICT (source_dataset, patient_id, visit_number, visit_date) "
            "WHERE source_dataset IS NOT NULL DO UPDATE SET "
            + ", ".join(
                f"{column} = EXCLUDED.{column}" for column in VISIT_COLUMNS
                if column not in {"patient_id", "visit_number", "visit_date", "created_at"}
            )
        )
        for batch in _chunks(visit_rows, batch_size):
            conn.execute(insert_visits, batch)

        # Build snapshots through the same canonical service used by temporal APIs.
        selected = [
            "id", "patient_id", "visit_number", "visit_date", *SOURCE_FEATURES,
            "smoking_status", "prediction",
        ]
        visit_rows_db = conn.execute(text(
            f"SELECT {', '.join(selected)} FROM patient_visits "
            "WHERE source_dataset = :source ORDER BY patient_id, visit_date, visit_number, id"
        ), {"source": source}).mappings()
        snapshot_rows: list[dict[str, Any]] = []
        current_patient = None
        previous = current = None

        def append_snapshot(pid: str, previous_visit: dict[str, Any] | None, current_visit: dict[str, Any]) -> None:
            snapshot = TemporalFeatureService.calculate_snapshot(
                current_data=current_visit,
                previous_data=previous_visit,
                patient_id=pid,
                current_visit_id=current_visit["id"],
                previous_visit_id=previous_visit["id"] if previous_visit else None,
                current_visit_date=current_visit["visit_date"],
                previous_visit_date=previous_visit["visit_date"] if previous_visit else None,
            )
            item = {
                "patient_id": pid, "source_dataset": source,
                "current_visit_id": snapshot["current_visit_id"],
                "previous_visit_id": snapshot["previous_visit_id"],
                "current_visit_date": snapshot["current_visit_date"],
                "previous_visit_date": snapshot["previous_visit_date"],
                "days_between_visits": snapshot["days_between_visits"],
                "current_smoking": snapshot["current_smoking"],
                "previous_smoking": snapshot["previous_smoking"],
                "smoking_changed": snapshot["smoking_changed"],
                "current_physical_activity": snapshot["current_physical_activity"],
                "previous_physical_activity": snapshot["previous_physical_activity"],
                "physical_activity_changed": snapshot["physical_activity_changed"],
                "created_at": now, "updated_at": now,
            }
            for variable in NUMERICAL_VARIABLES:
                item[f"current_{variable}"] = snapshot[f"current_{variable}"]
                item[f"previous_{variable}"] = snapshot[f"previous_{variable}"]
                item[f"{variable}_change"] = snapshot[f"{variable}_change"]
                item[f"{variable}_rate"] = snapshot[f"{variable}_rate"]
            snapshot_rows.append(item)

        for row in visit_rows_db:
            record = dict(row)
            pid = record["patient_id"]
            if current_patient != pid:
                if current is not None:
                    append_snapshot(current_patient, previous, current)
                current_patient, previous, current = pid, None, record
            else:
                previous, current = current, record
        if current is not None:
            append_snapshot(current_patient, previous, current)

        snapshot_columns = [
            "patient_id", "source_dataset", "current_visit_id", "previous_visit_id",
            "current_visit_date", "previous_visit_date", "days_between_visits",
        ]
        for variable in NUMERICAL_VARIABLES:
            snapshot_columns.extend([
                f"current_{variable}", f"previous_{variable}",
                f"{variable}_change", f"{variable}_rate",
            ])
        snapshot_columns.extend([
            "current_smoking", "previous_smoking", "smoking_changed",
            "current_physical_activity", "previous_physical_activity",
            "physical_activity_changed", "created_at", "updated_at",
        ])
        insert_snapshots = text(
            f"INSERT INTO temporal_patient_data ({', '.join(snapshot_columns)}) "
            f"VALUES ({', '.join(':' + column for column in snapshot_columns)}) "
            "ON CONFLICT (patient_id) DO UPDATE SET "
            + ", ".join(
                f"{column} = EXCLUDED.{column}" for column in snapshot_columns
                if column not in {"patient_id", "created_at"}
            )
        )
        for batch in _chunks(snapshot_rows, batch_size):
            conn.execute(insert_snapshots, batch)

        counts.update({
            "patients_inserted": len(set(patient_ids) - existing_target),
            "patients_updated": len(existing_target),
            "patients_deleted": len(stale_ids),
            "visits_deleted": len(stale_visit_ids),
            "visits_inserted": len(incoming_identities - prior_identities),
            "visits_updated": len(incoming_identities & prior_identities),
            "temporal_deleted": len(stale_ids),
            "temporal_inserted": len(snapshot_rows) - int(source_temporal_count - len(stale_ids)),
            "temporal_updated": int(source_temporal_count - len(stale_ids)),
        })

        # Keep verification inside the same transaction. Any discrepancy raises here and
        # rolls back the deletes and inserts together.
        in_tx_source_counts = {
            table: int(conn.execute(text(
                f"SELECT COUNT(*) FROM {table} WHERE source_dataset = :source"
            ), {"source": source}).scalar_one())
            for table in ("patients", "patient_visits", "temporal_patient_data")
        }
        in_tx_distribution = {
            str(label): int(count) for label, count in conn.execute(text("""
                SELECT prediction, COUNT(*) FROM patient_visits
                WHERE source_dataset = :source GROUP BY prediction ORDER BY prediction
            """), {"source": source}).all()
        }
        in_tx_duplicate_keys = int(conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT patient_id, visit_number, visit_date FROM patient_visits
                WHERE source_dataset = :source
                GROUP BY patient_id, visit_number, visit_date HAVING COUNT(*) > 1
            ) duplicates
        """), {"source": source}).scalar_one())
        in_tx_synthetic_count = int(conn.execute(text("""
            SELECT COUNT(*) FROM patient_visits
            WHERE source_dataset = :source AND is_synthetic IS TRUE
        """), {"source": source}).scalar_one())
        expected_source_counts = {
            "patients": len(patient_ids),
            "patient_visits": len(frame),
            "temporal_patient_data": int(frame.patient_id.nunique()),
        }
        if in_tx_source_counts != expected_source_counts:
            raise RuntimeError(
                f"In-transaction row count verification failed: {in_tx_source_counts}; "
                f"expected {expected_source_counts}"
            )
        if in_tx_distribution != report["target_distribution"] or in_tx_duplicate_keys or in_tx_synthetic_count:
            raise RuntimeError(
                "In-transaction verification failed: "
                f"distribution={in_tx_distribution}, duplicates={in_tx_duplicate_keys}, "
                f"synthetic={in_tx_synthetic_count}"
            )

    with engine.connect() as conn:
        final_counts = {
            table: int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())
            for table in ("patients", "patient_visits", "temporal_patient_data", "predictions")
        }
        source_counts = {
            table: int(conn.execute(text(
                f"SELECT COUNT(*) FROM {table} WHERE source_dataset = :source"
            ), {"source": source}).scalar_one())
            for table in ("patients", "patient_visits", "temporal_patient_data")
        }
        distribution = {
            str(label): int(count) for label, count in conn.execute(text("""
                SELECT prediction, COUNT(*) FROM patient_visits
                WHERE source_dataset = :source GROUP BY prediction ORDER BY prediction
            """), {"source": source}).all()
        }
        duplicate_keys = int(conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT patient_id, visit_number, visit_date FROM patient_visits
                WHERE source_dataset = :source
                GROUP BY patient_id, visit_number, visit_date HAVING COUNT(*) > 1
            ) duplicates
        """), {"source": source}).scalar_one())
        synthetic_count = int(conn.execute(text("""
            SELECT COUNT(*) FROM patient_visits
            WHERE source_dataset = :source AND is_synthetic IS TRUE
        """), {"source": source}).scalar_one())
        sample_patient = frame.iloc[0]["patient_id"]
        sample = conn.execute(text("""
            SELECT p.patient_id, p.gender, p.age, COUNT(v.id) AS visits,
                   MIN(v.visit_date) AS first_visit, MAX(v.visit_date) AS last_visit
            FROM patients p LEFT JOIN patient_visits v ON v.patient_id = p.patient_id
            WHERE p.patient_id = :patient_id AND v.source_dataset = :source
            GROUP BY p.patient_id, p.gender, p.age
        """), {"patient_id": str(sample_patient), "source": source}).mappings().first()

    return {
        "source": report,
        "source_row_count": len(frame),
        "inserted_rows": int(
            counts["patients_inserted"] + counts["visits_inserted"] + counts["temporal_inserted"]
        ),
        "updated_rows": int(
            counts["patients_updated"] + counts["visits_updated"] + counts["temporal_updated"]
        ),
        "table_row_actions": {
            "patients": {
                "inserted": int(counts["patients_inserted"]),
                "updated": int(counts["patients_updated"]),
                "deleted": int(counts["patients_deleted"]),
            },
            "patient_visits": {
                "inserted": int(counts["visits_inserted"]),
                "updated": int(counts["visits_updated"]),
                "deleted": int(counts["visits_deleted"]),
            },
            "temporal_patient_data": {
                "inserted": int(counts["temporal_inserted"]),
                "updated": int(counts["temporal_updated"]),
                "deleted": int(counts["temporal_deleted"]),
            },
        },
        "deleted_rows": {
            "patients": int(counts["patients_deleted"]),
            "patient_visits": int(counts["visits_deleted"]),
            "temporal_patient_data": int(counts["temporal_deleted"]),
        },
        "skipped_duplicates": 0,
        "rejected_rows": 0,
        "patient_rows_inserted": int(counts["patients_inserted"]),
        "temporal_rows_inserted": int(counts["temporal_inserted"]),
        "final_counts": final_counts,
        "source_counts": source_counts,
        "target_distribution": distribution,
        "duplicate_visit_keys": duplicate_keys,
        "synthetic_source_rows": synthetic_count,
        "sample_patient_lookup": dict(sample) if sample else None,
    }
