"""Restore the patient_code ORM field in databases created from older schemas."""

from sqlalchemy import text

VERSION = "0017_patient_code_compatibility"


def upgrade(connection) -> None:
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version VARCHAR(100) PRIMARY KEY, "
        "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    ))
    if connection.execute(
        text("SELECT 1 FROM schema_migrations WHERE version = :version"),
        {"version": VERSION},
    ).scalar():
        return
    connection.execute(text(
        "ALTER TABLE patients ADD COLUMN IF NOT EXISTS patient_code VARCHAR(100)"
    ))
    connection.execute(text(
        "UPDATE patients SET patient_code = patient_id WHERE patient_code IS NULL"
    ))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_patients_patient_code ON patients (patient_code)"
    ))
    connection.execute(text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                       {"version": VERSION})


def downgrade(connection) -> None:
    raise RuntimeError("Existing patient codes are retained; restore from an explicit backup to downgrade.")
