"""Create durable import and Challenger lineage tables."""

from sqlalchemy import text

VERSION = "0016_incremental_learning"


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

    # Base.metadata.create_all runs before migrations in init_db; these definitions
    # also make the migration safe when applied independently.
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS learning_import_batches ("
        "batch_id VARCHAR(100) PRIMARY KEY, idempotency_key VARCHAR(64) NOT NULL UNIQUE, "
        "source_filename VARCHAR(255) NOT NULL, source_sha256 VARCHAR(64) NOT NULL, "
        "temporal_filename VARCHAR(255), temporal_sha256 VARCHAR(64), source_path TEXT NOT NULL, "
        "status VARCHAR(30) NOT NULL DEFAULT 'validated', validation_report JSON NOT NULL, "
        "accepted_rows INTEGER NOT NULL DEFAULT 0, rejected_rows INTEGER NOT NULL DEFAULT 0, "
        "duplicate_rows INTEGER NOT NULL DEFAULT 0, missing_value_cells INTEGER NOT NULL DEFAULT 0, "
        "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, confirmed_at TIMESTAMP)"
    ))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_learning_import_batches_status "
        "ON learning_import_batches (status)"
    ))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_learning_import_batches_idempotency_key "
        "ON learning_import_batches (idempotency_key)"
    ))
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS learning_training_runs ("
        "training_run_id VARCHAR(100) PRIMARY KEY, status VARCHAR(30) NOT NULL, "
        "artifact_path TEXT, manifest_path TEXT, dataset_sha256 VARCHAR(64), "
        "dataset_rows INTEGER, report JSON, decision VARCHAR(30), decision_reason TEXT, "
        "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP)"
    ))
    connection.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_learning_training_runs_status "
        "ON learning_training_runs (status)"
    ))
    connection.execute(text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                       {"version": VERSION})


def downgrade(connection) -> None:
    raise RuntimeError("Incremental learning provenance is retained; restore from an explicit database backup to downgrade.")
