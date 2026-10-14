"""Add a request idempotency key without changing existing run history."""

from sqlalchemy import text

VERSION = "0019_incremental_run_idempotency"


def upgrade(connection) -> None:
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version VARCHAR(100) PRIMARY KEY, "
        "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    ))
    columns = set(connection.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() "
        "AND table_name = 'learning_training_runs'"
    )).scalars())
    if not columns:
        raise RuntimeError("learning_training_runs must exist before migration 0019")
    if "idempotency_key" not in columns:
        connection.execute(text(
            "ALTER TABLE learning_training_runs "
            "ADD COLUMN idempotency_key VARCHAR(100)"
        ))
    connection.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_learning_training_runs_idempotency_key "
        "ON learning_training_runs (idempotency_key) WHERE idempotency_key IS NOT NULL"
    ))
    connection.execute(text(
        "INSERT INTO schema_migrations (version) VALUES (:version) "
        "ON CONFLICT (version) DO NOTHING"
    ), {"version": VERSION})


def downgrade(connection) -> None:
    raise RuntimeError(
        "Run idempotency provenance is retained; restore from a verified backup "
        "for a schema rollback."
    )
