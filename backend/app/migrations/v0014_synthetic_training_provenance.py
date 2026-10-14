"""Add run-scoped idempotency and dataset provenance to synthetic samples."""

from sqlalchemy import text

VERSION = "0014_synthetic_training_provenance"


def upgrade(connection) -> None:
    """Add nullable lineage fields so existing rows remain explicitly unverified."""
    connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version VARCHAR(100) PRIMARY KEY, "
            "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    applied = connection.execute(
        text("SELECT 1 FROM schema_migrations WHERE version = :version"),
        {"version": VERSION},
    ).scalar()
    if applied:
        return

    connection.execute(
        text("ALTER TABLE synthetic_training_samples "
             "ADD COLUMN IF NOT EXISTS training_run_id VARCHAR(255)")
    )
    connection.execute(
        text("ALTER TABLE synthetic_training_samples "
             "ADD COLUMN IF NOT EXISTS dataset_path TEXT")
    )
    connection.execute(
        text("ALTER TABLE synthetic_training_samples "
             "ADD COLUMN IF NOT EXISTS dataset_sha256 VARCHAR(64)")
    )
    connection.execute(
        text("ALTER TABLE synthetic_training_samples "
             "ADD COLUMN IF NOT EXISTS sample_index INTEGER")
    )
    connection.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_synthetic_training_run_sample "
            "ON synthetic_training_samples (training_run_id, sample_index) "
            "WHERE training_run_id IS NOT NULL AND sample_index IS NOT NULL"
        )
    )
    connection.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": VERSION},
    )


def downgrade(connection) -> None:
    """Remove only this migration's columns/index; retained sample rows stay intact."""
    connection.execute(text("DROP INDEX IF EXISTS uq_synthetic_training_run_sample"))
    for column in ("sample_index", "dataset_sha256", "dataset_path", "training_run_id"):
        connection.execute(
            text(f"ALTER TABLE synthetic_training_samples DROP COLUMN IF EXISTS {column}")
        )
    connection.execute(
        text("DELETE FROM schema_migrations WHERE version = :version"),
        {"version": VERSION},
    )
