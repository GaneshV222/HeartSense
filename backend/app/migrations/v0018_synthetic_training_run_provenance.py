"""Record synthetic sampler configuration and feature-schema provenance."""

from sqlalchemy import text

VERSION = "0018_synthetic_training_run_provenance"


def upgrade(connection) -> None:
    """Reconcile the two v0018 fields without dropping or rewriting sample rows."""
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version VARCHAR(100) PRIMARY KEY, "
        "applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    ))
    # The version marker can exist even when a previous deployment lost this source
    # file. Make the schema check idempotent instead of trusting the marker alone.
    columns = set(connection.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() "
        "AND table_name = 'synthetic_training_samples'"
    )).scalars())
    if not columns:
        raise RuntimeError("synthetic_training_samples must exist before migration 0018")
    if "feature_schema_version" not in columns:
        connection.execute(text(
            "ALTER TABLE synthetic_training_samples "
            "ADD COLUMN feature_schema_version VARCHAR(100)"
        ))
    if "sampler_configuration" not in columns:
        connection.execute(text(
            "ALTER TABLE synthetic_training_samples "
            "ADD COLUMN sampler_configuration JSON NOT NULL DEFAULT '{}'::json"
        ))
    connection.execute(text(
        "INSERT INTO schema_migrations (version) VALUES (:version) "
        "ON CONFLICT (version) DO NOTHING"
    ), {"version": VERSION})


def downgrade(connection) -> None:
    raise RuntimeError(
        "Migration 0018 is provenance-only and is not automatically downgraded; "
        "restore from a verified database backup for a schema rollback."
    )
