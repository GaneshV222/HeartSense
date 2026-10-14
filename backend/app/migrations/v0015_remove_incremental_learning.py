"""Remove empty tables used only by the retired incremental-learning feature.

The migration is deliberately guarded: it refuses to remove any table that has
rows or an unexpected foreign-key dependency. PostgreSQL RESTRICT provides a
second dependency check at the point each table is dropped.
"""

from sqlalchemy import text

VERSION = "0015_remove_incremental_learning"

_TABLES = (
    "incremental_learning_batches",
    "evaluation_manifests",
    "global_evaluation_state",
    "model_evaluations",
    "model_promotions",
    "replay_buffer_snapshots",
    "model_versions",
    "dataset_versions",
)

_EXPECTED_FOREIGN_KEYS = {
    ("replay_buffer_snapshots", "model_versions"),
}


def upgrade(connection) -> None:
    """Drop only confirmed empty feature tables, in dependency order."""
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

    existing = set(connection.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
    )).scalars())
    targets = existing.intersection(_TABLES)

    for table in sorted(targets):
        count = connection.execute(text(f'SELECT count(*) FROM "{table}"')).scalar_one()
        if count != 0:
            raise RuntimeError(
                f"Refusing to remove {table}: it contains {count} rows."
            )

    foreign_keys = set(connection.execute(text(
        "SELECT source.relname, target.relname "
        "FROM pg_constraint fk "
        "JOIN pg_class source ON source.oid = fk.conrelid "
        "JOIN pg_namespace source_ns ON source_ns.oid = source.relnamespace "
        "JOIN pg_class target ON target.oid = fk.confrelid "
        "JOIN pg_namespace target_ns ON target_ns.oid = target.relnamespace "
        "WHERE fk.contype = 'f' AND source_ns.nspname = current_schema() "
        "AND target_ns.nspname = current_schema()"
    )).all())
    relevant = {
        (source, target)
        for source, target in foreign_keys
        if source in targets or target in targets
    }
    if relevant != (_EXPECTED_FOREIGN_KEYS & relevant):
        unexpected = sorted(relevant - _EXPECTED_FOREIGN_KEYS)
        raise RuntimeError(
            "Refusing incremental-learning schema cleanup because unexpected "
            f"foreign-key dependencies exist: {unexpected}"
        )

    # replay_buffer_snapshots references model_versions, so remove the child first.
    for table in _TABLES:
        if table in targets:
            connection.execute(text(f'DROP TABLE "{table}" RESTRICT'))

    connection.execute(
        text("INSERT INTO schema_migrations (version) VALUES (:version)"),
        {"version": VERSION},
    )


def downgrade(connection) -> None:
    """Recreate the empty tables from a schema-only backup before reverting.

    The feature's table definitions are intentionally not maintained in the
    application after retirement. Recreating them here would silently restore
    a deleted feature surface, so downgrade requires a deliberate DBA restore.
    """
    raise RuntimeError(
        "This migration is not automatically reversible. Restore the schema "
        "from the pre-migration pg_dump if rollback is required."
    )
