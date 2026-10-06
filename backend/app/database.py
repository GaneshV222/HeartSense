"""
HeartSense – Database Connection

SQLAlchemy engine, session factory, and table-creation utilities.
Requires PostgreSQL as the production database.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.config import DATABASE_URL

if not DATABASE_URL or not DATABASE_URL.startswith(("postgresql", "postgres")):
    raise RuntimeError(
        f"Invalid database configuration: DATABASE_URL must point to PostgreSQL. "
        f"Current: '{DATABASE_URL}'. Please check your .env file."
    )

try:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
except Exception as e:
    raise RuntimeError(f"Failed to create PostgreSQL engine for URL '{DATABASE_URL}': {e}") from e

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session() -> Session:
    """Create and return a new database session."""
    return SessionLocal()


def verify_db_connection() -> bool:
    """Verify active PostgreSQL connection."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        raise RuntimeError(
            f"PostgreSQL connection failed. Please ensure PostgreSQL server is running "
            f"and DATABASE_URL in .env is correct. Error: {e}"
        ) from e


def init_db():
    """Create all tables defined in models.py idempotently and verify connection."""
    verify_db_connection()
    from app.models import Base  # imported here to avoid circular imports
    Base.metadata.create_all(bind=engine)

    # Ensure indexes and tables exist idempotently
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_patient_visits_patient_id ON patient_visits (patient_id);
            CREATE INDEX IF NOT EXISTS ix_patient_visits_visit_date ON patient_visits (visit_date);
            CREATE INDEX IF NOT EXISTS ix_temporal_patient_data_patient_id ON temporal_patient_data (patient_id);
            CREATE INDEX IF NOT EXISTS ix_predictions_patient_id ON predictions (patient_id);
        """))

    print("[Database] PostgreSQL tables created and verified successfully.")


def reset_db_tables():
    """Drop and recreate all tables for a clean migration to the new dataset."""
    verify_db_connection()
    from app.models import Base
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS patient_temporal_features CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS temporal_patient_data CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS predictions CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS model_runs CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS patient_visits CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS patients CASCADE;"))

    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_patient_visits_patient_id ON patient_visits (patient_id);
            CREATE INDEX IF NOT EXISTS ix_patient_visits_visit_date ON patient_visits (visit_date);
            CREATE INDEX IF NOT EXISTS ix_temporal_patient_data_patient_id ON temporal_patient_data (patient_id);
            CREATE INDEX IF NOT EXISTS ix_predictions_patient_id ON predictions (patient_id);
        """))
    print("[Database] PostgreSQL tables reset and recreated successfully.")
