"""
HeartSense – Database Connection

SQLAlchemy engine, session factory, and table-creation utility.
Requires PostgreSQL as the production database.
"""

import sys
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

    # Ensure indexes and compatibility columns exist idempotently
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE patient_visits ADD COLUMN IF NOT EXISTS smoking_status VARCHAR(50)"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patient_temporal_features (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(100) NOT NULL,
                visit_id INTEGER,
                assessment_date TIMESTAMP NOT NULL,
                previous_systolic_bp FLOAT,
                current_systolic_bp FLOAT,
                delta_systolic_bp FLOAT,
                previous_diastolic_bp FLOAT,
                current_diastolic_bp FLOAT,
                delta_diastolic_bp FLOAT,
                previous_cholesterol FLOAT,
                current_cholesterol FLOAT,
                delta_cholesterol FLOAT,
                previous_ldl FLOAT,
                current_ldl FLOAT,
                delta_ldl FLOAT,
                previous_hdl FLOAT,
                current_hdl FLOAT,
                delta_hdl FLOAT,
                previous_bmi FLOAT,
                current_bmi FLOAT,
                delta_bmi FLOAT,
                previous_hba1c FLOAT,
                current_hba1c FLOAT,
                delta_hba1c FLOAT,
                previous_resting_heart_rate FLOAT,
                current_resting_heart_rate FLOAT,
                delta_resting_heart_rate FLOAT,
                previous_smoking_status VARCHAR(50),
                current_smoking_status VARCHAR(50),
                smoking_status_changed BOOLEAN,
                previous_physical_activity VARCHAR(100),
                current_physical_activity VARCHAR(100),
                physical_activity_changed BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_ptf_patient_id ON patient_temporal_features (patient_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_ptf_assessment_date ON patient_temporal_features (assessment_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_ptf_visit_id ON patient_temporal_features (visit_id)"))

    print("[Database] PostgreSQL tables created and verified successfully.")
