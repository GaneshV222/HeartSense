"""
HeartSense – Database Connection

SQLAlchemy engine, session factory, and table-creation utility.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import DATABASE_URL


engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session() -> Session:
    """Create and return a new database session."""
    return SessionLocal()


def init_db():
    """Create all tables defined in models.py (idempotent)."""
    from app.models import Base  # imported here to avoid circular imports
    Base.metadata.create_all(bind=engine)
    print("[Database] Tables created / verified.")
