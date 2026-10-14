"""Load the configured HeartSense CSV into the existing PostgreSQL database."""

from app.services.dataset_ingestion import ingest_csv


if __name__ == "__main__":
    ingest_csv()
