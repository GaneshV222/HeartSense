"""Legacy manual cleanup utility. Requires an explicit connection string and is not run on startup."""

import os

import psycopg2

database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("Set DATABASE_URL explicitly before running this cleanup utility")

conn = psycopg2.connect(database_url)
cur = conn.cursor()

cur.execute("DELETE FROM patient_visits WHERE source_dataset = 'incremental:il-492461d14523bd060c0f';")
cur.execute("DELETE FROM learning_import_batches WHERE batch_id = 'il-492461d14523bd060c0f';")
cur.execute("DELETE FROM temporal_patient_data WHERE patient_id LIKE 'CONFIRM-TEST-%' OR patient_id LIKE 'IDEMP-TEST-%';")
cur.execute("DELETE FROM patients WHERE patient_id LIKE 'CONFIRM-TEST-%' OR patient_id LIKE 'IDEMP-TEST-%';")
conn.commit()

cur.execute("SELECT COUNT(*) FROM patient_visits;")
print("Visits:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM patients;")
print("Patients:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM temporal_patient_data;")
print("Temporal:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM synthetic_training_samples;")
print("Synthetic:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM learning_import_batches;")
print("Batches:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM learning_training_runs;")
print("Runs:", cur.fetchone()[0])
conn.close()
