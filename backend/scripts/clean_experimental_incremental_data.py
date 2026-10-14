"""Transactional cleanup script for experimental incremental-learning data.
Preserves baseline dataset, legitimate patient records, temporal snapshots, and active Champion.
"""

import sys
from pathlib import Path
from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import engine
from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR

BATCH_IDS = [
    "il-9e8b62a8cd4cf7302a5b",
    "il-a451e3fc726d3789414b",
    "il-f17abd893c800c3a52f6",
]

EXPERIMENTAL_RUN_IDS = [
    "ilrun-20261013T155144Z-bc61351746",
    "ilrun-20261013T065042Z-8a0a7c61ff",
    "ilrun-20261013T062218Z-6d548c0f1b",
    "ilrun-20261013T055933Z-6e061314de",
    "ilrun-20261013T045727Z-79194db3e4",
]


def run_cleanup():
    print("=== STARTING TRANSACTIONAL EXPERIMENTAL CLEANUP ===")
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # 1. Identify visits belonging to the experimental batches
            inc_sources = [f"incremental:{b}" for b in BATCH_IDS]
            visit_res = conn.execute(
                text("SELECT id, patient_id FROM patient_visits WHERE source_dataset = ANY(:sources)"),
                {"sources": inc_sources},
            ).all()
            visit_ids = [r[0] for r in visit_res]
            inc_patients = list(set([r[1] for r in visit_res]))
            print(f"Identified {len(visit_ids)} visits and {len(inc_patients)} patients for cleanup.")

            # Safety assertion: verify no non-experimental visits exist for these patients
            other_visits = conn.execute(
                text("SELECT COUNT(*) FROM patient_visits WHERE patient_id = ANY(:pts) AND NOT (source_dataset = ANY(:sources))"),
                {"pts": inc_patients, "sources": inc_sources},
            ).scalar()
            if other_visits > 0:
                raise RuntimeError(f"Safety violation: {other_visits} baseline visits found for experimental patients!")

            # 2. Delete temporal patient data for these patients
            d_temp = conn.execute(
                text("DELETE FROM temporal_patient_data WHERE patient_id = ANY(:pts)"),
                {"pts": inc_patients},
            ).rowcount
            print(f"Deleted {d_temp} temporal_patient_data records.")

            # 3. Delete patient visits
            d_vis = conn.execute(
                text("DELETE FROM patient_visits WHERE id = ANY(:vids)"),
                {"vids": visit_ids},
            ).rowcount
            print(f"Deleted {d_vis} patient_visits records.")

            # 4. Delete patients
            d_pts = conn.execute(
                text("DELETE FROM patients WHERE patient_id = ANY(:pts)"),
                {"pts": inc_patients},
            ).rowcount
            print(f"Deleted {d_pts} patients records.")

            # 5. Delete synthetic samples belonging exclusively to experimental runs
            d_syn = conn.execute(
                text("DELETE FROM synthetic_training_samples WHERE training_run_id = ANY(:rids)"),
                {"rids": EXPERIMENTAL_RUN_IDS},
            ).rowcount
            print(f"Deleted {d_syn} synthetic_training_samples records.")

            # 6. Delete experimental training runs
            d_runs = conn.execute(
                text("DELETE FROM learning_training_runs WHERE training_run_id = ANY(:rids)"),
                {"rids": EXPERIMENTAL_RUN_IDS},
            ).rowcount
            print(f"Deleted {d_runs} learning_training_runs records.")

            # 7. Delete learning import batches
            d_b = conn.execute(
                text("DELETE FROM learning_import_batches WHERE batch_id = ANY(:bids)"),
                {"bids": BATCH_IDS},
            ).rowcount
            print(f"Deleted {d_b} learning_import_batches records.")

            # 8. Verification assertions
            c_vis = conn.execute(text("SELECT COUNT(*) FROM patient_visits")).scalar()
            c_pts = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
            c_temp = conn.execute(text("SELECT COUNT(*) FROM temporal_patient_data")).scalar()
            c_syn = conn.execute(text("SELECT COUNT(*) FROM synthetic_training_samples")).scalar()
            c_b = conn.execute(text("SELECT COUNT(*) FROM learning_import_batches")).scalar()
            c_runs = conn.execute(text("SELECT COUNT(*) FROM learning_training_runs")).scalar()

            print("\n=== POST-CLEANUP VERIFICATION ===")
            print(f"Patient Visits:        {c_vis} (expected 100001)")
            print(f"Patients:              {c_pts} (expected 99973)")
            print(f"Temporal Patient Data: {c_temp} (expected 99973)")
            print(f"Synthetic Samples:     {c_syn} (expected 0)")
            print(f"Import Batches:        {c_b} (expected 0)")
            print(f"Training Runs:         {c_runs} (expected 1 - baseline run)")

            assert c_vis == 100001, f"Expected 100001 visits, got {c_vis}"
            assert c_pts == 99973, f"Expected 99973 patients, got {c_pts}"
            assert c_temp == 99973, f"Expected 99973 temporal records, got {c_temp}"
            assert c_syn == 0, f"Expected 0 synthetic samples, got {c_syn}"
            assert c_b == 0, f"Expected 0 batches, got {c_b}"
            assert c_runs == 1, f"Expected 1 run, got {c_runs}"

            trans.commit()
            print("\nSUCCESS: Transaction committed successfully!")
        except Exception as e:
            trans.rollback()
            print(f"\nERROR: Cleanup rolled back due to error: {e}")
            raise


if __name__ == "__main__":
    run_cleanup()
