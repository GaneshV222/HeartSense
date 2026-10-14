"""Transactional execution script for safe incremental data cleanup.
Restores baseline 100,001 real visits, removes targeted incremental imports,
preserves baseline patients, Golden Benchmark, and Champion artifact.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import text
from app.database import engine
from app.config import MODEL_ARTIFACT_DIR

TARGET_BATCHES = [
    "il-9e8b62a8cd4cf7302a5b",
    "il-a451e3fc726d3789414b",
]

TARGET_RUNS = [
    "ilrun-20261013T174259Z-2800b41800",
    "ilrun-20261013T183627Z-7a743c132a",
]

CHAMPION_PATH = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
EXPECTED_CHAMPION_SHA = "45576cce7069b298f7c2350cc3b53a4e626f8219f1172d218760dc7669ab144b"

GOLDEN_BENCHMARK_PATH = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark" / "golden_benchmark_records.csv"
EXPECTED_BENCHMARK_SHA = "559bfadd4539554507407d9ca49f1ceda220ad4b1ec30ee553f0045b42f5ab18"


def execute_cleanup():
    print("=== STARTING TRANSACTIONAL SAFE INCREMENTAL CLEANUP ===")

    # Pre-cleanup checksum verifications
    assert CHAMPION_PATH.is_file(), "Champion artifact missing!"
    champ_sha_pre = hashlib.sha256(CHAMPION_PATH.read_bytes()).hexdigest()
    assert champ_sha_pre == EXPECTED_CHAMPION_SHA, f"Champion altered! {champ_sha_pre}"

    assert GOLDEN_BENCHMARK_PATH.is_file(), "Golden Benchmark records missing!"
    bm_sha_pre = hashlib.sha256(GOLDEN_BENCHMARK_PATH.read_bytes()).hexdigest()
    assert bm_sha_pre == EXPECTED_BENCHMARK_SHA, f"Benchmark altered! {bm_sha_pre}"

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # 1. Identify visits belonging to the targeted batches
            target_sources = [f"incremental:{b}" for b in TARGET_BATCHES]
            visit_rows = conn.execute(
                text("SELECT id, patient_id FROM patient_visits WHERE source_dataset = ANY(:sources)"),
                {"sources": target_sources},
            ).fetchall()
            visit_ids = [r[0] for r in visit_rows]
            patient_ids = list(set(r[1] for r in visit_rows))

            print(f"Targeted visits to remove: {len(visit_ids)}")
            print(f"Targeted patients to remove: {len(patient_ids)}")
            assert len(visit_ids) == 1000, f"Expected 1000 visits, got {len(visit_ids)}"
            assert len(patient_ids) == 1000, f"Expected 1000 patients, got {len(patient_ids)}"

            # Safety assertion: verify none of these patients have baseline visits
            baseline_overlap = conn.execute(
                text("SELECT COUNT(*) FROM patient_visits WHERE patient_id = ANY(:pts) AND NOT (source_dataset = ANY(:sources))"),
                {"pts": patient_ids, "sources": target_sources},
            ).scalar()
            if baseline_overlap > 0:
                raise RuntimeError(f"Safety violation: {baseline_overlap} baseline visits found for targeted patients!")

            # 2. Delete temporal patient data for these patients
            d_temp = conn.execute(
                text("DELETE FROM temporal_patient_data WHERE patient_id = ANY(:pts)"),
                {"pts": patient_ids},
            ).rowcount
            print(f"Deleted {d_temp} temporal_patient_data records.")

            # 3. Delete patient visits
            d_vis = conn.execute(
                text("DELETE FROM patient_visits WHERE id = ANY(:vids)"),
                {"vids": visit_ids},
            ).rowcount
            print(f"Deleted {d_vis} patient_visits records.")

            # 4. Delete patients (who only existed for these visits)
            d_pts = conn.execute(
                text("DELETE FROM patients WHERE patient_id = ANY(:pts)"),
                {"pts": patient_ids},
            ).rowcount
            print(f"Deleted {d_pts} patients records.")

            # 5. Delete synthetic samples belonging exclusively to targeted runs
            d_syn = conn.execute(
                text("DELETE FROM synthetic_training_samples WHERE training_run_id = ANY(:rids)"),
                {"rids": TARGET_RUNS},
            ).rowcount
            print(f"Deleted {d_syn} synthetic_training_samples records.")

            # 6. Update targeted batches status to 'rolled_back' so they don't contribute to active accepted counts
            u_batches = conn.execute(
                text("UPDATE learning_import_batches SET status = 'rolled_back' WHERE batch_id = ANY(:bids)"),
                {"bids": TARGET_BATCHES},
            ).rowcount
            print(f"Updated {u_batches} learning_import_batches to status='rolled_back'.")

            # 7. Update targeted runs status to 'rolled_back' while preserving report and metrics for historical audit
            u_runs = conn.execute(
                text("""
                    UPDATE learning_training_runs 
                    SET status = 'rolled_back',
                        decision_reason = COALESCE(decision_reason, '') || ' [ROLLED_BACK: Incremental dataset restored to 100,001 baseline]'
                    WHERE training_run_id = ANY(:rids)
                """),
                {"rids": TARGET_RUNS},
            ).rowcount
            print(f"Updated {u_runs} learning_training_runs to status='rolled_back'.")

            # 8. Post-cleanup assertions before commit
            total_visits_after = conn.execute(text("SELECT COUNT(*) FROM patient_visits")).scalar()
            baseline_visits_after = conn.execute(
                text("SELECT COUNT(*) FROM patient_visits WHERE source_dataset = 'heart_disease_prediction_2026.csv'")
            ).scalar()
            inc_visits_after = conn.execute(
                text("SELECT COUNT(*) FROM patient_visits WHERE source_dataset LIKE 'incremental:%'")
            ).scalar()
            accepted_batch_rows = conn.execute(
                text("SELECT COALESCE(SUM(accepted_rows), 0) FROM learning_import_batches WHERE status = 'accepted'")
            ).scalar()
            active_synths = conn.execute(
                text("SELECT COUNT(*) FROM synthetic_training_samples WHERE training_run_id = ANY(:rids)"),
                {"rids": TARGET_RUNS},
            ).scalar()

            print("\n=== VERIFICATION ASSERTIONS ===")
            print(f"Total Visits: {total_visits_after} (expected: 100001)")
            print(f"Baseline Visits Preserved: {baseline_visits_after} (expected: 100001)")
            print(f"Incremental Visits Remaining: {inc_visits_after} (expected: 0)")
            print(f"Active Accepted Batch Rows: {accepted_batch_rows} (expected: 0)")
            print(f"Active Targeted Synthetic Samples: {active_synths} (expected: 0)")

            assert total_visits_after == 100001, f"Expected exactly 100001 visits, got {total_visits_after}"
            assert baseline_visits_after == 100001, f"Expected 100001 baseline visits, got {baseline_visits_after}"
            assert inc_visits_after == 0, f"Expected 0 incremental visits, got {inc_visits_after}"
            assert accepted_batch_rows == 0, f"Expected 0 active accepted batch rows, got {accepted_batch_rows}"
            assert active_synths == 0, f"Expected 0 active synthetic samples, got {active_synths}"

            trans.commit()
            print("\n>>> TRANSACTION COMMITTED SUCCESSFULLY! <<<")

        except Exception as exc:
            trans.rollback()
            print(f"\nERROR: Cleanup rolled back due to error: {exc}")
            raise

    # Post-cleanup checksum verifications
    champ_sha_post = hashlib.sha256(CHAMPION_PATH.read_bytes()).hexdigest()
    assert champ_sha_post == EXPECTED_CHAMPION_SHA, f"Champion altered! {champ_sha_post}"

    bm_sha_post = hashlib.sha256(GOLDEN_BENCHMARK_PATH.read_bytes()).hexdigest()
    assert bm_sha_post == EXPECTED_BENCHMARK_SHA, f"Benchmark altered! {bm_sha_post}"

    print(f"Champion checksum preserved: {champ_sha_post}")
    print(f"Golden Benchmark checksum preserved: {bm_sha_post}")


if __name__ == "__main__":
    execute_cleanup()
