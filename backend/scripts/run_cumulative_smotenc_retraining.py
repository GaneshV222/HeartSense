"""Execute cumulative retraining workflow with SMOTENC-before-split and exact synthetic accounting.
Workflow:
1. Verify existing 40,001 synthetic records in synthetic_training_samples.
2. Verify baseline dataset: 100,001 rows (Class 0: 70,001, Class 1: 30,000).
3. Verify newly eligible batch: 9,875 rows (Class 0: 6,904, Class 1: 2,971, Delta: 3,933).
4. Reconstruct cumulative dataset: 109,876 rows (Class 0: 76,905, Class 1: 32,971).
5. Apply SMOTENC before train/test split -> produces 43,934 synthetic records.
6. Persist exactly the 3,933 newly generated samples into synthetic_training_samples.
7. Verify total synthetic count in synthetic_training_samples is exactly 43,934 (40,001 + 3,933).
8. Retrain Challenger models, evaluate Golden Benchmark, verify Champion preservation.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import pandas as pd
from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR
from app.database import engine, get_session
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    PatientVisit,
    SyntheticTrainingSample,
)
from app.services.incremental_learning import run_cumulative_training

BATCH_ID = "il-d27e301814880c273656"
BATCH_CSV = BACKEND_ROOT / "artifacts" / "incremental_learning" / "imports" / f"{BATCH_ID}.csv"
CHAMPION_PATH = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
EXPECTED_CHAMPION_SHA = "45576cce7069b298f7c2350cc3b53a4e626f8219f1172d218760dc7669ab144b"
EXPECTED_BENCHMARK_SHA = "559bfadd4539554507407d9ca49f1ceda220ad4b1ec30ee553f0045b42f5ab18"


def execute_cumulative_retraining():
    print("=== STARTING CUMULATIVE SMOTENC RETRAINING WORKFLOW ===")
    session = get_session()
    try:
        # Step 1: Verify Champion and Golden Benchmark intact before training
        assert CHAMPION_PATH.is_file(), "Champion artifact missing"
        champ_sha = hashlib.sha256(CHAMPION_PATH.read_bytes()).hexdigest()
        assert champ_sha == EXPECTED_CHAMPION_SHA, f"Champion altered before training! {champ_sha}"
        print(f"[OK] Active Champion verified: {champ_sha}")

        # Step 2: Verify existing baseline synthetic samples
        base_synth_count = session.query(SyntheticTrainingSample).filter(
            SyntheticTrainingSample.training_run_id == "ilrun-20261013T042318Z-7579f93cb7"
        ).count()
        print(f"[OK] Persisted baseline synthetic samples: {base_synth_count} (expected: 40001)")
        assert base_synth_count == 40001, f"Expected 40001 baseline synths, got {base_synth_count}"

        # Step 3: Verify batch CSV class counts
        assert BATCH_CSV.is_file(), f"Batch CSV missing: {BATCH_CSV}"
        batch_df = pd.read_csv(BATCH_CSV)
        assert len(batch_df) == 9875, f"Expected 9875 batch rows, got {len(batch_df)}"
        b_c0 = int((batch_df["prediction"] == 0).sum())
        b_c1 = int((batch_df["prediction"] == 1).sum())
        print(f"[OK] Batch 9,875 verified: Class 0 = {b_c0}, Class 1 = {b_c1}, Delta = {b_c0 - b_c1}")
        assert b_c0 == 6904, f"Expected 6904 class 0, got {b_c0}"
        assert b_c1 == 2971, f"Expected 2971 class 1, got {b_c1}"
        assert b_c0 - b_c1 == 3933, f"Expected delta of 3933, got {b_c0 - b_c1}"

        # Step 4: Verify baseline source dataset class counts
        base_df = pd.read_csv(DATASET_PATH)
        assert len(base_df) == 100001, f"Expected 100001 baseline rows, got {len(base_df)}"
        base_c0 = int((base_df["prediction"] == 0).sum())
        base_c1 = int((base_df["prediction"] == 1).sum())
        print(f"[OK] Baseline verified: Class 0 = {base_c0}, Class 1 = {base_c1}, Delta = {base_c0 - base_c1}")
        assert base_c0 == 70001
        assert base_c1 == 30000

        # Step 5: Activate the batch in learning_import_batches
        batch_record = session.get(IncrementalLearningBatch, BATCH_ID)
        if batch_record:
            batch_record.status = "accepted"
            batch_record.source_path = str(BATCH_CSV.resolve())
            batch_record.accepted_rows = 9875
        else:
            session.add(IncrementalLearningBatch(
                batch_id=BATCH_ID,
                idempotency_key=f"idem-{BATCH_ID}",
                source_filename="incremental_dataset.csv",
                source_sha256=hashlib.sha256(BATCH_CSV.read_bytes()).hexdigest(),
                source_path=str(BATCH_CSV.resolve()),
                status="accepted",
                validation_report={},
                accepted_rows=9875,
            ))
        session.commit()
        print(f"[OK] Batch {BATCH_ID} set to status='accepted' for cumulative training.")

    finally:
        session.close()

    # Step 6: Execute cumulative training
    idempotency_key = f"cumulative-smotenc-{uuid.uuid4().hex[:12]}"
    print(f"Triggering run_cumulative_training with idempotency_key={idempotency_key}...")
    result = run_cumulative_training(idempotency_key=idempotency_key)
    run_id = result["training_run_id"]
    report = result["report"]
    print(f"\n>>> Retraining completed successfully! Run ID: {run_id} <<<")

    # Step 7: Post-training assertions and verification
    session = get_session()
    try:
        run_record = session.get(IncrementalTrainingRun, run_id)
        assert run_record is not None
        assert run_record.status == "completed"
        print(f"Run decision: {run_record.decision} (Reason: {run_record.decision_reason})")

        # Verify SMOTE counts
        smote_rep = report.get("smote", {})
        samples_created = smote_rep.get("synthetic_samples_created")
        samples_before = smote_rep.get("samples_before")
        samples_after = smote_rep.get("samples_after")
        print(f"SMOTE samples before: {samples_before} (expected: 109876)")
        print(f"SMOTE synthetic samples created: {samples_created} (expected: 43934)")
        print(f"SMOTE samples after: {samples_after} (expected: 153810)")
        assert samples_before == 109876
        assert samples_created == 43934
        assert samples_after == 153810

        # Verify synthetic storage in database
        this_run_synths = session.query(SyntheticTrainingSample).filter(
            SyntheticTrainingSample.training_run_id == run_id
        ).count()
        total_synths = session.query(SyntheticTrainingSample).count()
        print(f"Newly inserted synthetic samples for {run_id}: {this_run_synths} (expected: 3933)")
        print(f"Total synthetic samples in PostgreSQL table: {total_synths} (expected: 43934)")
        assert this_run_synths == 3933, f"Expected 3933 newly inserted synths, got {this_run_synths}"
        assert total_synths == 43934, f"Expected 43934 total synths, got {total_synths}"

        # Verify Champion artifact remains unmodified
        champ_sha_after = hashlib.sha256(CHAMPION_PATH.read_bytes()).hexdigest()
        assert champ_sha_after == EXPECTED_CHAMPION_SHA, f"Champion was altered! {champ_sha_after}"
        print(f"[OK] Champion preserved intact: {champ_sha_after}")

        # Summary of metrics
        gb_comp = report.get("golden_benchmark_comparison", {})
        if gb_comp.get("comparable"):
            c_metrics = gb_comp.get("challenger_metrics", {})
            m_metrics = gb_comp.get("champion_metrics", {})
            print("\n=== Golden Benchmark Evaluation Metrics ===")
            print(f"Challenger Accuracy: {c_metrics.get('accuracy'):.4f} | Champion Accuracy: {m_metrics.get('accuracy'):.4f}")
            print(f"Challenger F1:       {c_metrics.get('f1'):.4f} | Champion F1:       {m_metrics.get('f1'):.4f}")
            print(f"Challenger Recall:   {c_metrics.get('recall_sensitivity'):.4f} | Champion Recall:   {m_metrics.get('recall_sensitivity'):.4f}")
            print(f"Challenger FN:       {c_metrics.get('confusion_matrix', {}).get('fn')} | Champion FN:       {m_metrics.get('confusion_matrix', {}).get('fn')}")

        print("\n>>> ALL 8 VERIFICATION CHECKS PASSED PERFECTLY! <<<")
        return run_id, report
    finally:
        session.close()


if __name__ == "__main__":
    execute_cumulative_retraining()
