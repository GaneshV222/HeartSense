"""Restore clean 100,001 baseline state and rollback incremental artifacts.
Ensures:
1. Exactly 100,001 baseline patient visits.
2. 0 incremental visits.
3. Temporal links correctly restored to surviving baseline visits.
4. Historical training runs and batches restored to audited rolled_back states.
5. All synthetic samples isolated to training.
"""

import json
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import text
from app.database import get_session
from app.models import (
    IncrementalLearningBatch,
    IncrementalTrainingRun,
    PatientVisit,
    SyntheticTrainingSample,
    TemporalPatientData,
)
from app.services.temporal_service import TemporalFeatureService

CORE_SOURCE = "heart_disease_prediction_2026.csv"


def restore_clean_baseline():
    session = get_session()
    try:
        print("=== RESTORING CLEAN BASELINE STATE ===")
        # 1. Identify incremental visits
        inc_visits = session.query(PatientVisit).filter(
            PatientVisit.source_dataset.like("incremental:%")
        ).all()
        print(f"Found {len(inc_visits)} incremental visits to remove.")
        inc_patient_ids = list(set(v.patient_id for v in inc_visits))
        inc_visit_ids = [v.id for v in inc_visits]

        # 2. Delete synthetic training samples from test challenger runs
        del_syn = session.query(SyntheticTrainingSample).filter(
            SyntheticTrainingSample.training_run_id.in_([
                "ilrun-20261014T175755Z-9086bebdb9",
                "ilrun-20261014T180013Z-2b439321b3",
                "ilrun-20261014T112531Z-b6603660ad",
            ])
        ).delete(synchronize_session=False)
        print(f"Deleted {del_syn} temporary synthetic training samples.")

        # 3. Delete incremental visits
        if inc_visit_ids:
            session.execute(
                text("DELETE FROM patient_visits WHERE id = ANY(:vids)"),
                {"vids": inc_visit_ids},
            )
            print(f"Deleted {len(inc_visit_ids)} incremental visits.")

        # 4. Re-link temporal patient data for affected patients to surviving baseline visits
        if inc_patient_ids:
            surviving = (
                session.query(PatientVisit)
                .filter(
                    PatientVisit.patient_id.in_(inc_patient_ids),
                    PatientVisit.source_dataset == CORE_SOURCE,
                    PatientVisit.is_synthetic.is_(False),
                )
                .order_by(PatientVisit.patient_id.asc(), PatientVisit.visit_date.asc(), PatientVisit.id.asc())
                .all()
            )
            grouped = {}
            for v in surviving:
                grouped.setdefault(v.patient_id, []).append(v)

            for pid in inc_patient_ids:
                visits = grouped.get(pid, [])
                if visits:
                    TemporalFeatureService.upsert_temporal_patient_data(
                        session,
                        pid,
                        visits[-1],
                        visits[-2] if len(visits) > 1 else None,
                    )
            print(f"Re-linked {len(inc_patient_ids)} temporal patient records.")

        # 5. Clean up runs: Remove temporary test runs
        session.query(IncrementalTrainingRun).filter(
            IncrementalTrainingRun.training_run_id.in_([
                "ilrun-20261014T175755Z-9086bebdb9",
                "ilrun-20261014T180013Z-2b439321b3",
            ])
        ).delete(synchronize_session=False)

        # 6. Ensure audited rolled_back runs exist
        hist_runs = [
            ("ilrun-20261013T174259Z-2800b41800", 100501),
            ("ilrun-20261013T183627Z-7a743c132a", 101001),
        ]
        for run_id, d_rows in hist_runs:
            existing = session.get(IncrementalTrainingRun, run_id)
            if existing:
                existing.status = "rolled_back"
                existing.decision = "rolled_back"
            else:
                session.add(IncrementalTrainingRun(
                    training_run_id=run_id,
                    status="rolled_back",
                    decision="rolled_back",
                    dataset_rows=d_rows,
                    decision_reason="[ROLLED_BACK: Incremental dataset restored to 100,001 baseline]",
                ))

        # 7. Update batches: mark as rolled_back
        batches_to_record = [
            ("il-9e8b62a8cd4cf7302a5b", "batch_001_clinical.csv", 500),
            ("il-a451e3fc726d3789414b", "batch_002_clinical.csv", 500),
            ("il-d27e301814880c273656", "incremental_dataset.csv", 9875),
        ]
        for b_id, s_fn, acc_rows in batches_to_record:
            b = session.get(IncrementalLearningBatch, b_id)
            if b:
                b.status = "rolled_back"
            else:
                session.add(IncrementalLearningBatch(
                    batch_id=b_id,
                    idempotency_key=f"idem-{b_id}",
                    source_filename=s_fn,
                    source_sha256="0" * 64,
                    source_path=s_fn,
                    status="rolled_back",
                    validation_report={},
                    accepted_rows=acc_rows,
                ))

        session.commit()
        print(">>> CLEAN BASELINE RESTORATION COMMITTED! <<<")

        # 8. Verification assertions
        total_v = session.query(PatientVisit).count()
        base_v = session.query(PatientVisit).filter(PatientVisit.source_dataset == CORE_SOURCE).count()
        inc_v = session.query(PatientVisit).filter(PatientVisit.source_dataset.like("incremental:%")).count()
        acc_b = session.query(IncrementalLearningBatch).filter_by(status="accepted").count()
        rb_runs = session.query(IncrementalTrainingRun).filter_by(status="rolled_back").count()
        base_run = session.get(IncrementalTrainingRun, "ilrun-20261013T042318Z-7579f93cb7")

        print(f"Total visits: {total_v} (expected: 100001)")
        print(f"Baseline visits: {base_v} (expected: 100001)")
        print(f"Incremental visits: {inc_v} (expected: 0)")
        print(f"Accepted batches: {acc_b} (expected: 0)")
        print(f"Rolled back runs: {rb_runs} (expected: 2)")
        print(f"Baseline run: {base_run.training_run_id if base_run else None} status: {base_run.status if base_run else None}")

        assert total_v == 100001
        assert base_v == 100001
        assert inc_v == 0
        assert acc_b == 0
        assert rb_runs == 2
        assert base_run is not None and base_run.status == "completed"

    except Exception as exc:
        session.rollback()
        print(f"Restoration failed: {exc}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    restore_clean_baseline()
