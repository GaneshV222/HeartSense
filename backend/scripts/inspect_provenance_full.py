import sys
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import get_session
from app.models import IncrementalLearningBatch, IncrementalTrainingRun, PatientVisit, SyntheticTrainingSample

session = get_session()
try:
    print("=== SUMMARY COUNTS ===")
    total_visits = session.query(PatientVisit).count()
    real_visits = session.query(PatientVisit).filter(PatientVisit.is_synthetic == False).count()
    synth_visits = session.query(PatientVisit).filter(PatientVisit.is_synthetic == True).count()
    synth_samples = session.query(SyntheticTrainingSample).count()
    print(f"Total visits: {total_visits}")
    print(f"Real visits: {real_visits}")
    print(f"Synthetic visits in patient_visits: {synth_visits}")
    print(f"Synthetic samples in synthetic_training_samples: {synth_samples}")

    print("\n=== PATIENT VISITS BY SOURCE DATASET ===")
    sources = session.execute(text("SELECT source_dataset, count(*) FROM patient_visits GROUP BY source_dataset")).fetchall()
    for s, c in sources:
        print(f"  {s}: {c}")

    print("\n=== BATCHES ===")
    for b in session.query(IncrementalLearningBatch).all():
        print(f"  Batch: {b.batch_id}, Status: {b.status}, Source: {b.source_filename}, Accepted: {b.accepted_rows}")

    print("\n=== RUNS ===")
    for r in session.query(IncrementalTrainingRun).all():
        print(f"  Run: {r.training_run_id}, Status: {r.status}, Dataset rows: {r.dataset_rows}, Decision: {r.decision}")

    print("\n=== PATIENT OVERLAP CHECKS ===")
    q_overlap = """
        SELECT count(distinct pv1.patient_id)
        FROM patient_visits pv1
        JOIN patient_visits pv2 ON pv1.patient_id = pv2.patient_id
        WHERE pv1.source_dataset = :s1 AND pv2.source_dataset = :s2
    """
    o_1_2 = session.execute(text(q_overlap), {"s1": "incremental:il-9e8b62a8cd4cf7302a5b", "s2": "incremental:il-a451e3fc726d3789414b"}).scalar()
    o_1_base = session.execute(text(q_overlap), {"s1": "incremental:il-9e8b62a8cd4cf7302a5b", "s2": "heart_disease_prediction_2026.csv"}).scalar()
    o_2_base = session.execute(text(q_overlap), {"s1": "incremental:il-a451e3fc726d3789414b", "s2": "heart_disease_prediction_2026.csv"}).scalar()
    print(f"  Overlap between Batch 1 and Batch 2: {o_1_2}")
    print(f"  Overlap between Batch 1 and Baseline: {o_1_base}")
    print(f"  Overlap between Batch 2 and Baseline: {o_2_base}")

    print("\n=== FOREIGN KEY & DERIVED RECORD CHECKS ===")
    # Predictions
    pred_count = session.execute(text("""
        SELECT count(*) FROM predictions p
        JOIN patient_visits pv ON p.visit_id = pv.id
        WHERE pv.source_dataset LIKE 'incremental:%'
    """)).scalar()
    print(f"  Predictions pointing to incremental visits: {pred_count}")

    # Temporal patient data
    temporal_count = session.execute(text("""
        SELECT count(*) FROM temporal_patient_data tpd
        JOIN patient_visits pv ON tpd.current_visit_id = pv.id
        WHERE pv.source_dataset LIKE 'incremental:%'
    """)).scalar()
    print(f"  Temporal records pointing to incremental current_visit_id: {temporal_count}")

    # Patients that ONLY exist in incremental batches
    inc_only_patients = session.execute(text("""
        SELECT count(distinct p.patient_id)
        FROM patients p
        WHERE NOT EXISTS (
            SELECT 1 FROM patient_visits pv
            WHERE pv.patient_id = p.patient_id
              AND pv.source_dataset = 'heart_disease_prediction_2026.csv'
        )
    """)).scalar()
    print(f"  Patients with NO baseline visits (only incremental): {inc_only_patients}")

finally:
    session.close()
