"""Populate the 40,001 synthetic training samples for the surviving completed baseline run.
Run: ilrun-20261013T171913Z-29dc570c33
Source: backend/artifacts/incremental_learning/runs/ilrun-20261013T171913Z-29dc570c33/smote_training_records.csv
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import pandas as pd
from sqlalchemy import text

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import engine, get_session
from app.ml.verified_pipeline import MODEL_FEATURE_COLUMNS
from app.models import SyntheticTrainingSample

RUN_ID = "ilrun-20261013T042318Z-7579f93cb7"
CSV_PATH = BACKEND_ROOT / "artifacts" / "incremental_learning" / "runs" / RUN_ID / "smote_training_records.csv"


def restore_samples():
    print(f"Reading CSV from: {CSV_PATH}")
    assert CSV_PATH.is_file(), f"CSV file not found: {CSV_PATH}"

    session = get_session()
    try:
        existing = session.query(SyntheticTrainingSample).filter_by(training_run_id=RUN_ID).count()
        if existing == 40001:
            print("Already 40,001 synthetic samples present in database.")
            return

        print("Loading synthetic rows from CSV...")
        df = pd.read_csv(CSV_PATH)
        synth_df = df[df["is_synthetic"] == True].copy()
        print(f"Synthetic rows found: {len(synth_df)}")
        assert len(synth_df) == 40001, f"Expected 40001 synthetic rows, got {len(synth_df)}"

        # Prepare records for bulk insert
        features_records = synth_df[list(MODEL_FEATURE_COLUMNS)].to_dict(orient="records")
        targets = synth_df["prediction"].astype(int).tolist()

        dataset_sha = synth_df["dataset_sha256"].iloc[0] if "dataset_sha256" in synth_df.columns else None

        records_to_insert = []
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        for idx, (feat_dict, tgt) in enumerate(zip(features_records, targets)):
            normalized = {}
            for k, v in feat_dict.items():
                if pd.isna(v):
                    normalized[k] = None
                else:
                    normalized[k] = v
            records_to_insert.append({
                "source_batch_id": RUN_ID,
                "generation_method": "SMOTENC",
                "is_synthetic": True,
                "training_only": True,
                "feature_values": json.dumps(normalized),
                "target": tgt,
                "created_at": now,
                "training_run_id": RUN_ID,
                "dataset_path": str(CSV_PATH),
                "dataset_sha256": dataset_sha,
                "sample_index": idx,
            })

        print(f"Inserting {len(records_to_insert)} records into synthetic_training_samples in chunks...")
        chunk_size = 5000
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                # Clear any stale rows for this run
                conn.execute(
                    text("DELETE FROM synthetic_training_samples WHERE training_run_id = :rid"),
                    {"rid": RUN_ID},
                )
                for i in range(0, len(records_to_insert), chunk_size):
                    chunk = records_to_insert[i : i + chunk_size]
                    conn.execute(
                        text("""
                            INSERT INTO synthetic_training_samples (
                                source_batch_id, generation_method, is_synthetic, training_only,
                                feature_values, target, created_at, training_run_id,
                                dataset_path, dataset_sha256, sample_index
                            ) VALUES (
                                :source_batch_id, :generation_method, :is_synthetic, :training_only,
                                CAST(:feature_values AS json), :target, :created_at, :training_run_id,
                                :dataset_path, :dataset_sha256, :sample_index
                            )
                        """),
                        chunk,
                    )
                    print(f"Inserted chunk {i} to {i + len(chunk)}...")
                trans.commit()
                print("Commit successful!")
            except Exception as e:
                trans.rollback()
                print(f"Error during insert: {e}")
                raise

        count_after = session.query(SyntheticTrainingSample).filter_by(training_run_id=RUN_ID).count()
        print(f"Total synthetic samples in DB now: {count_after}")
        assert count_after == 40001
    finally:
        session.close()


if __name__ == "__main__":
    restore_samples()
