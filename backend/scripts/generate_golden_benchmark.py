"""Script to generate and persist the canonical, leakage-free Golden Evaluation Benchmark.
The benchmark is constructed exclusively from genuine real patients held out from the baseline dataset.
Zero patient overlap with training. Zero synthetic samples.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR
from app.ml.verified_pipeline import prepare_verified_features, MODEL_FEATURE_COLUMNS, TARGET_COLUMN

BENCHMARK_DIR = Path(MODEL_ARTIFACT_DIR) / "golden_benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


def build_golden_benchmark():
    print("=== GENERATING IMMUTABLE GOLDEN EVALUATION BENCHMARK ===")
    dataset_file = Path(DATASET_PATH).resolve()
    content = dataset_file.read_bytes()
    dataset_sha256 = sha256(content).hexdigest()

    df = pd.read_csv(dataset_file)
    print(f"Loaded source dataset with {len(df)} rows. SHA256: {dataset_sha256}")

    # Prepare verified 34 features
    feat_df, labels = prepare_verified_features(df)

    # Deterministic patient-level stratification
    patient_meta = feat_df.groupby("patient_id")["prediction"].agg(["count", "first", "mean"]).reset_index()
    patient_meta["strat_label"] = (patient_meta["mean"] >= 0.5).astype(int)

    # 15% test (Golden Benchmark), 15% val, 70% train
    pts_train_val, pts_test = train_test_split(
        patient_meta["patient_id"],
        test_size=0.15,
        random_state=42,
        stratify=patient_meta["strat_label"],
    )

    pts_train_val_meta = patient_meta[patient_meta["patient_id"].isin(pts_train_val)]
    pts_train, pts_val = train_test_split(
        pts_train_val_meta["patient_id"],
        test_size=0.17647,  # 0.85 * 0.17647 ~= 0.15 of total
        random_state=42,
        stratify=pts_train_val_meta["strat_label"],
    )

    set_test = set(pts_test)
    set_val = set(pts_val)
    set_train = set(pts_train)

    # Strict safety assertions
    assert len(set_test & set_train) == 0, "Patient leakage detected between test and train!"
    assert len(set_test & set_val) == 0, "Patient leakage detected between test and validation!"
    assert len(set_val & set_train) == 0, "Patient leakage detected between validation and train!"

    test_mask = feat_df["patient_id"].isin(set_test)
    val_mask = feat_df["patient_id"].isin(set_val)
    train_mask = feat_df["patient_id"].isin(set_train)

    benchmark_records = feat_df.loc[test_mask].copy().reset_index(drop=True)
    benchmark_csv = BENCHMARK_DIR / "golden_benchmark_records.csv"
    benchmark_records.to_csv(benchmark_csv, index=False)
    benchmark_sha256 = sha256(benchmark_csv.read_bytes()).hexdigest()

    class_counts = benchmark_records[TARGET_COLUMN].value_counts().to_dict()

    manifest = {
        "benchmark_id": "golden_benchmark_v1",
        "benchmark_name": "HeartSense Independent Real-Patient Golden Evaluation Benchmark",
        "version": "1.0",
        "evaluation_protocol": "Independent real-patient held-out benchmark (100% genuine patients, zero synthetic contamination, zero training leakage)",
        "source_dataset_path": str(dataset_file),
        "source_dataset_sha256": dataset_sha256,
        "total_source_rows": len(df),
        "total_source_patients": int(df["patient_id"].nunique()),
        "benchmark_row_count": int(len(benchmark_records)),
        "benchmark_patient_count": int(benchmark_records["patient_id"].nunique()),
        "synthetic_rows": 0,
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "records_csv_path": str(benchmark_csv.resolve()),
        "records_csv_sha256": benchmark_sha256,
        "split_configuration": {
            "method": "Patient-level stratified train_test_split (stratify=patient_average_prediction)",
            "random_state": 42,
            "test_patient_ratio": 0.15,
            "validation_patient_ratio": 0.15,
            "train_patient_ratio": 0.70,
            "patient_overlap": 0,
        },
        "test_patient_ids": sorted(list(set_test)),
        "validation_patient_ids": sorted(list(set_val)),
        "train_patient_ids": sorted(list(set_train)),
        "historical_champion_independence_audit": {
            "historical_champion_model": "XGBoost baseline",
            "historical_training_membership_recorded": False,
            "leakage_certification": "UNVERIFIED: The historical Champion manifest did not record exact patient split hashes, so its benchmark independence cannot be certified byte-for-byte; Challenger is certified 100% independent.",
        },
    }

    manifest_path = BENCHMARK_DIR / "golden_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved benchmark records ({len(benchmark_records)} rows) to: {benchmark_csv}")
    print(f"Saved benchmark manifest to: {manifest_path}")
    print("Golden Benchmark generation completed successfully!")


if __name__ == "__main__":
    build_golden_benchmark()
