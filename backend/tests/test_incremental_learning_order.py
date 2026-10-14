from pathlib import Path
import sys
from io import BytesIO
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.verified_pipeline import (
    MODEL_FEATURE_COLUMNS,
    _candidate_selection_scores,
    _prepare_patient_level_partitions,
    _sample_then_split,
)
from app.config import BASELINE_FEATURE_COLS, DATASET_PATH
from app.services import incremental_learning as incremental
from app.services.incremental_learning import _restore_simple_imputer_runtime_dtype
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def test_production_pipeline_partitions_patients_before_training_only_smote():
    visits = pd.read_csv(DATASET_PATH)
    features = visits.drop(columns=["prediction"]).copy()
    features["visit_number"] = 1
    features["visit_date"] = pd.Timestamp("2026-01-01")
    for column in MODEL_FEATURE_COLUMNS:
        if column not in features:
            features[column] = pd.NA
    features = features[["patient_id", "visit_number", "visit_date", *MODEL_FEATURE_COLUMNS]]
    labels = visits["prediction"].astype(int)

    result = _prepare_patient_level_partitions(features, labels)

    assert result["patient_overlaps"] == {
        "training_validation": 0,
        "training_test": 0,
        "validation_test": 0,
    }
    assert result["sampler_info"]["applied_to"] == "training partition only"
    assert result["sampler_info"]["samples_before"] == result["partitions"]["training"]["real_rows"]
    assert result["synthetic_generated_count"] == int(
        result["training_metadata"]["is_synthetic"].sum()
    )
    assert result["sampler_info"]["synthetic_training_partition_count"] == result["synthetic_generated_count"]
    assert result["sampler_info"]["synthetic_validation_partition_count"] == 0
    assert result["sampler_info"]["synthetic_test_partition_count"] == 0


def test_manual_batch_validation_matches_temporal_rows_by_identity(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    rows = []
    for patient_id, systolic in (("patient-a", 120), ("patient-b", 130)):
        row = {
            column: 1 for column in BASELINE_FEATURE_COLS
        }
        row.update({
            "patient_id": patient_id,
            "visit_number": 2,
            "visit_date": "2026-01-01",
            "prediction": 1,
            "age": 55,
            "bmi": 25,
            "diastolic_bp": 80,
            "systolic_bp": systolic,
            "resting_heart_rate": 70,
            "max_heart_rate": 150,
            "cholesterol": 200,
            "hdl": 50,
            "ldl": 100,
            "hba1c": 6,
            "oldpeak": 1,
            "num_major_vessels": 1,
            "stress_level": 5,
            "gender": "Female",
            "chest_pain_type": "Asymptomatic",
            "fasting_blood_sugar": "false",
            "diabetes": "false",
            "resting_ecg": "Normal",
            "exercise_angina": "false",
            "st_slope": "Flat",
            "thalassemia": "Normal",
            "smoking": "false",
            "family_history": "false",
            "physical_activity": "Low",
        })
        rows.append(row)
    clinical = pd.DataFrame(rows).to_csv(index=False).encode()
    # Intentionally reverse workbook rows. Subject identifiers, not row order,
    # establish the pair; current measures agree after identity joining.
    temporal = pd.DataFrame([
        {"subject_id": "patient-b", "current_systolic_bp": 130},
        {"subject_id": "patient-a", "current_systolic_bp": 120},
    ])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        temporal.to_excel(writer, index=False)

    result = incremental.validate_batch(clinical, "batch.csv", buffer.getvalue(), "batch.xlsx")

    assert result["report"]["accepted_rows"] == 2
    assert result["report"]["temporal_validation"]["row_order_used_for_matching"] is False
    assert result["report"]["temporal_validation"]["current_measurement_conflicts"] == 0


def test_manual_batch_rejects_unsupported_and_duplicate_visits(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    row = {
        column: 1 for column in BASELINE_FEATURE_COLS
    }
    row.update({
        "patient_id": "patient-a", "visit_number": 1, "visit_date": "2026-01-01",
        "prediction": 0, "age": 55, "bmi": 25, "systolic_bp": 120,
        "diastolic_bp": 80, "resting_heart_rate": 70, "max_heart_rate": 150,
        "cholesterol": 200, "hdl": 50, "ldl": 100, "hba1c": 6,
        "oldpeak": 1, "num_major_vessels": 1, "stress_level": 5,
        "gender": "Female", "chest_pain_type": "Asymptomatic",
        "fasting_blood_sugar": "false", "diabetes": "false", "resting_ecg": "Normal",
        "exercise_angina": "false", "st_slope": "Flat", "thalassemia": "Normal",
        "smoking": "false", "family_history": "false", "physical_activity": "Low",
    })
    duplicate_csv = pd.DataFrame([row, row]).to_csv(index=False).encode()
    result = incremental.validate_batch(duplicate_csv, "duplicate.csv")
    assert result["report"]["accepted_rows"] == 1
    assert result["report"]["duplicate_rows"] == 1
    with pytest.raises(ValueError, match="Unsupported columns: unsupported"):
        incremental.validate_batch(
            pd.DataFrame([{**row, "unsupported": "value"}]).to_csv(index=False).encode(),
            "unsupported.csv",
        )


def _supported_visit_row(patient_id="patient-a"):
    row = {column: 1 for column in BASELINE_FEATURE_COLS}
    row.update({
        "patient_id": patient_id, "visit_number": 1, "visit_date": "2026-01-01",
        "prediction": 0, "age": 55, "bmi": 25, "systolic_bp": 120,
        "diastolic_bp": 80, "resting_heart_rate": 70, "max_heart_rate": 150,
        "cholesterol": 200, "hdl": 50, "ldl": 100, "hba1c": 6,
        "oldpeak": 1, "num_major_vessels": 1, "stress_level": 5,
        "gender": "Female", "chest_pain_type": "Asymptomatic",
        "fasting_blood_sugar": "false", "diabetes": "false",
        "resting_ecg": "Normal", "exercise_angina": "false", "st_slope": "Flat",
        "thalassemia": "Normal", "smoking": "false", "family_history": "false",
        "physical_activity": "Low",
    })
    return row


def test_uploaded_clinical_header_aliases_normalize_to_authoritative_schema(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    row = _supported_visit_row("patient-alias")
    aliases = {
        "patient_id": "subject_id", "visit_number": "encounter_no", "visit_date": "encounter_date",
        "age": "age_years", "gender": "sex", "bmi": "body_mass_index",
        "chest_pain_type": "cp_category", "systolic_bp": "sbp_mmhg",
        "diastolic_bp": "dbp_mmhg", "resting_heart_rate": "rest_hr_bpm",
        "max_heart_rate": "peak_hr_bpm", "cholesterol": "total_chol_mgdl",
        "hdl": "hdl_chol_mgdl", "ldl": "ldl_chol_mgdl",
        "fasting_blood_sugar": "fbs_elevated", "hba1c": "glycated_hb_pct",
        "diabetes": "diabetic_flag", "resting_ecg": "rest_ecg_result",
        "exercise_angina": "exertional_angina", "oldpeak": "st_depression",
        "st_slope": "st_segment_slope", "num_major_vessels": "major_vessel_count",
        "thalassemia": "thal_status", "smoking": "current_smoker",
        "family_history": "family_cvd_history", "prediction": "heart_disease_label",
    }
    renamed = {aliases.get(column, column): value for column, value in row.items()}
    # Exercise harmless header case and surrounding-space normalization too.
    renamed = {f" {key.upper()} " if key == "subject_id" else key: value for key, value in renamed.items()}
    result = incremental.validate_batch(pd.DataFrame([renamed]).to_csv(index=False).encode(), "batch_001_clinical.csv")
    report = result["report"]
    assert report["accepted_rows"] == 1
    assert report["validation_status"] == "valid"
    assert report["column_mappings"]["heart_disease_label"] == "prediction"
    assert report["column_mappings"]["sex"] == "gender"
    assert result["accepted"].iloc[0]["prediction"] == 0
    assert result["accepted"].iloc[0]["oldpeak"] == 1


def test_unmapped_activity_stress_and_conflicting_aliases_are_actionable(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    row = _supported_visit_row()
    row.pop("physical_activity")
    row.pop("stress_level")
    row["activity_level"] = "active"
    row["stress_score"] = 7
    result = incremental.validate_batch(pd.DataFrame([row]).to_csv(index=False).encode(), "unmapped.csv")
    assert result["report"]["accepted_rows"] == 0
    assert any("physical_activity value 'active' is unsupported" in issue for issue in result["report"]["row_errors"][0]["errors"])

    conflicting = _supported_visit_row()
    conflicting["subject_id"] = "different-patient"
    with pytest.raises(ValueError, match="Conflicting columns.*patient_id.*subject_id"):
        incremental.validate_batch(pd.DataFrame([conflicting]).to_csv(index=False).encode(), "conflict.csv")


def test_invalid_categories_are_rejected_instead_of_silently_kept(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    row = _supported_visit_row()
    row["gender"] = "unknown-code"
    result = incremental.validate_batch(pd.DataFrame([row]).to_csv(index=False).encode(), "invalid-category.csv")
    assert result["report"]["accepted_rows"] == 0
    assert result["report"]["validation_status"] == "invalid"
    assert any("gender value" in issue for issue in result["report"]["row_errors"][0]["errors"])


def test_source_dataset_defines_activity_categories_and_stress_scale():
    source = pd.read_csv(DATASET_PATH, usecols=["physical_activity", "stress_level"])
    assert set(source["physical_activity"].dropna().unique()) == {"Low", "Moderate", "High"}
    assert str(source["physical_activity"].dtype) in {"object", "str"}
    assert source["physical_activity"].isna().sum() == 1964
    assert pd.api.types.is_numeric_dtype(source["stress_level"].dtype)
    assert source["stress_level"].min() == 1
    assert source["stress_level"].max() == 10
    assert source["stress_level"].isna().sum() == 1476
    assert source["stress_level"].dropna().between(1, 10).all()
    assert source["physical_activity"].value_counts().to_dict() == {
        "Moderate": 41317, "Low": 36596, "High": 20124
    }


def test_five_hundred_row_activity_and_stress_alias_batch_normalizes_and_profiles(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    rows = []
    activities = ["Low", "Moderate", "High"]
    for index in range(500):
        row = _supported_visit_row(f"batch-patient-{index:04d}")
        row.pop("physical_activity")
        row.pop("stress_level")
        row["activity_level"] = activities[index % len(activities)]
        row["stress_score"] = str(index % 10 + 1)
        row["chest_pain_type"] = "Non Anginal Pain" if index in (272, 487) else "Asymptomatic"
        row["visit_date"] = f"2026-01-{index % 28 + 1:02d}"
        if index % 100 == 0:
            row["activity_level"] = None
        if index % 125 == 0:
            row["stress_score"] = None
        rows.append(row)
    result = incremental.validate_batch(
        pd.DataFrame(rows).to_csv(index=False).encode(), "batch_001_clinical.csv"
    )
    report = result["report"]
    assert report["source_rows"] == report["accepted_rows"] == 500
    assert report["validation_status"] == "valid"
    assert report["column_mappings"]["activity_level"] == "physical_activity"
    assert report["column_mappings"]["stress_score"] == "stress_level"
    assert report["source_column_profiles"]["stress_score"]["dtype"] in {"int64", "float64"}
    assert report["source_column_profiles"]["stress_score"]["numeric_min"] == 1
    assert report["source_column_profiles"]["stress_score"]["numeric_max"] == 10
    assert result["accepted"]["physical_activity"].isna().sum() == 5
    assert result["accepted"]["stress_level"].isna().sum() == 4
    assert set(result["accepted"]["physical_activity"].dropna().unique()) == {"Low", "Moderate", "High"}
    assert result["accepted"]["stress_level"].dropna().between(1, 10).all()
    assert (result["accepted"]["chest_pain_type"] == "Non-anginal Pain").sum() == 2


def test_stress_score_outside_observed_source_scale_is_rejected(monkeypatch):
    monkeypatch.setattr(incremental, "_existing_visit_keys", lambda frame: set())
    row = _supported_visit_row()
    row.pop("stress_level")
    row["stress_score"] = 11
    result = incremental.validate_batch(pd.DataFrame([row]).to_csv(index=False).encode(), "bad-stress.csv")
    assert result["report"]["accepted_rows"] == 0
    assert any("observed stress_level scale of 1 to 10" in issue for issue in result["report"]["row_errors"][0]["errors"])


def test_candidate_selection_uses_all_validation_metrics(monkeypatch):
    monkeypatch.delenv("HEARTSENSE_IL_METRIC_WEIGHTS", raising=False)
    metrics = {
        "accuracy_only": {
            key: (0.99 if key == "accuracy" else 0.60)
            for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
        },
        "balanced": {
            key: 0.90
            for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
        },
    }
    scores = _candidate_selection_scores(metrics)
    assert scores["balanced"] > scores["accuracy_only"]


def test_old_simple_imputer_runtime_state_can_be_restored(monkeypatch):
    # temporal_pipeline installs a compatibility descriptor at import time. Remove
    # it here to reproduce a deserialized sklearn 1.7 object in the newer runtime.
    monkeypatch.delattr(SimpleImputer, "_fill_dtype", raising=False)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit([[2.0], [4.0]])
    preprocessor = Pipeline([("imputer", imputer)])

    _restore_simple_imputer_runtime_dtype(preprocessor)

    assert imputer._fill_dtype == imputer.statistics_.dtype
    assert preprocessor.transform([[None]])[0, 0] == 3.0
