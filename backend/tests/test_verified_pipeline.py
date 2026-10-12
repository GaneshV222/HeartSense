from pathlib import Path
import sys
from functools import lru_cache

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import BASELINE_FEATURE_COLS
from app.ml.verified_pipeline import (
    MODEL_FEATURE_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    _fit_preprocessor_and_smote_before_split,
    _fit_preprocessor_and_smote,
    load_and_validate_dataset,
    prepare_verified_features,
    validate_temporal_workbook,
)


DATASET_PATH = BACKEND_ROOT / "data" / "heart_disease_prediction_2026.csv"
WORKBOOK_PATH = BACKEND_ROOT / "data" / "temporal_dataset.xlsx"


@lru_cache(maxsize=1)
def get_visits_and_report():
    return load_and_validate_dataset(DATASET_PATH)


@lru_cache(maxsize=1)
def get_feature_data():
    visits, _ = get_visits_and_report()
    return prepare_verified_features(visits)


def test_verified_data_and_model_schema():
    visits, report = get_visits_and_report()
    workbook_report = validate_temporal_workbook(WORKBOOK_PATH, visits)

    assert len(MODEL_FEATURE_COLUMNS) == 34
    assert len(set(MODEL_FEATURE_COLUMNS)) == 34
    assert list(MODEL_FEATURE_COLUMNS[:24]) == BASELINE_FEATURE_COLS
    assert list(MODEL_FEATURE_COLUMNS[24:]) == list(TEMPORAL_FEATURE_COLUMNS)
    assert report["rows"] == 100001
    assert report["class_distribution"] == {"0": 70001, "1": 30000}
    assert report["duplicate_rows"] == report["duplicate_visit_keys"] == 0
    assert set(workbook_report["missing_24_base_features"]) == set(BASELINE_FEATURE_COLS)
    assert workbook_report["contains_target"] is False
    assert workbook_report["used_as_training_source"] is False
    assert workbook_report["patient_ids_missing_from_workbook"] == 0
    assert workbook_report["patient_ids_not_in_visits"] == 0
    assert workbook_report["single_visit_patients_with_claimed_prior_values"] == 99960
    assert all(
        check["conflicts"] == 0
        for check in workbook_report["current_measurement_checks"].values()
        if "conflicts" in check
    )


def test_temporal_features_use_only_same_patient_previous_visit():
    visits, _ = get_visits_and_report()
    featured, labels = get_feature_data()
    first_rows = featured.drop_duplicates("patient_id", keep="first")

    assert len(featured) == len(labels) == len(visits)
    assert first_rows[list(TEMPORAL_FEATURE_COLUMNS)].isna().all().all()

    multi_id = (
        visits.groupby("patient_id").size().loc[lambda counts: counts > 1].index[0]
    )
    patient_visits = visits.loc[visits["patient_id"] == multi_id].sort_values(
        ["visit_date", "visit_number"]
    )
    patient_features = featured.loc[featured["patient_id"] == multi_id].sort_values(
        ["visit_date", "visit_number"]
    )
    current = patient_visits.iloc[1]
    previous = patient_visits.iloc[0]
    generated = patient_features.iloc[1]

    for variable in (
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "ldl",
        "hdl",
        "bmi",
        "hba1c",
        "resting_heart_rate",
    ):
        if np.isfinite(current[variable]) and np.isfinite(previous[variable]):
            assert generated[f"delta_{variable}"] == current[variable] - previous[variable]


def test_smote_is_applied_before_split_to_complete_dataset():
    featured, labels = get_feature_data()
    _, matrix, resampled_labels, records, metadata, smote_info = (
        _fit_preprocessor_and_smote_before_split(featured, labels)
    )

    assert smote_info["samples_before"] == 100001
    assert smote_info["before"] == {"0": 70001, "1": 30000}
    assert smote_info["after"] == {"0": 70001, "1": 70001}
    assert smote_info["samples_after"] == 140002
    assert smote_info["synthetic_samples_created"] == 40001
    assert smote_info["applied_to"] == "full 100001-row dataset before train/validation/test split"
    assert len(matrix) == len(resampled_labels) == len(records) == len(metadata)
    assert int(metadata["is_synthetic"].sum()) == 40001
    assert list(records.columns) == list(MODEL_FEATURE_COLUMNS)


def test_training_only_smote_helper_still_balances_sample_partitions():
    featured, labels = get_feature_data()
    class_zero_indices = [index for index, value in enumerate(labels) if value == 0][:40]
    class_one_indices = [index for index, value in enumerate(labels) if value == 1][:12]
    sample_indices = np.asarray(class_zero_indices + class_one_indices)
    training_sample = featured.iloc[sample_indices]
    training_labels = labels.iloc[sample_indices]
    _, resampled, resampled_labels, smote_info = _fit_preprocessor_and_smote(
        training_sample[list(MODEL_FEATURE_COLUMNS)], training_labels
    )

    assert smote_info["before"] == {
        str(label): int(count)
        for label, count in training_labels.value_counts().sort_index().items()
    }
    assert smote_info["after"]["0"] == smote_info["after"]["1"]
    assert len(resampled) == len(resampled_labels)
