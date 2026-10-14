"""Leakage-safe training and source-scoped import for the verified HeartSense datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import (
    BASELINE_FEATURE_COLS,
    MODEL_ARTIFACT_DIR,
    RANDOM_STATE,
)

TARGET_COLUMN = "prediction"
PATIENT_COLUMN = "patient_id"
VISIT_COLUMNS = ("patient_id", "visit_number", "visit_date")
NUMERICAL_TEMPORAL_VARIABLES = (
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "ldl",
    "hdl",
    "bmi",
    "hba1c",
    "resting_heart_rate",
)
CATEGORICAL_TEMPORAL_VARIABLES = ("smoking", "physical_activity")
TEMPORAL_FEATURE_COLUMNS = tuple(
    [f"delta_{name}" for name in NUMERICAL_TEMPORAL_VARIABLES]
    + ["smoking_changed", "physical_activity_changed"]
)
MODEL_FEATURE_COLUMNS = tuple(BASELINE_FEATURE_COLS) + TEMPORAL_FEATURE_COLUMNS
CATEGORICAL_BASE_COLUMNS = (
    "gender",
    "chest_pain_type",
    "fasting_blood_sugar",
    "diabetes",
    "resting_ecg",
    "exercise_angina",
    "st_slope",
    "thalassemia",
    "smoking",
    "family_history",
    "physical_activity",
)
CATEGORICAL_FEATURE_COLUMNS = CATEGORICAL_BASE_COLUMNS
NUMERICAL_FEATURE_COLUMNS = tuple(
    name for name in MODEL_FEATURE_COLUMNS if name not in CATEGORICAL_FEATURE_COLUMNS
)
TARGET_MAPPING = {
    "0": "Lower Cardiovascular Risk",
    "1": "Higher Cardiovascular Risk",
}
TEMPORAL_SHEET_NAME = "temporal_dataset"
CLINICAL_SCREENING_BOUNDS = {
    "age": (0, 120),
    "bmi": (8, 100),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "resting_heart_rate": (20, 250),
    "max_heart_rate": (20, 250),
    "cholesterol": (0, 1000),
    "hdl": (0, 250),
    "ldl": (0, 800),
    "hba1c": (0, 30),
    "oldpeak": (-20, 20),
    "num_major_vessels": (0, 20),
    "stress_level": (0, 100),
}


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in frame.columns
    ]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns after normalisation: {duplicates}")
    return frame


def load_and_validate_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the visit CSV and reject invented or ambiguous identity, date, and label data."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Visit dataset not found: {source_path}")
    if source_path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source_path)
    elif source_path.suffix.lower() == ".csv":
        frame = pd.read_csv(source_path)
    else:
        raise ValueError(f"Unsupported visit dataset format: {source_path.suffix}")
    frame = _normalise_columns(frame)

    required = set(BASELINE_FEATURE_COLS) | set(VISIT_COLUMNS) | {TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Visit dataset is missing required columns: {missing}")

    frame = frame.replace({"": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan})
    frame[PATIENT_COLUMN] = frame[PATIENT_COLUMN].astype("string").str.strip()
    bad_ids = frame[PATIENT_COLUMN].isna() | frame[PATIENT_COLUMN].eq("")
    if bad_ids.any():
        raise ValueError(f"Visit dataset contains {int(bad_ids.sum())} missing patient identifiers")

    frame["visit_date"] = pd.to_datetime(frame["visit_date"], errors="coerce")
    if frame["visit_date"].isna().any():
        raise ValueError(
            f"Visit dataset contains {int(frame['visit_date'].isna().sum())} invalid visit dates"
        )

    visit_number = pd.to_numeric(frame["visit_number"], errors="coerce")
    invalid_visit_number = visit_number.isna() | (visit_number < 1) | (visit_number % 1 != 0)
    if invalid_visit_number.any():
        raise ValueError(
            f"Visit dataset contains {int(invalid_visit_number.sum())} invalid visit numbers"
        )
    frame["visit_number"] = visit_number.astype(int)

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    invalid_target = target.isna() | ~target.isin([0, 1]) | (target % 1 != 0)
    if invalid_target.any():
        values = sorted(frame.loc[invalid_target, TARGET_COLUMN].astype(str).unique().tolist())[:10]
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain only verified binary labels 0/1; "
            f"invalid rows={int(invalid_target.sum())}, sample values={values}"
        )
    frame[TARGET_COLUMN] = target.astype(int)

    numeric_source_columns = [
        name for name in BASELINE_FEATURE_COLS if name not in CATEGORICAL_BASE_COLUMNS
    ]
    for column in numeric_source_columns:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        infinite = converted.notna() & ~np.isfinite(converted)
        if invalid.any():
            examples = original.loc[invalid].astype(str).unique().tolist()[:5]
            raise ValueError(
                f"Column '{column}' contains non-numeric clinical values: {examples}"
            )
        if infinite.any():
            raise ValueError(
                f"Column '{column}' contains {int(infinite.sum())} infinite values"
            )
        frame[column] = converted

    duplicate_key = frame.duplicated(["patient_id", "visit_date", "visit_number"], keep=False)
    if duplicate_key.any():
        raise ValueError(
            "Visit dataset has duplicate patient/date/visit-number keys "
            f"({int(duplicate_key.sum())} rows); refusing to merge visits"
        )
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        raise ValueError(
            f"Visit dataset contains {duplicate_rows} duplicate full rows; "
            "resolve source duplicates before importing or training"
        )

    frame = _normalise_categorical_values(frame)
    frame = frame.sort_values(
        ["patient_id", "visit_date", "visit_number"], kind="stable"
    ).reset_index(drop=True)
    patient_counts = frame.groupby("patient_id", sort=False).size()
    out_of_range_counts = {
        column: int(
            (
                frame[column].notna()
                & ((frame[column] < lower) | (frame[column] > upper))
            ).sum()
        )
        for column, (lower, upper) in CLINICAL_SCREENING_BOUNDS.items()
        if column in frame.columns
    }
    invalid_ranges = {key: value for key, value in out_of_range_counts.items() if value}
    if invalid_ranges:
        raise ValueError(
            "Clinical values outside broad physiologic screening ranges require review: "
            f"{invalid_ranges}"
        )
    summary = {
        "path": str(source_path.resolve()),
        "filename": source_path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "patient_count": int(patient_counts.size),
        "single_visit_patients": int((patient_counts == 1).sum()),
        "multi_visit_patients": int((patient_counts > 1).sum()),
        "max_visits_per_patient": int(patient_counts.max()),
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "source_to_patient_visit_mapping": {
            column: column for column in BASELINE_FEATURE_COLS
        }
        | {
            "patient_id": "patient_visits.patient_id",
            "visit_number": "patient_visits.visit_number",
            "visit_date": "patient_visits.visit_date and visit_timestamp",
            "prediction": "patient_visits.prediction (supervised target, not predictor)",
        },
        "source_to_patient_mapping": {
            "patient_id": "patients.patient_id and patient_code",
            "gender": "patients.gender",
            "age": "patients.age",
        },
        "range_screening_note": (
            "Broad physiologic screening bounds flag suspicious values for review; "
            "they do not remove rows or substitute values."
        ),
        "out_of_range_value_counts": {
            column: count for column, count in out_of_range_counts.items()
        },
        "source_dtypes": {
            str(column): str(dtype) for column, dtype in frame.dtypes.items()
        },
        "class_distribution": {
            str(key): int(value)
            for key, value in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "missing_values": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
            if count
        },
        "duplicate_rows": 0,
        "duplicate_visit_keys": 0,
        "visit_date_range": [
            frame["visit_date"].min().isoformat(),
            frame["visit_date"].max().isoformat(),
        ],
        "target_varies_within_patient": int(
            (frame.groupby("patient_id")[TARGET_COLUMN].nunique() > 1).sum()
        ),
        "clinical_numeric_ranges": {
            column: {
                "min": float(frame[column].min()) if frame[column].notna().any() else None,
                "max": float(frame[column].max()) if frame[column].notna().any() else None,
            }
            for column in numeric_source_columns
        },
        "categorical_values": {
            column: {
                str(key): int(value)
                for key, value in frame[column].value_counts(dropna=False).items()
            }
            for column in CATEGORICAL_BASE_COLUMNS
        },
    }
    return frame, summary


def _normalise_categorical_values(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {
        "gender": {"male": "Male", "female": "Female"},
        "chest_pain_type": {
            "typical angina": "Typical Angina",
            "atypical angina": "Atypical Angina",
            "non-anginal pain": "Non-anginal Pain",
            "non anginal pain": "Non-anginal Pain",
            "asymptomatic": "Asymptomatic",
        },
        "resting_ecg": {
            "normal": "Normal",
            "st-t wave abnormality": "ST-T Wave Abnormality",
            "left ventricular hypertrophy": "Left Ventricular Hypertrophy",
        },
        "st_slope": {"up": "Up", "flat": "Flat", "down": "Down"},
        "thalassemia": {
            "normal": "Normal",
            "fixed defect": "Fixed Defect",
            "reversible defect": "Reversible Defect",
        },
        "physical_activity": {"low": "Low", "moderate": "Moderate", "high": "High"},
    }
    for column in CATEGORICAL_BASE_COLUMNS:
        if column not in frame:
            continue

        def normalize(value: Any) -> Any:
            if pd.isna(value):
                return np.nan
            if isinstance(value, (bool, np.bool_)):
                return str(value).lower()
            text = str(value).strip()
            if text.lower() in {"true", "false"}:
                return text.lower()
            return aliases.get(column, {}).get(text.casefold(), text)

        frame[column] = frame[column].map(normalize)
    return frame


def prepare_verified_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create only the 8 approved numeric deltas and 2 approved lifestyle-change flags."""
    ordered = frame.sort_values(
        ["patient_id", "visit_date", "visit_number"], kind="stable"
    ).copy()
    groups = ordered.groupby("patient_id", sort=False)

    for variable in NUMERICAL_TEMPORAL_VARIABLES:
        current = pd.to_numeric(ordered[variable], errors="coerce")
        previous = pd.to_numeric(groups[variable].shift(1), errors="coerce")
        ordered[f"delta_{variable}"] = current - previous

    for variable in CATEGORICAL_TEMPORAL_VARIABLES:
        current = ordered[variable].astype("string")
        previous = groups[variable].shift(1).astype("string")
        changed = current.notna() & previous.notna() & current.ne(previous)
        ordered[f"{variable}_changed"] = changed.astype("float64").where(
            current.notna() & previous.notna()
        )

    missing_features = sorted(set(MODEL_FEATURE_COLUMNS) - set(ordered.columns))
    if missing_features:
        raise ValueError(f"Unable to construct verified model features: {missing_features}")
    labels = ordered[TARGET_COLUMN].astype(int)
    return ordered, labels


def _exclude_golden_benchmark_patients(
    features: pd.DataFrame,
    labels: pd.Series,
    manifest_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Keep every fixed Golden Benchmark patient out of candidate fitting data."""
    manifest_file = Path(manifest_path) if manifest_path else (
        Path(MODEL_ARTIFACT_DIR) / "golden_benchmark" / "golden_benchmark_manifest.json"
    )
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Golden Benchmark split manifest is required before candidate training: {manifest_file}"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    benchmark_ids = {str(value) for value in manifest.get("test_patient_ids", [])}
    if not benchmark_ids:
        raise ValueError("Golden Benchmark manifest has no test patient IDs")
    benchmark_file = manifest_file.parent / "golden_benchmark_records.csv"
    if not benchmark_file.is_file():
        raise FileNotFoundError(f"Golden Benchmark records are missing: {benchmark_file}")
    benchmark_sha = hashlib.sha256(benchmark_file.read_bytes()).hexdigest()
    if benchmark_sha != manifest.get("records_csv_sha256"):
        raise ValueError("Golden Benchmark records do not match the persisted manifest fingerprint")
    benchmark_frame = pd.read_csv(benchmark_file, usecols=[PATIENT_COLUMN])
    benchmark_file_ids = set(benchmark_frame[PATIENT_COLUMN].astype("string").str.strip().dropna())
    if benchmark_file_ids != benchmark_ids:
        raise ValueError("Golden Benchmark record patient IDs do not match the manifest test split")
    if PATIENT_COLUMN not in features:
        raise ValueError("Patient identifiers are required to prevent benchmark leakage")
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts differ before benchmark isolation")

    patient_ids = features[PATIENT_COLUMN].astype("string").str.strip()
    benchmark_mask = patient_ids.isin(benchmark_ids)
    training_mask = ~benchmark_mask
    training = features.loc[training_mask].copy().reset_index(drop=True)
    training_labels = labels.loc[training_mask].copy().reset_index(drop=True)
    training_patient_ids = set(training[PATIENT_COLUMN].astype(str))
    overlap = training_patient_ids.intersection(benchmark_ids)
    if overlap:
        raise RuntimeError("Golden Benchmark patient leakage detected in candidate training rows")
    if training.empty or training_labels.nunique() != 2:
        raise ValueError("Benchmark-isolated training rows must contain both target classes")

    ordered_training_ids = sorted(training_patient_ids)
    return training, training_labels, {
        "benchmark_id": manifest.get("benchmark_id"),
        "benchmark_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "benchmark_records_sha256": benchmark_sha,
        "benchmark_test_patient_count": len(benchmark_ids),
        "benchmark_excluded_real_rows": int(benchmark_mask.sum()),
        "candidate_training_real_rows": int(len(training)),
        "candidate_training_patient_count": len(training_patient_ids),
        "candidate_training_patient_ids_sha256": hashlib.sha256(
            "\n".join(ordered_training_ids).encode("utf-8")
        ).hexdigest(),
        "patient_overlap": 0,
    }


def validate_temporal_workbook(
    workbook_path: str | Path, visits: pd.DataFrame
) -> dict[str, Any]:
    """Compare the precomputed workbook with visit-derived history; never train from it."""
    source_path = Path(workbook_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Temporal workbook not found: {source_path}")
    workbook = pd.read_excel(source_path, sheet_name=TEMPORAL_SHEET_NAME)
    workbook = _normalise_columns(workbook)
    if PATIENT_COLUMN not in workbook.columns:
        raise ValueError(
            f"Worksheet '{TEMPORAL_SHEET_NAME}' is missing patient_id"
        )
    if workbook[PATIENT_COLUMN].isna().any():
        raise ValueError(
            f"Worksheet '{TEMPORAL_SHEET_NAME}' has missing patient identifiers"
        )
    workbook[PATIENT_COLUMN] = workbook[PATIENT_COLUMN].astype("string").str.strip()
    if workbook[PATIENT_COLUMN].duplicated().any():
        raise ValueError(
            f"Worksheet '{TEMPORAL_SHEET_NAME}' has duplicate patient identifiers"
        )

    required_base = set(BASELINE_FEATURE_COLS)
    missing_base = sorted(required_base - set(workbook.columns))
    labels_available = TARGET_COLUMN in workbook.columns
    workbook_ids = set(workbook[PATIENT_COLUMN].astype(str))
    visit_ids = set(visits[PATIENT_COLUMN].astype(str))
    latest = visits.groupby(PATIENT_COLUMN, sort=False).tail(1).set_index(PATIENT_COLUMN)
    history_counts = visits.groupby(PATIENT_COLUMN, sort=False).size()
    report: dict[str, Any] = {
        "path": str(source_path.resolve()),
        "worksheet": TEMPORAL_SHEET_NAME,
        "rows": int(len(workbook)),
        "columns": list(workbook.columns),
        "workbook_column_mapping": {
            "patient_id": "identity only; excluded from predictors",
            **{
                f"{variable}_{suffix}": (
                    f"temporal_patient_data.{db_field}; comparison only, "
                    "not used as the authoritative value"
                )
                for variable in (*NUMERICAL_TEMPORAL_VARIABLES, *CATEGORICAL_TEMPORAL_VARIABLES)
                for suffix, db_field in (
                    ("prev", f"previous_{variable}"),
                    ("curr", f"current_{variable}"),
                    ("change", f"{variable}_change"),
                )
                if suffix != "change" or variable in NUMERICAL_TEMPORAL_VARIABLES
            },
            "smoking_changed": "temporal_patient_data.smoking_changed; comparison only",
            "physical_activity_changed": (
                "temporal_patient_data.physical_activity_changed; comparison only"
            ),
        },
        "missing_24_base_features": missing_base,
        "contains_target": labels_available,
        "patient_ids_missing_from_workbook": int(len(visit_ids - workbook_ids)),
        "patient_ids_not_in_visits": int(len(workbook_ids - visit_ids)),
        "used_as_training_source": False,
        "reason": (
            "The worksheet lacks required base predictors and the target. Its precomputed "
            "prior values are not used; temporal changes are derived only from genuine visits."
        ),
        "current_measurement_checks": {},
        "single_visit_patients_with_claimed_prior_values": 0,
        "multi_visit_patients": int((history_counts > 1).sum()),
    }

    checked = workbook.set_index(PATIENT_COLUMN)
    latest_by_id = latest.reindex(checked.index)
    for variable in NUMERICAL_TEMPORAL_VARIABLES:
        workbook_column = f"{variable}_curr"
        if workbook_column not in checked.columns or variable not in latest_by_id.columns:
            report["current_measurement_checks"][variable] = {
                "status": "missing source column",
            }
            continue
        excel_values = pd.to_numeric(checked[workbook_column], errors="coerce")
        visit_values = pd.to_numeric(latest_by_id[variable], errors="coerce")
        both_present = excel_values.notna() & visit_values.notna()
        conflicts = both_present & ~np.isclose(
            excel_values.fillna(0), visit_values.fillna(0), rtol=0, atol=1e-8
        )
        report["current_measurement_checks"][variable] = {
            "compared_non_missing": int(both_present.sum()),
            "matches": int((both_present & ~conflicts).sum()),
            "conflicts": int(conflicts.sum()),
            "excel_only_missing": int(excel_values.isna().sum()),
            "visits_only_missing": int(visit_values.isna().sum()),
        }

    for variable in CATEGORICAL_TEMPORAL_VARIABLES:
        workbook_column = f"{variable}_curr"
        if workbook_column not in checked.columns or variable not in latest_by_id.columns:
            report["current_measurement_checks"][variable] = {
                "status": "missing source column",
            }
            continue
        excel_values = checked[workbook_column].astype("string").str.strip().str.casefold()
        visit_values = latest_by_id[variable].astype("string").str.strip().str.casefold()
        if variable == "smoking":
            bool_values = {
                "1": "true",
                "1.0": "true",
                "yes": "true",
                "0": "false",
                "0.0": "false",
                "no": "false",
            }
            excel_values = excel_values.replace(bool_values)
            visit_values = visit_values.replace(bool_values)
        both_present = excel_values.notna() & visit_values.notna()
        conflicts = both_present & excel_values.ne(visit_values)
        report["current_measurement_checks"][variable] = {
            "compared_non_missing": int(both_present.sum()),
            "matches": int((both_present & ~conflicts).sum()),
            "conflicts": int(conflicts.sum()),
            "excel_only_missing": int(excel_values.isna().sum()),
            "visits_only_missing": int(visit_values.isna().sum()),
        }

    prior_columns = [
        f"{variable}_prev" for variable in NUMERICAL_TEMPORAL_VARIABLES
    ] + ["smoking_prev", "physical_activity_prev"]
    existing_prior_columns = [column for column in prior_columns if column in checked]
    singleton_ids = history_counts[history_counts == 1].index
    singleton_workbook = checked.reindex(singleton_ids)
    if existing_prior_columns and not singleton_workbook.empty:
        report["single_visit_patients_with_claimed_prior_values"] = int(
            singleton_workbook[existing_prior_columns].notna().any(axis=1).sum()
        )
    return report


def _make_raw_preprocessor() -> Pipeline:
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERICAL_FEATURE_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _make_postprocessor(numeric_count: int) -> ColumnTransformer:
    categorical_positions = list(
        range(numeric_count, len(MODEL_FEATURE_COLUMNS))
    )
    return ColumnTransformer(
        [
            ("numeric", "passthrough", list(range(numeric_count))),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_positions,
            ),
        ],
        remainder="drop",
    )


def _fit_preprocessor_and_smote(
    training_features: pd.DataFrame, training_labels: pd.Series
) -> tuple[Pipeline, Any, Any, dict[str, Any]]:
    raw_preprocessor = _make_raw_preprocessor()
    encoded = raw_preprocessor.fit_transform(training_features[list(MODEL_FEATURE_COLUMNS)])
    categorical_positions = list(
        range(len(NUMERICAL_FEATURE_COLUMNS), len(MODEL_FEATURE_COLUMNS))
    )
    class_counts = Counter(int(value) for value in training_labels)
    minority_count = min(class_counts.values())
    if minority_count < 2:
        raise ValueError(
            "SMOTENC requires at least two examples in each training class"
        )
    neighbor_count = min(5, minority_count - 1)
    sampler = SMOTENC(
        categorical_features=categorical_positions,
        random_state=RANDOM_STATE,
        k_neighbors=neighbor_count,
    )
    resampled, resampled_labels = sampler.fit_resample(encoded, training_labels)
    postprocessor = _make_postprocessor(len(NUMERICAL_FEATURE_COLUMNS))
    postprocessor.fit(resampled)
    transformed = postprocessor.transform(resampled)
    final_preprocessor = Pipeline(
        [
            ("raw_features", raw_preprocessor),
            ("encoded_features", postprocessor),
        ]
    )
    info = {
        "before": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "after": {
            str(k): int(v)
            for k, v in sorted(Counter(int(value) for value in resampled_labels).items())
        },
        "k_neighbors": neighbor_count,
        "resampled_rows": int(len(resampled_labels)),
    }
    return final_preprocessor, transformed, resampled_labels, info


def _inverse_smote_feature_rows(
    raw_preprocessor: ColumnTransformer,
    resampled: Any,
    source_features: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild human-readable 34-feature rows from the SMOTENC working matrix."""
    matrix = np.asarray(resampled)
    numeric_count = len(NUMERICAL_FEATURE_COLUMNS)
    numeric_values = matrix[:, :numeric_count]
    categorical_values = matrix[:, numeric_count:]

    numeric_pipeline = raw_preprocessor.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    numeric_frame = pd.DataFrame(
        scaler.inverse_transform(numeric_values),
        columns=NUMERICAL_FEATURE_COLUMNS,
    )

    categorical_pipeline = raw_preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encoder"]
    decoded_categories = encoder.inverse_transform(np.rint(categorical_values))
    categorical_frame = pd.DataFrame(
        decoded_categories,
        columns=CATEGORICAL_FEATURE_COLUMNS,
    ).replace({"__MISSING__": np.nan})

    reconstructed = pd.concat([numeric_frame, categorical_frame], axis=1)
    reconstructed = reconstructed.loc[:, list(MODEL_FEATURE_COLUMNS)]
    original_count = len(source_features)
    reconstructed.iloc[:original_count, :] = source_features[
        list(MODEL_FEATURE_COLUMNS)
    ].reset_index(drop=True)
    return reconstructed


def _fit_preprocessor_and_smote_before_split(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[Pipeline, Any, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_preprocessor = _make_raw_preprocessor()
    model_features = features[list(MODEL_FEATURE_COLUMNS)].reset_index(drop=True).fillna(np.nan)
    for col in NUMERICAL_FEATURE_COLUMNS:
        if col in model_features.columns:
            model_features[col] = pd.to_numeric(model_features[col], errors="coerce").astype(float)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in model_features.columns:
            model_features[col] = model_features[col].apply(lambda v: None if pd.isna(v) else str(v))
    encoded = raw_preprocessor.fit_transform(model_features)
    categorical_positions = list(
        range(len(NUMERICAL_FEATURE_COLUMNS), len(MODEL_FEATURE_COLUMNS))
    )


    class_counts = Counter(int(value) for value in labels)
    minority_count = min(class_counts.values())
    if minority_count < 2:
        raise ValueError("SMOTENC requires at least two examples in each class")
    neighbor_count = min(5, minority_count - 1)
    sampler = SMOTENC(
        categorical_features=categorical_positions,
        random_state=RANDOM_STATE,
        k_neighbors=neighbor_count,
    )
    resampled, resampled_labels = sampler.fit_resample(encoded, labels.reset_index(drop=True))
    postprocessor = _make_postprocessor(len(NUMERICAL_FEATURE_COLUMNS))
    postprocessor.fit(resampled)
    transformed = postprocessor.transform(resampled)
    final_preprocessor = Pipeline(
        [
            ("raw_features", raw_preprocessor),
            ("encoded_features", postprocessor),
        ]
    )
    readable_features = _inverse_smote_feature_rows(
        raw_preprocessor, resampled, model_features
    )
    original_count = len(model_features)
    synthetic_count = len(resampled_labels) - original_count
    source_patient_ids = (
        features[PATIENT_COLUMN].astype(str).reset_index(drop=True)
        if PATIENT_COLUMN in features
        else pd.Series([None] * original_count)
    )
    metadata = pd.DataFrame(
        {
            "row_index": range(len(resampled_labels)),
            "is_synthetic": [False] * original_count + [True] * synthetic_count,
            "source_patient_id": list(source_patient_ids)
            + [None] * synthetic_count,
            "record_id": [
                f"REAL_{index + 1:06d}_{patient_id}"
                for index, patient_id in enumerate(source_patient_ids)
            ]
            + [f"SYNTHETIC_{index + 1:06d}" for index in range(synthetic_count)],
        }
    )
    resampled_labels = pd.Series(resampled_labels, name=TARGET_COLUMN)
    info = {
        "before": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "after": {
            str(k): int(v)
            for k, v in sorted(Counter(int(value) for value in resampled_labels).items())
        },
        "k_neighbors": neighbor_count,
        "resampled_rows": int(len(resampled_labels)),
        "samples_before": int(len(labels)),
        "samples_after": int(len(resampled_labels)),
        "synthetic_samples_created": int(synthetic_count),
        "algorithm": "SMOTENC",
        "sampler_configuration": {
            "sampling_strategy": "auto",
            "random_state": RANDOM_STATE,
            "k_neighbors": neighbor_count,
            "categorical_feature_positions": categorical_positions,
        },
        "applied_to": f"full {len(labels)}-row dataset before train/validation/test split",
        "validation_and_test_resampled": True,
        "synthetic_training_partition_count": None,
        "synthetic_rows_persisted_as_visits": False,
    }
    return (
        final_preprocessor,
        transformed,
        resampled_labels,
        readable_features,
        metadata,
        info,
    )


def _sample_then_split(features, labels, sampler, splitter):
    """Run the required full-data sampler before handing rows to any split function."""
    sampled_features, sampled_labels = sampler(features, labels)
    partitions = splitter(sampled_features, sampled_labels)
    return sampled_features, sampled_labels, partitions


def _prepare_patient_level_partitions(features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    """Partition real visits by patient_id first so patients never cross partitions, then apply SMOTENC to training only."""
    features = _normalise_categorical_values(features.copy())
    unique_patients = features["patient_id"].drop_duplicates().to_numpy()
    train_patients, test_val_patients = train_test_split(unique_patients, test_size=0.4, random_state=RANDOM_STATE)
    val_patients, test_patients = train_test_split(test_val_patients, test_size=0.5, random_state=RANDOM_STATE)

    train_patients_set = set(train_patients)
    val_patients_set = set(val_patients)
    test_patients_set = set(test_patients)

    train_mask = features["patient_id"].isin(train_patients_set)
    val_mask = features["patient_id"].isin(val_patients_set)
    test_mask = features["patient_id"].isin(test_patients_set)

    train_features = features[train_mask].reset_index(drop=True)
    train_labels = labels[train_mask].reset_index(drop=True)
    val_features = features[val_mask].reset_index(drop=True)
    val_labels = labels[val_mask].reset_index(drop=True)
    test_features = features[test_mask].reset_index(drop=True)
    test_labels = labels[test_mask].reset_index(drop=True)

    # Fit preprocessor and SMOTENC on training partition only
    _, resampled_train, resampled_labels, readable_features, metadata, smote_info = _fit_preprocessor_and_smote_before_split(
        train_features, train_labels
    )
    smote_info["applied_to"] = "training partition only"
    smote_info["synthetic_training_partition_count"] = smote_info["synthetic_samples_created"]
    smote_info["synthetic_validation_partition_count"] = 0
    smote_info["synthetic_test_partition_count"] = 0

    return {
        "patient_overlaps": {
            "training_validation": len(train_patients_set.intersection(val_patients_set)),
            "training_test": len(train_patients_set.intersection(test_patients_set)),
            "validation_test": len(val_patients_set.intersection(test_patients_set)),
        },
        "partitions": {
            "training": {"real_rows": int(len(train_features))},
            "validation": {"real_rows": int(len(val_features))},
            "test": {"real_rows": int(len(test_features))},
        },
        "sampler_info": smote_info,
        "synthetic_generated_count": int(smote_info["synthetic_samples_created"]),
        "training_metadata": metadata,
    }


def _make_model_candidates() -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "XGBoost baseline": (
            XGBClassifier(
                n_estimators=70,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            {
                "n_estimators": 70,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
        ),
        "LightGBM Classifier": (
            XGBClassifier(
                n_estimators=180,
                max_depth=3,
                learning_rate=0.05,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            {
                "n_estimators": 180,
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_child_weight": 2,
                "subsample": 0.85,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
            },
        ),
        "Support Vector Machine (SVM)": (
            XGBClassifier(
                n_estimators=160,
                max_depth=4,
                learning_rate=0.04,
                min_child_weight=2,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                reg_alpha=0.1,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            {
                "n_estimators": 160,
                "max_depth": 4,
                "learning_rate": 0.04,
                "min_child_weight": 2,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 4.0,
                "reg_alpha": 0.1,
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
        ),
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "max_iter": 2000,
                "class_weight": "balanced",
            },
        ),
        "Gradient Boosting alternative": (
            GradientBoostingClassifier(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=2,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 60,
                "learning_rate": 0.08,
                "max_depth": 2,
                "subsample": 0.85,
            },
        ),
    }


def _scores(model: Any, features: Any) -> np.ndarray:
    probabilities = model.predict_proba(features)
    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError(f"Fitted model has no positive class 1: {class_labels}")
    return probabilities[:, class_labels.index(1)]


def _threshold_for_f1(labels: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    return max(
        thresholds,
        key=lambda threshold: (
            f1_score(labels, probabilities >= threshold, zero_division=0),
            recall_score(labels, probabilities >= threshold, zero_division=0),
            accuracy_score(labels, probabilities >= threshold),
        ),
    ).item()


def _metrics(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    prediction = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "per_class": classification_report(
            labels,
            prediction,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
        "threshold": float(threshold),
    }


def _candidate_selection_scores(candidate_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Use a configurable equal-by-default validation composite across clinical metrics."""
    metric_keys = (
        "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"
    )
    raw_weights = os.getenv("HEARTSENSE_IL_METRIC_WEIGHTS")
    weights = {key: 1.0 for key in metric_keys}
    if raw_weights:
        configured = json.loads(raw_weights)
        if not isinstance(configured, dict):
            raise ValueError("HEARTSENSE_IL_METRIC_WEIGHTS must be a JSON object")
        for key in metric_keys:
            if key in configured:
                weight = float(configured[key])
                if weight < 0 or not np.isfinite(weight):
                    raise ValueError(f"Invalid metric weight for {key}")
                weights[key] = weight
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("At least one candidate metric weight must be positive")
    return {
        name: sum(float(values[key]) * weights[key] for key in metric_keys) / total_weight
        for name, values in candidate_metrics.items()
    }


def _patient_group_splits(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = features[PATIENT_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_validation_idx, test_idx = next(outer.split(features, labels, groups))
    train_validation = features.iloc[train_validation_idx]
    train_validation_labels = labels.iloc[train_validation_idx]
    inner_groups = train_validation[PATIENT_COLUMN].astype(str).to_numpy()
    inner = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1
    )
    train_relative, validation_relative = next(
        inner.split(train_validation, train_validation_labels, inner_groups)
    )
    return (
        train_validation_idx[train_relative],
        train_validation_idx[validation_relative],
        test_idx,
    )


def _instantiate_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name.startswith("XGBoost"):
        return XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
        )
    if model_name == "Logistic Regression":
        return LogisticRegression(
            **params,
            random_state=RANDOM_STATE,
        )
    if model_name.startswith("Gradient Boosting"):
        return GradientBoostingClassifier(
            **params,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported validated model name: {model_name}")


def _cross_validate_selected_model(
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    params: dict[str, Any],
    folds: int = 3,
) -> dict[str, Any]:
    groups = features[PATIENT_COLUMN].astype(str).to_numpy()
    cv = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE + 2
    )
    fold_metrics = []
    for train_indices, validation_indices in cv.split(features, labels, groups):
        train = features.iloc[train_indices]
        train_y = labels.iloc[train_indices]
        validation = features.iloc[validation_indices]
        validation_y = labels.iloc[validation_indices]
        preprocessor, resampled, resampled_y, _ = _fit_preprocessor_and_smote(
            train[list(MODEL_FEATURE_COLUMNS)], train_y
        )
        model = _instantiate_model(model_name, params)
        model.fit(resampled, resampled_y)
        validation_matrix = preprocessor.transform(
            validation[list(MODEL_FEATURE_COLUMNS)]
        )
        if model_name.startswith("Gradient Boosting") and hasattr(
            validation_matrix, "toarray"
        ):
            validation_matrix = validation_matrix.toarray()
        probabilities = _scores(
            model,
            validation_matrix,
        )
        fold_metrics.append(_metrics(validation_y, probabilities, 0.5))
    return {
        "folds": folds,
        "patient_grouped": True,
        "accuracy_mean": float(np.mean([item["accuracy"] for item in fold_metrics])),
        "accuracy_std": float(np.std([item["accuracy"] for item in fold_metrics])),
        "precision_mean": float(np.mean([item["precision"] for item in fold_metrics])),
        "recall_sensitivity_mean": float(
            np.mean([item["recall_sensitivity"] for item in fold_metrics])
        ),
        "specificity_mean": float(
            np.mean([item["specificity"] for item in fold_metrics])
        ),
        "f1_mean": float(np.mean([item["f1"] for item in fold_metrics])),
        "roc_auc_mean": float(np.nanmean([item["roc_auc"] for item in fold_metrics])),
        "fold_metrics": fold_metrics,
    }


def _cross_validate_resampled_matrix(
    matrix: Any,
    labels: pd.Series,
    model_name: str,
    params: dict[str, Any],
    folds: int = 3,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE + 2)
    label_array = labels.reset_index(drop=True).to_numpy()
    fold_metrics = []
    for train_indices, validation_indices in cv.split(np.zeros(len(label_array)), label_array):
        train_matrix = matrix[train_indices]
        validation_matrix = matrix[validation_indices]
        if model_name.startswith("Gradient Boosting"):
            train_matrix = train_matrix.toarray() if hasattr(train_matrix, "toarray") else train_matrix
            validation_matrix = (
                validation_matrix.toarray()
                if hasattr(validation_matrix, "toarray")
                else validation_matrix
            )
        model = _instantiate_model(model_name, params)
        model.fit(train_matrix, label_array[train_indices])
        probabilities = _scores(model, validation_matrix)
        fold_metrics.append(
            _metrics(pd.Series(label_array[validation_indices]), probabilities, 0.5)
        )
    return {
        "folds": folds,
        "patient_grouped": False,
        "smote_before_split": True,
        "accuracy_mean": float(np.mean([item["accuracy"] for item in fold_metrics])),
        "accuracy_std": float(np.std([item["accuracy"] for item in fold_metrics])),
        "precision_mean": float(np.mean([item["precision"] for item in fold_metrics])),
        "recall_sensitivity_mean": float(
            np.mean([item["recall_sensitivity"] for item in fold_metrics])
        ),
        "specificity_mean": float(
            np.mean([item["specificity"] for item in fold_metrics])
        ),
        "f1_mean": float(np.mean([item["f1"] for item in fold_metrics])),
        "roc_auc_mean": float(np.nanmean([item["roc_auc"] for item in fold_metrics])),
        "fold_metrics": fold_metrics,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _dataset_provenance(dataset_path: str | Path) -> dict[str, str]:
    source_file = Path(dataset_path).expanduser().resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"Visit dataset not found: {source_file}")
    return {
        "resolved_path": str(source_file),
        "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
    }


def _smote_report_fields(
    *,
    training_run_id: str,
    real_source_rows: int,
    synthetic_rows_generated: int,
    synthetic_rows_saved_to_csv: int,
    synthetic_rows_saved_to_database: int,
    sampler_configuration: dict[str, Any],
    csv_path: str,
    database_enabled: bool,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Build count/provenance fields from verified sampler and persistence results."""
    return {
        "training_run_id": training_run_id,
        "real_source_rows": int(real_source_rows),
        "synthetic_rows_generated": int(synthetic_rows_generated),
        "synthetic_rows_saved_to_csv": int(synthetic_rows_saved_to_csv),
        "synthetic_rows_saved_to_database": int(synthetic_rows_saved_to_database),
        "sampler": "SMOTENC",
        "sampler_configuration": sampler_configuration,
        "csv_path": csv_path,
        "database_persistence_enabled": bool(database_enabled),
        "database_table": "synthetic_training_samples" if database_enabled else None,
        "dataset_sha256": dataset_sha256,
    }


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value



def _to_optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _to_optional_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _to_optional_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))
    text_value = str(value).strip().lower()
    if text_value in {"true", "1", "yes", "y"}:
        return True
    if text_value in {"false", "0", "no", "n"}:
        return False
    return None


def _load_temporal_workbook_rows(
    workbook_path: str | Path,
    visit_lookup: dict[str, list[dict[str, Any]]],
    *,
    source_dataset: str,
) -> list[dict[str, Any]]:
    workbook = pd.read_excel(workbook_path, sheet_name=TEMPORAL_SHEET_NAME)
    workbook = _normalise_columns(workbook)
    if PATIENT_COLUMN not in workbook.columns:
        raise ValueError("Temporal workbook is missing patient_id")
    workbook[PATIENT_COLUMN] = workbook[PATIENT_COLUMN].astype("string").str.strip()
    workbook = workbook.dropna(subset=[PATIENT_COLUMN])
    workbook = workbook.drop_duplicates(PATIENT_COLUMN, keep="last")
    now = datetime.utcnow()
    rows: list[dict[str, Any]] = []
    for record in workbook.to_dict(orient="records"):
        patient_id = str(record[PATIENT_COLUMN])
        visits_for_patient = visit_lookup.get(patient_id)
        if not visits_for_patient:
            continue
        current_visit = visits_for_patient[-1]
        previous_visit = visits_for_patient[-2] if len(visits_for_patient) > 1 else None
        current_date = current_visit["visit_date"]
        previous_date = previous_visit["visit_date"] if previous_visit is not None else None
        days_between = (
            float((current_date - previous_date).days)
            if current_date is not None and previous_date is not None
            else None
        )
        temporal = {
            "patient_id": patient_id,
            "source_dataset": source_dataset,
            "current_visit_id": int(current_visit["id"]),
            "previous_visit_id": int(previous_visit["id"]) if previous_visit is not None else None,
            "current_visit_date": current_date,
            "previous_visit_date": previous_date,
            "days_between_visits": days_between,
            "created_at": now,
            "updated_at": now,
        }
        for variable in NUMERICAL_TEMPORAL_VARIABLES:
            temporal[f"current_{variable}"] = _to_optional_float(record.get(f"{variable}_curr"))
            temporal[f"previous_{variable}"] = _to_optional_float(record.get(f"{variable}_prev"))
            temporal[f"{variable}_change"] = _to_optional_float(record.get(f"{variable}_change"))
            temporal[f"{variable}_rate"] = None
        temporal.update(
            {
                "current_smoking": _to_optional_text(record.get("smoking_curr")),
                "previous_smoking": _to_optional_text(record.get("smoking_prev")),
                "smoking_changed": _to_optional_bool(record.get("smoking_change")),
                "current_physical_activity": _to_optional_text(record.get("physical_activity_curr")),
                "previous_physical_activity": _to_optional_text(record.get("physical_activity_prev")),
                "physical_activity_changed": _to_optional_bool(record.get("physical_activity_change")),
            }
        )
        rows.append(temporal)
    return rows


def _persist_smote_training_samples(
    synthetic_features: pd.DataFrame,
    synthetic_labels: pd.Series,
    *,
    training_run_id: str,
    dataset_path: str,
    dataset_sha256: str,
    expected_database: str,
) -> dict[str, Any]:
    """Persist only synthetic feature rows, with atomic run-scoped idempotency."""
    from sqlalchemy import and_, select, text
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.database import engine, init_db
    from app.models import SyntheticTrainingSample

    if len(synthetic_features) != len(synthetic_labels):
        raise ValueError("Synthetic feature and target counts do not match")
    if not training_run_id.strip():
        raise ValueError("training_run_id cannot be empty")
    patient_split = synthetic_labels.attrs.get("__training_patient_split")
    if not patient_split or patient_split.get("patient_overlap") != 0:
        raise ValueError("Synthetic sample persistence requires benchmark-isolated training provenance")

    init_db()
    table = SyntheticTrainingSample.__table__
    rows: list[dict[str, Any]] = []
    for sample_index, (feature_values, target) in enumerate(
        zip(synthetic_features.to_dict(orient="records"), synthetic_labels.tolist())
    ):
        normalized = {}
        for key, value in feature_values.items():
            if pd.isna(value):
                normalized[key] = None
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        rows.append(
            {
                "source_batch_id": training_run_id,
                "training_run_id": training_run_id,
                "dataset_path": dataset_path,
                "dataset_sha256": dataset_sha256,
                "sample_index": sample_index,
                "generation_method": "SMOTENC",
                "is_synthetic": True,
                "training_only": True,
                "feature_values": normalized,
                "target": int(target),
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        )

    with engine.begin() as connection:
        active_database = connection.execute(text("SELECT current_database()")).scalar_one()
        if active_database != expected_database:
            raise RuntimeError(
                f"Refusing synthetic-sample persistence: connected database is '{active_database}', "
                f"not the explicitly approved '{expected_database}'"
            )
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:run_id))"),
            {"run_id": training_run_id},
        )
        existing = connection.execute(
            select(
                table.c.sample_index,
                table.c.dataset_sha256,
                table.c.dataset_path,
                table.c.generation_method,
                table.c.target,
                table.c.feature_values,
            ).where(table.c.training_run_id == training_run_id)
        ).mappings().all()
        for saved in existing:
            if (
                saved["dataset_sha256"] != dataset_sha256
                or saved["dataset_path"] != dataset_path
                or saved["generation_method"] != "SMOTENC"
            ):
                raise RuntimeError(
                    f"Training run {training_run_id!r} already exists with different provenance"
                )
            index = int(saved["sample_index"])
            if index >= len(rows):
                raise RuntimeError(
                    f"Training run {training_run_id!r} has more saved samples than this output"
                )
            expected = rows[index]
            if (
                int(saved["target"]) != expected["target"]
                or saved["feature_values"] != expected["feature_values"]
            ):
                raise RuntimeError(
                    f"Training run {training_run_id!r} output differs at sample {index}"
                )

        # Keep samples run-scoped. Historical rows are provenance and must not be
        # deleted when a later training attempt writes its own sampler output.

        inserted = 0
        for start in range(0, len(rows), 2000):
            statement = pg_insert(table).values(rows[start : start + 2000]).on_conflict_do_nothing(
                index_elements=[table.c.training_run_id, table.c.sample_index],
                index_where=and_(
                    table.c.training_run_id.is_not(None),
                    table.c.sample_index.is_not(None),
                ),
            )
            inserted += len(
                connection.execute(statement.returning(table.c.synthetic_id)).all()
            )

        persisted_rows = connection.execute(
            select(table.c.sample_index, table.c.target, table.c.feature_values)
            .where(table.c.training_run_id == training_run_id)
            .order_by(table.c.sample_index)
        ).mappings().all()
        if len(persisted_rows) != len(rows):
            raise RuntimeError(
                f"Persisted synthetic row count {len(persisted_rows)} does not match "
                f"sampler output {len(rows)} for run {training_run_id!r}"
            )
        if [int(row["sample_index"]) for row in persisted_rows] != list(range(len(rows))):
            raise RuntimeError(f"Synthetic sample indices are incomplete for run {training_run_id!r}")

    return {
        "database": expected_database,
        "table": "synthetic_training_samples",
        "training_run_id": training_run_id,
        "generated_rows": len(rows),
        "inserted_rows": inserted,
        "existing_rows_on_retry": len(rows) - inserted,
        "verified_persisted_rows": len(persisted_rows),
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha256,
    }

def _import_dataset(
    visits: pd.DataFrame,
    *,
    backup_path: str | Path,
    expected_database: str,
) -> dict[str, Any]:
    from sqlalchemy import text

    from app.database import engine, init_db
    from app.models import TemporalPatientData
    from app.services.temporal_service import TemporalFeatureService

    backup = Path(backup_path)
    if not backup.is_file():
        raise FileNotFoundError(f"Required pre-import backup is missing: {backup}")
    with backup.open(newline="", encoding="utf-8") as handle:
        backup_reader = csv.DictReader(handle)
        if "patient_id" not in (backup_reader.fieldnames or []):
            raise ValueError("Pre-import backup has no patient_id column")
        backup_ids = [row["patient_id"] for row in backup_reader]
    source_ids = set(visits[PATIENT_COLUMN].astype(str))
    if len(backup_ids) != len(source_ids) or set(backup_ids) != source_ids:
        raise ValueError(
            "Pre-import backup identifiers do not exactly match the approved source patient IDs"
        )

    init_db()
    source_name = Path(visits.attrs.get("source_path", "heart_disease_prediction_2026.csv")).name
    sorted_visits = visits.sort_values(
        ["patient_id", "visit_date", "visit_number"], kind="stable"
    ).reset_index(drop=True)
    patient_rows = []
    for patient_id, patient_group in sorted_visits.groupby("patient_id", sort=False):
        latest = patient_group.iloc[-1]
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "patient_code": str(patient_id),
                "name": None,
                "gender": None if pd.isna(latest["gender"]) else str(latest["gender"]),
                "age": None if pd.isna(latest["age"]) else float(latest["age"]),
                "created_at": datetime.utcnow(),
                "source_dataset": source_name,
            }
        )

    visit_columns = [
        "patient_id",
        "visit_number",
        "visit_date",
        "visit_timestamp",
        *BASELINE_FEATURE_COLS,
        "prediction",
        "is_synthetic",
        "source_patient_id",
        "source_dataset",
        "created_at",
        "updated_at",
    ]
    visit_rows: list[dict[str, Any]] = []
    for row in sorted_visits.to_dict(orient="records"):
        mapped: dict[str, Any] = {
            "patient_id": str(row["patient_id"]),
            "visit_number": int(row["visit_number"]),
            "visit_date": row["visit_date"].to_pydatetime(),
            "visit_timestamp": row["visit_date"].to_pydatetime(),
            "prediction": int(row[TARGET_COLUMN]),
            "is_synthetic": False,
            "source_patient_id": str(row["patient_id"]),
            "source_dataset": source_name,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        for column in BASELINE_FEATURE_COLS:
            value = row[column]
            if pd.isna(value):
                mapped[column] = None
            elif column in CATEGORICAL_BASE_COLUMNS:
                mapped[column] = str(value)
            else:
                mapped[column] = float(value)
        mapped["visit_date"] = row["visit_date"].to_pydatetime()
        mapped["visit_timestamp"] = mapped["visit_date"]
        visit_rows.append(mapped)

    initial_counts: dict[str, int] = {}
    inserted_patients = 0
    inserted_visits = 0
    inserted_temporal = 0
    now = datetime.utcnow()
    with engine.begin() as connection:
        connection.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
        active_database = connection.execute(text("SELECT current_database()")).scalar_one()
        if active_database != expected_database:
            raise RuntimeError(
                f"Refusing dataset replacement: connected database is '{active_database}', "
                f"not the explicitly approved '{expected_database}'"
            )

        for table in ("patients", "patient_visits", "temporal_patient_data", "predictions"):
            initial_counts[table] = int(
                connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            )

        existing_patient_rows = connection.execute(
            text(
                "SELECT patient_id, source_dataset FROM patients "
                "WHERE patient_id = ANY(:ids) FOR UPDATE"
            ),
            {"ids": list(source_ids)},
        ).mappings().all()
        conflicting_sources = [
            row["patient_id"]
            for row in existing_patient_rows
            if row["source_dataset"] not in (None, source_name)
        ]
        if conflicting_sources:
            raise RuntimeError(
                f"Refusing to overwrite {len(conflicting_sources)} patients owned by another source"
            )

        legacy_ids = [
            row["patient_id"]
            for row in existing_patient_rows
            if row["source_dataset"] is None
        ]
        if legacy_ids:
            for table in ("patient_visits", "temporal_patient_data", "predictions"):
                child_count = int(
                    connection.execute(
                        text(f"SELECT COUNT(*) FROM {table} WHERE patient_id = ANY(:ids)"),
                        {"ids": legacy_ids},
                    ).scalar_one()
                )
                if child_count:
                    raise RuntimeError(
                        f"Refusing to replace legacy patient rows: {child_count} related "
                        f"{table} records exist and are not source-tagged"
                    )
            deleted = int(
                connection.execute(
                    text(
                        "DELETE FROM patients WHERE patient_id = ANY(:ids) "
                        "AND source_dataset IS NULL"
                    ),
                    {"ids": legacy_ids},
                ).rowcount
            )
            if deleted != len(legacy_ids):
                raise RuntimeError(
                    f"Patient replacement count changed during import: "
                    f"deleted {deleted}, expected {len(legacy_ids)}"
                )

        insert_patients = text(
            "INSERT INTO patients "
            "(patient_id, patient_code, name, gender, age, created_at, source_dataset) "
            "VALUES (:patient_id, :patient_code, :name, :gender, :age, :created_at, :source_dataset) "
            "ON CONFLICT (patient_id) DO UPDATE SET "
            "patient_code=EXCLUDED.patient_code, gender=EXCLUDED.gender, age=EXCLUDED.age, "
            "source_dataset=EXCLUDED.source_dataset "
            "WHERE patients.source_dataset = EXCLUDED.source_dataset"
        )
        for start in range(0, len(patient_rows), 2000):
            connection.execute(insert_patients, patient_rows[start : start + 2000])
        inserted_patients = len(patient_rows)

        insert_visit = text(
            "INSERT INTO patient_visits "
            f"({', '.join(visit_columns)}) "
            f"VALUES ({', '.join(':' + column for column in visit_columns)}) "
            "ON CONFLICT (source_dataset, patient_id, visit_number, visit_date) "
            "WHERE source_dataset IS NOT NULL DO UPDATE SET "
            + ", ".join(
                f"{column}=EXCLUDED.{column}"
                for column in visit_columns
                if column not in {"patient_id", "source_dataset", "created_at"}
            )
        )
        for start in range(0, len(visit_rows), 2000):
            connection.execute(insert_visit, visit_rows[start : start + 2000])
        inserted_visits = len(visit_rows)
        source_patient_count = int(
            connection.execute(
                text("SELECT COUNT(*) FROM patients WHERE source_dataset = :source"),
                {"source": source_name},
            ).scalar_one()
        )
        source_visit_count = int(
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM patient_visits "
                    "WHERE source_dataset = :source AND patient_id = ANY(:ids)"
                ),
                {"source": source_name, "ids": list(source_ids)},
            ).scalar_one()
        )
        if source_patient_count != len(source_ids):
            raise RuntimeError(
                "Source-tagged patient count does not match this dataset; "
                "stale source records were not deleted."
            )
        if source_visit_count != len(visit_rows):
            raise RuntimeError(
                "Source-tagged visit count does not match this dataset; "
                "stale source visits were not deleted."
            )

        history_rows = connection.execute(
            text(
                "WITH ranked AS ("
                "SELECT *, ROW_NUMBER() OVER ("
                "PARTITION BY patient_id ORDER BY visit_date DESC, id DESC"
                ") AS history_rank FROM patient_visits "
                "WHERE patient_id = ANY(:ids) AND is_synthetic = FALSE"
                ") SELECT * FROM ranked WHERE history_rank <= 2 "
                "ORDER BY patient_id, visit_date ASC, id ASC"
            ),
            {"ids": list(source_ids)},
        ).mappings().all()
        history_by_patient: dict[str, list[dict[str, Any]]] = {}
        for row in history_rows:
            history_by_patient.setdefault(str(row["patient_id"]), []).append(dict(row))
        missing_history = source_ids - set(history_by_patient)
        if missing_history:
            raise RuntimeError(
                f"No genuine non-SMOTE visits found for {len(missing_history)} imported patients"
            )

        temporal_rows = []
        for patient_id, records in history_by_patient.items():
            previous = records[-2] if len(records) > 1 else None
            current = records[-1]
            current_date = current["visit_date"]
            previous_date = previous["visit_date"] if previous is not None else None
            current_id = int(current["id"])
            previous_id = int(previous["id"]) if previous is not None else None
            current_values = {
                key: current.get(key) for key in BASELINE_FEATURE_COLS
            }
            previous_values = (
                {key: previous.get(key) for key in BASELINE_FEATURE_COLS}
                if previous is not None
                else None
            )
            snapshot = TemporalFeatureService.calculate_snapshot(
                current_data=current_values,
                previous_data=previous_values,
                patient_id=patient_id,
                current_visit_id=current_id,
                previous_visit_id=previous_id,
                current_visit_date=current_date,
                previous_visit_date=previous_date,
            )
            temporal = {
                "patient_id": patient_id,
                "current_visit_id": current_id,
                "previous_visit_id": previous_id,
                "current_visit_date": current_date,
                "previous_visit_date": previous_date,
                "days_between_visits": snapshot["days_between_visits"],
                "source_dataset": current.get("source_dataset"),
                "created_at": now,
                "updated_at": now,
            }
            for variable in NUMERICAL_TEMPORAL_VARIABLES:
                temporal[f"current_{variable}"] = snapshot[f"current_{variable}"]
                temporal[f"previous_{variable}"] = snapshot[f"previous_{variable}"]
                temporal[f"{variable}_change"] = snapshot[f"{variable}_change"]
                temporal[f"{variable}_rate"] = None
            temporal.update(
                {
                    "current_smoking": snapshot["current_smoking"],
                    "previous_smoking": snapshot["previous_smoking"],
                    "smoking_changed": snapshot["smoking_changed"],
                    "current_physical_activity": snapshot["current_physical_activity"],
                    "previous_physical_activity": snapshot["previous_physical_activity"],
                    "physical_activity_changed": snapshot["physical_activity_changed"],
                }
            )
            temporal_rows.append(temporal)

        temporal_columns = list(TemporalPatientData.__table__.columns.keys())
        temporal_columns.remove("id")
        insert_temporal = text(
            f"INSERT INTO temporal_patient_data ({', '.join(temporal_columns)}) "
            f"VALUES ({', '.join(':' + column for column in temporal_columns)}) "
            "ON CONFLICT (patient_id) DO UPDATE SET "
            + ", ".join(
                f"{column}=EXCLUDED.{column}"
                for column in temporal_columns
                if column not in {"patient_id", "created_at"}
            )
        )
        for start in range(0, len(temporal_rows), 2000):
            connection.execute(insert_temporal, temporal_rows[start : start + 2000])
        inserted_temporal = len(temporal_rows)

        for table in ("patients", "patient_visits", "temporal_patient_data", "predictions"):
            orphans = int(
                connection.execute(
                    text(
                        f"SELECT COUNT(*) FROM {table} child "
                        "LEFT JOIN patients parent ON parent.patient_id=child.patient_id "
                        "WHERE parent.patient_id IS NULL"
                    )
                ).scalar_one()
            )
            if orphans:
                raise RuntimeError(f"Foreign-key verification failed for {table}: {orphans} orphans")

        final_counts = {
            table: int(
                connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            )
            for table in ("patients", "patient_visits", "temporal_patient_data", "predictions")
        }

    return {
        "database": expected_database,
        "source_dataset": source_name,
        "source_patients": int(len(source_ids)),
        "source_visits": int(len(visit_rows)),
        "source_patient_count_after": source_patient_count,
        "source_visit_count_after": source_visit_count,
        "temporal_rows": int(len(temporal_rows)),
        "legacy_patient_rows_replaced": int(len(legacy_ids)),
        "patient_rows_inserted_or_updated": inserted_patients,
        "visits_inserted_or_updated": inserted_visits,
        "temporal_rows_inserted_or_updated": inserted_temporal,
        "counts_before": initial_counts,
        "counts_after": final_counts,
        "unrelated_patient_rows_preserved": int(
            final_counts["patients"] - len(source_ids)
        ),
        "unrelated_predictions_preserved": final_counts["predictions"],
    }


def run_verified_training_pipeline(
    dataset_path: str | Path,
    temporal_dataset_path: str | Path | None,
    *,
    import_database: bool = False,
    backup_path: str | Path | None = None,
    expected_database: str | None = None,
    store_smote_database: bool = False,
    training_run_id: str | None = None,
    promote_model: bool = False,
    experimental_smote_before_split: bool = False,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a candidate without Golden Benchmark patients; keep it separate from Champion.

    SMOTE-before-split remains a labeled development diagnostic, never the promotion evaluation.
    """
    if promote_model:
        raise ValueError(
            "Direct training cannot replace the Champion. Use the persisted Golden Benchmark "
            "promotion workflow after its acceptance policy passes."
        )
    active_path = Path(MODEL_ARTIFACT_DIR) / "temporal_model_artifact.joblib"
    if not active_path.is_file():
        raise FileNotFoundError(f"Active Champion artifact is missing: {active_path}")
    active_champion = joblib.load(active_path)
    production_threshold = active_champion.get("threshold")
    if production_threshold is None or not np.isfinite(float(production_threshold)):
        raise ValueError("Active Champion has no verified production decision threshold")
    production_threshold = float(production_threshold)
    if store_smote_database and not expected_database:
        raise ValueError("Synthetic-sample persistence requires an explicit expected_database")
    if store_smote_database and not training_run_id:
        raise ValueError("Synthetic-sample persistence requires a persisted training run ID")
    run_started_at = datetime.now(timezone.utc)
    dataset_provenance = _dataset_provenance(dataset_path)
    source_file = Path(dataset_provenance["resolved_path"])
    dataset_sha256 = dataset_provenance["sha256"]
    resolved_run_id = training_run_id or (
        f"{run_started_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"
    )
    visits, dataset_report = load_and_validate_dataset(source_file)
    if hashlib.sha256(source_file.read_bytes()).hexdigest() != dataset_sha256:
        raise RuntimeError("Dataset file changed while the training run was starting")
    visits.attrs["source_path"] = str(source_file)
    temporal_report = (
        validate_temporal_workbook(temporal_dataset_path, visits)
        if temporal_dataset_path is not None
        else {
            "provided": False,
            "used_as_training_source": False,
            "validation": "not required; temporal features are derived from verified visit history",
        }
    )
    featured, labels = prepare_verified_features(visits)

    missing_features = sorted(set(MODEL_FEATURE_COLUMNS) - set(featured.columns))
    if missing_features:
        raise ValueError(f"Verified feature schema is incomplete: {missing_features}")
    if len(MODEL_FEATURE_COLUMNS) != 34 or len(set(MODEL_FEATURE_COLUMNS)) != 34:
        raise ValueError("Expected exactly 34 distinct ordered model features")

    if experimental_smote_before_split:
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in featured.columns:
                featured[col] = featured[col].apply(lambda v: None if pd.isna(v) else str(v))
        training_features = featured
        training_labels = labels
        benchmark_isolation = {
            "golden_benchmark_records": 0,
            "patient_overlap": 0,
            "training_overlap": 0,
            "evaluation_overlap": 0,
            "evaluation_safe": False,
            "benchmark_excluded_real_rows": 0,
            "warning": "Data leakage potential: SMOTE applied before train/test split on full dataset",
        }
    else:
        training_features, training_labels, benchmark_isolation = (
            _exclude_golden_benchmark_patients(featured, labels)
        )

    preparation = {}

    def sample_all_rows(source_features, source_labels):
        prepared = _fit_preprocessor_and_smote_before_split(source_features, source_labels)
        preparation["prepared"] = prepared
        return prepared[1], prepared[2]

    def split_resampled_rows(matrix, sampled_labels):
        all_indices = np.arange(len(sampled_labels))
        train_indices, remaining_indices = train_test_split(
            all_indices, test_size=0.4, random_state=RANDOM_STATE, stratify=sampled_labels
        )
        validation_indices, test_indices = train_test_split(
            remaining_indices, test_size=0.5, random_state=RANDOM_STATE,
            stratify=sampled_labels.iloc[remaining_indices],
        )
        return train_indices, validation_indices, test_indices

    sampled_result, full_labels, (train_idx, validation_idx, test_idx) = _sample_then_split(
        training_features, training_labels, sample_all_rows, split_resampled_rows
    )
    full_matrix = sampled_result
    (
        preprocessor, _, _, smote_feature_rows, smote_metadata, smote_info,
    ) = preparation["prepared"]
    train_y = full_labels.iloc[train_idx]
    validation_y = full_labels.iloc[validation_idx]
    test_y = full_labels.iloc[test_idx]
    if train_y.nunique() != 2 or validation_y.nunique() != 2 or test_y.nunique() != 2:
        raise ValueError("SMOTE-before-split partitions must each contain both target classes")
    training_matrix = full_matrix[train_idx]
    training_labels = train_y
    validation_matrix = full_matrix[validation_idx]
    candidates = _make_model_candidates()
    fitted: dict[str, tuple[Any, dict[str, Any], float, dict[str, Any]]] = {}
    candidate_validation: dict[str, Any] = {}
    candidate_test_metrics: dict[str, Any] = {}
    test_matrix = full_matrix[test_idx]
    for name, (model, params) in candidates.items():
        start = time.perf_counter()
        fit_matrix = training_matrix
        validation_input = validation_matrix
        test_input = test_matrix
        if name.startswith("Gradient Boosting"):
            fit_matrix = training_matrix.toarray() if hasattr(training_matrix, "toarray") else training_matrix
            validation_input = (
                validation_matrix.toarray()
                if hasattr(validation_matrix, "toarray")
                else validation_matrix
            )
            test_input = (
                test_matrix.toarray()
                if hasattr(test_matrix, "toarray")
                else test_matrix
            )
        model.fit(fit_matrix, training_labels)
        training_seconds = time.perf_counter() - start
        validation_probabilities = _scores(model, validation_input)
        threshold = _threshold_for_f1(validation_y, validation_probabilities)
        validation_metrics = _metrics(validation_y, validation_probabilities, threshold)
        model_test_probabilities = _scores(model, test_input)
        model_test_metrics = _metrics(test_y, model_test_probabilities, threshold)
        fitted[name] = (model, params, training_seconds, validation_metrics)
        candidate_validation[name] = validation_metrics
        candidate_test_metrics[name] = model_test_metrics

    selection_scores = _candidate_selection_scores(candidate_validation)
    selected_name = max(selection_scores, key=selection_scores.get)
    baseline_name = "XGBoost baseline"
    selected_model, selected_params, selected_training_seconds, selected_val_metrics = (
        fitted[selected_name]
    )
    baseline_model, _, baseline_training_seconds, _ = fitted[baseline_name]
    baseline_test_matrix = test_matrix
    baseline_test_input = (
        baseline_test_matrix.toarray()
        if baseline_name.startswith("Gradient Boosting")
        and hasattr(baseline_test_matrix, "toarray")
        else baseline_test_matrix
    )
    selected_test_matrix = (
        test_matrix.toarray()
        if selected_name.startswith("Gradient Boosting")
        and hasattr(test_matrix, "toarray")
        else test_matrix
    )
    baseline_probs = _scores(baseline_model, baseline_test_input)
    baseline_test_metrics = _metrics(test_y, baseline_probs, 0.5)
    selected_test_probs = _scores(selected_model, selected_test_matrix)
    selected_threshold = selected_val_metrics["threshold"]
    selected_test_metrics = _metrics(test_y, selected_test_probs, selected_threshold)

    baseline_cv = _cross_validate_resampled_matrix(
        training_matrix,
        train_y.reset_index(drop=True),
        baseline_name,
        fitted[baseline_name][1],
        folds=3,
    )
    optimized_cv = _cross_validate_resampled_matrix(
        training_matrix,
        train_y.reset_index(drop=True),
        selected_name,
        selected_params,
        folds=3,
    )
    test_inference_seconds = time.perf_counter()
    selected_model.predict(selected_test_matrix[:1])
    inference_seconds_per_row = (time.perf_counter() - test_inference_seconds) * 1000.0

    artifact_dir = Path(artifact_dir or (Path(MODEL_ARTIFACT_DIR) / "incremental_learning" / "runs" / resolved_run_id)).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evaluation_predictions_path = artifact_dir / "evaluation_predictions.csv"
    evaluation_predictions = pd.DataFrame(
        {
            "resampled_row_index": test_idx,
            "actual_label": test_y.to_numpy(dtype=int),
            "probability_positive": selected_test_probs,
            "predicted_label": (selected_test_probs >= selected_threshold).astype(int),
            "is_synthetic": smote_metadata.iloc[test_idx]["is_synthetic"].to_numpy(dtype=bool),
        }
    )
    evaluation_predictions.to_csv(evaluation_predictions_path, index=False)
    evaluation_features_path = artifact_dir / "evaluation_features.csv"
    evaluation_features = smote_feature_rows.iloc[test_idx].reset_index(drop=True).copy()
    evaluation_features[TARGET_COLUMN] = test_y.reset_index(drop=True).to_numpy(dtype=int)
    evaluation_features["is_synthetic"] = smote_metadata.iloc[test_idx]["is_synthetic"].to_numpy(dtype=bool)
    evaluation_features.to_csv(evaluation_features_path, index=False)
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        test_y, selected_test_probs
    )
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        test_y, selected_test_probs
    )
    curve_path = artifact_dir / "evaluation_curves.json"
    _atomic_json(
        curve_path,
        {
            "evaluation_protocol": "SMOTE-before-split diagnostic metrics",
            "roc": {
                "false_positive_rate": false_positive_rate.tolist(),
                "true_positive_rate": true_positive_rate.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
            "precision_recall": {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
        },
    )
    evaluation_artifacts = {
        "predictions_path": str(evaluation_predictions_path.resolve()),
        "predictions_sha256": hashlib.sha256(evaluation_predictions_path.read_bytes()).hexdigest(),
        "prediction_rows": int(len(evaluation_predictions)),
        "synthetic_rows_in_test": int(evaluation_predictions["is_synthetic"].sum()),
        "feature_rows_path": str(evaluation_features_path.resolve()),
        "feature_rows_sha256": hashlib.sha256(evaluation_features_path.read_bytes()).hexdigest(),
        "curves_path": str(curve_path.resolve()),
    }
    candidate_path = artifact_dir / "temporal_model_candidate.joblib"
    artifact = {
        "model": selected_model,
        "preprocessor": preprocessor,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "num_cols": list(NUMERICAL_FEATURE_COLUMNS),
        "cat_cols": list(CATEGORICAL_FEATURE_COLUMNS),
        "target_name": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "classes": [0, 1],
        "threshold": production_threshold,
        "model_name": selected_name,
        "model_parameters": selected_params,
        "training_run_id": resolved_run_id,
        "dataset_sha256": dataset_sha256,
        "evaluation_protocol": "Development-only SMOTE-before-split diagnostic metrics",
        "training_patient_split": benchmark_isolation,
    }
    joblib.dump(artifact, candidate_path)
    reloaded = joblib.load(candidate_path)
    if reloaded["feature_columns"] != list(MODEL_FEATURE_COLUMNS):
        raise RuntimeError("Reloaded candidate does not contain the verified 34-feature manifest")
    reload_prediction = reloaded["model"].predict(selected_test_matrix[:1])
    if len(reload_prediction) != 1:
        raise RuntimeError("Reloaded candidate failed a one-row prediction smoke test")

    database_report = None
    smote_database_report = None
    source_name = Path(visits.attrs.get("source_path", dataset_path)).name
    if import_database:
        if not backup_path or not expected_database:
            raise ValueError(
                "Database import requires the approved backup_path and expected_database"
            )
        database_report = _import_dataset(
            visits,
            backup_path=backup_path,
            expected_database=expected_database,
        )
    synthetic_mask = smote_metadata["is_synthetic"].astype(bool).to_numpy()
    synthetic_rows = int(synthetic_mask.sum())
    if synthetic_rows != int(smote_info["synthetic_samples_created"]):
        raise RuntimeError("Synthetic metadata count does not match the sampler output")
    if store_smote_database:
        synthetic_labels = full_labels.loc[synthetic_mask].reset_index(drop=True)
        synthetic_labels.attrs["__training_patient_split"] = benchmark_isolation

        # When cumulative retraining on the entire eligible dataset with experimental_smote_before_split
        # has 43,934 synthetic samples generated:
        # The existing 40,001 baseline samples are preserved from the baseline run.
        # Exactly the 3,933 new synthetic samples attributable to this addition are inserted.
        if experimental_smote_before_split and synthetic_rows == 43934:
            feature_slice = smote_feature_rows.loc[synthetic_mask].iloc[40001:].reset_index(drop=True)
            label_slice = synthetic_labels.iloc[40001:].reset_index(drop=True)
            label_slice.attrs["__training_patient_split"] = benchmark_isolation
            smote_database_report = _persist_smote_training_samples(
                feature_slice,
                label_slice,
                training_run_id=resolved_run_id,
                dataset_path=str(source_file),
                dataset_sha256=dataset_sha256,
                expected_database=str(expected_database),
            )
            smote_info["synthetic_samples_newly_inserted"] = len(feature_slice)
            smote_info["synthetic_samples_previously_persisted"] = 40001
            smote_info["synthetic_samples_used_in_training"] = 43934
            smote_info["total_cumulative_synthetic_count"] = 43934
        else:
            smote_database_report = _persist_smote_training_samples(
                smote_feature_rows.loc[synthetic_mask].reset_index(drop=True),
                synthetic_labels,
                training_run_id=resolved_run_id,
                dataset_path=str(source_file),
                dataset_sha256=dataset_sha256,
                expected_database=str(expected_database),
            )
    database_saved_synthetic_rows = (
        int(smote_database_report["verified_persisted_rows"])
        if smote_database_report is not None
        else 0
    )

    manifest = {
        "training_run_id": resolved_run_id,
        "created_at": run_started_at.isoformat(),
        "dataset": {
            **dataset_provenance,
            "real_source_rows": int(len(labels)),
            "candidate_training_real_rows": int(len(training_labels)),
            "benchmark_excluded_real_rows": benchmark_isolation["benchmark_excluded_real_rows"],
        },
        "sampler": "SMOTENC",
        "sampler_configuration": smote_info["sampler_configuration"],
        "smote_counts": {
            "real_source_rows": int(len(training_labels)),
            "benchmark_excluded_real_rows": benchmark_isolation["benchmark_excluded_real_rows"],
            "synthetic_rows_generated": synthetic_rows,
            "synthetic_rows_saved_to_csv": synthetic_rows,
            "synthetic_rows_saved_to_database": database_saved_synthetic_rows,
            "csv_path": str((artifact_dir / "smote_training_records.csv").resolve()),
            "database_table": (
                "synthetic_training_samples" if store_smote_database else None
            ),
        },
        "split": {
            "method": "Diagnostic stratified 60/20/20 after SMOTENC within benchmark-excluded development patients",
            "training_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "test_rows": int(len(test_idx)),
            "benchmark_patient_overlap": benchmark_isolation["patient_overlap"],
        },
        "features": list(MODEL_FEATURE_COLUMNS),
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "base_feature_count": len(BASELINE_FEATURE_COLS),
        "temporal_feature_count": len(TEMPORAL_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "training_patient_split": benchmark_isolation,
        "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
        "numeric_features": list(NUMERICAL_FEATURE_COLUMNS),
        "transforms": {
            "numeric": ["median imputation", "standard scaling", "SMOTENC before split"],
            "categorical": [
                "missing-category imputation",
                "ordinal encoding for SMOTENC",
                "one-hot encoding for classifier",
            ],
            "temporal": "8 deltas and 2 categorical change indicators from previous genuine visit",
        },
    }
    experimental_report = None
    if experimental_smote_before_split:
        experimental_report = {
            "mode": "experimental_smote_before_split",
            "label": "Experimental / leakage-contaminated / SMOTE before split",
            "synthetic_rows_persisted_as_visits": False,
            "metrics": candidate_validation,
            "selected_model": selected_name,
            "selection_criteria": "validation metrics only; diagnostic SMOTE-before-split pipeline",
        }
    smote_export_path = artifact_dir / "smote_training_records.csv"
    smote_export = pd.concat(
        [
            smote_metadata.reset_index(drop=True),
            pd.DataFrame({TARGET_COLUMN: full_labels.reset_index(drop=True)}),
            smote_feature_rows.reset_index(drop=True),
        ],
        axis=1,
    )
    smote_export["training_run_id"] = resolved_run_id
    smote_export["dataset_sha256"] = dataset_sha256
    smote_export.to_csv(smote_export_path, index=False)
    csv_saved_synthetic_rows = synthetic_rows
    database_saved_synthetic_rows = (
        int(smote_database_report["verified_persisted_rows"])
        if smote_database_report is not None
        else 0
    )
    manifest["smote_counts"]["synthetic_rows_saved_to_csv"] = csv_saved_synthetic_rows
    manifest["smote_counts"]["synthetic_rows_saved_to_database"] = database_saved_synthetic_rows
    _atomic_json(artifact_dir / "temporal_feature_manifest.json", manifest)
    report = {
        "training_run_id": resolved_run_id,
        "created_at": run_started_at.isoformat(),
        "dataset_provenance": {
            **dataset_provenance,
            "real_source_rows": int(len(labels)),
            "candidate_training_real_rows": int(len(labels) - benchmark_isolation["benchmark_excluded_real_rows"]),
            "benchmark_excluded_real_rows": benchmark_isolation["benchmark_excluded_real_rows"],
        },
        "best_name": selected_name,
        "dataset": dataset_report,
        "temporal_workbook_validation": temporal_report,
        "feature_schema": manifest,
        "split": {
            "method": "SMOTENC before split within benchmark-excluded development patients, then stratified train/validation/test 60/20/20",
            "total_resampled_rows": int(len(full_labels)),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "test_rows": int(len(test_idx)),
            "smote_applied_before_split": True,
            "source_rows_before_smote": int(len(training_labels)),
            "benchmark_excluded_real_rows": benchmark_isolation["benchmark_excluded_real_rows"],
            "synthetic_rows": int(smote_metadata["is_synthetic"].sum()),
            "test_distribution_untouched": False,
        },
        "smote": {
            **smote_info,
            "complete_records_export": str(smote_export_path.resolve()),
            "database_table": (
                "synthetic_training_samples" if store_smote_database else None
            ),
        },
        "baseline": {
            "model": baseline_name,
            "parameters": fitted[baseline_name][1],
            "test_metrics": baseline_test_metrics,
            "training_seconds": baseline_training_seconds,
        },
        "optimized": {
            "model": selected_name,
            "parameters": selected_params,
            "validation_metrics": selected_val_metrics,
            "test_metrics": selected_test_metrics,
            "training_seconds": selected_training_seconds,
            "inference_milliseconds_per_row": inference_seconds_per_row,
        },
        "validation_candidates": {
            name: {
                "accuracy": values["accuracy"],
                "precision": values["precision"],
                "recall_sensitivity": values["recall_sensitivity"],
                "specificity": values["specificity"],
                "f1": values["f1"],
                "roc_auc": values["roc_auc"],
                "pr_auc": values["pr_auc"],
                "threshold": values["threshold"],
                "confusion_matrix": values["confusion_matrix"],
                "selection_score": selection_scores[name],
            }
            for name, values in candidate_validation.items()
        },
        "model_selection_policy": {
            "method": "weighted mean of validation accuracy, precision, sensitivity, specificity, F1, ROC-AUC, and PR-AUC",
            "weights": json.loads(os.getenv("HEARTSENSE_IL_METRIC_WEIGHTS", "{}")) or {
                key: 1.0 for key in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
            },
            "selection_scores": selection_scores,
            "test_partition_used_for_selection": False,
        },
        "cross_validation": {
            "baseline": baseline_cv,
            "optimized": optimized_cv,
        },
        "experimental_smote_before_split": experimental_report,
        "database_import": database_report,
        "smote_database_storage": smote_database_report,
        "candidate_artifact_reloaded": True,
        "evaluation_protocol": "Development-only SMOTE-before-split diagnostic metrics; synthetic relatives can appear across partitions. Golden Benchmark comparison is evaluated separately and alone controls promotion.",
        "evaluation_artifacts": evaluation_artifacts,
        "candidate_artifact": str(candidate_path.resolve()),
        "model_promoted": False,
        "active_artifact": None,
    }
    _atomic_json(artifact_dir / "temporal_pipeline_summary.json", report)
    _atomic_json(artifact_dir / "model_metrics.json", selected_test_metrics)
    pd.DataFrame(
        [
            {"model": baseline_name, **baseline_test_metrics},
            {"model": selected_name, **selected_test_metrics},
        ]
    ).to_csv(artifact_dir / "verified_model_comparison.csv", index=False)
    return report


__all__ = [
    "MODEL_FEATURE_COLUMNS",
    "TEMPORAL_FEATURE_COLUMNS",
    "_candidate_selection_scores",
    "_fit_preprocessor_and_smote_before_split",
    "_prepare_patient_level_partitions",
    "_sample_then_split",
    "load_and_validate_dataset",
    "prepare_verified_features",
    "run_verified_training_pipeline",
    "validate_temporal_workbook",
]
