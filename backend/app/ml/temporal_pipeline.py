"""
HeartSense – True Longitudinal Temporal Pipeline + Database + ML/DL

This module implements:
  1. Dataset inspection and verification (heart_disease_prediction_2026.csv as Single Source of Truth)
  2. PostgreSQL ingestion:
     - patients: unique patient master records
     - patient_visits: permanent history of all visits
     - temporal_patient_data: ONLY the latest two visits per patient for temporal changes and rates
  3. Zero-leakage temporal feature preparation:
     - 24 baseline clinical & lifestyle features
     - 10 previous values (8 numerical, 2 lifestyle)
     - 10 temporal changes (8 numerical changes, 2 lifestyle changes)
     - 8 temporal rates of change (change / days_between_visits)
  4. SMOTE balancing applied before train/test split
  5. Dual Pipeline Model Suite & Comparative Analysis:
     - Baseline Models (Before Temporal Data: 24 standard features)
     - Longitudinal Models (After Temporal Data: 24 baseline + 10 previous + 10 changes + 8 rates)
     Approved Models:
       - XGBoost (XGBClassifier)
       - Gradient Boosting (GradientBoostingClassifier)
       - AdaBoost (AdaBoostClassifier)
       - LightGBM (LGBMClassifier)
       - Compact MLP Neural Network (MLPClassifier)
  6. Multi-metric evaluation and feature importance extraction
  7. Prediction and manual visit ingestion endpoints
"""

from __future__ import annotations

import os
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import time
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
except Exception:
    pass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import text

# --- Scikit-Learn SimpleImputer Backward/Forward Compatibility Shim ---
if not hasattr(SimpleImputer, "_fill_dtype"):
    def _get_fill_dtype(self):
        if "_fill_dtype" in self.__dict__ and self.__dict__["_fill_dtype"] is not None:
            return self.__dict__["_fill_dtype"]
        if hasattr(self, "_fit_dtype") and self._fit_dtype is not None:
            return self._fit_dtype
        if hasattr(self, "statistics_") and self.statistics_ is not None and hasattr(self.statistics_, "dtype"):
            return self.statistics_.dtype
        return np.dtype("O") if getattr(self, "strategy", "") in ("most_frequent", "constant") else np.dtype("float64")

    def _set_fill_dtype(self, val):
        self.__dict__["_fill_dtype"] = val

    SimpleImputer._fill_dtype = property(_get_fill_dtype, _set_fill_dtype)

if not hasattr(SimpleImputer, "_fit_dtype"):
    def _get_fit_dtype(self):
        if "_fit_dtype" in self.__dict__ and self.__dict__["_fit_dtype"] is not None:
            return self.__dict__["_fit_dtype"]
        if hasattr(self, "_fill_dtype") and self._fill_dtype is not None:
            return self._fill_dtype
        if hasattr(self, "statistics_") and self.statistics_ is not None and hasattr(self.statistics_, "dtype"):
            return self.statistics_.dtype
        return np.dtype("O") if getattr(self, "strategy", "") in ("most_frequent", "constant") else np.dtype("float64")

    def _set_fit_dtype(self, val):
        self.__dict__["_fit_dtype"] = val

    SimpleImputer._fit_dtype = property(_get_fit_dtype, _set_fit_dtype)


def sanitize_preprocessor(preprocessor: Any) -> Any:
    """
    Ensure all SimpleImputer and transformer steps have valid _fill_dtype and _fit_dtype
    attributes regardless of how they were serialized or deserialized.
    """
    if preprocessor is None:
        return preprocessor

    def _fix_step(est):
        if est is None:
            return
        if hasattr(est, "steps"):
            for _, step in est.steps:
                _fix_step(step)
        if hasattr(est, "transformers_"):
            for item in est.transformers_:
                if len(item) >= 2 and item[1] not in ("drop", "passthrough", None):
                    _fix_step(item[1])
        if hasattr(est, "transformers"):
            for item in est.transformers:
                if len(item) >= 2 and item[1] not in ("drop", "passthrough", None):
                    _fix_step(item[1])

        cls_name = est.__class__.__name__
        if "SimpleImputer" in cls_name or hasattr(est, "statistics_"):
            stat = getattr(est, "statistics_", None)
            stat_dtype = stat.dtype if (stat is not None and hasattr(stat, "dtype")) else None
            strategy = getattr(est, "strategy", "mean")
            default_dtype = np.dtype("O") if strategy in ("most_frequent", "constant") else np.dtype("float64")
            target_dtype = stat_dtype if stat_dtype is not None else default_dtype

            try:
                if not hasattr(est, "_fill_dtype") or getattr(est, "_fill_dtype") is None:
                    est.__dict__["_fill_dtype"] = target_dtype
                if not hasattr(est, "_fit_dtype") or getattr(est, "_fit_dtype") is None:
                    est.__dict__["_fit_dtype"] = target_dtype
            except Exception:
                pass

    try:
        _fix_step(preprocessor)
    except Exception:
        pass
    return preprocessor

# Import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from app.config import (
    DATASET_PATH,
    MODEL_ARTIFACT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    BASELINE_FEATURE_COLS,
    CRITICAL_NUMERICAL_FEATURES,
    CRITICAL_LIFESTYLE_FEATURES,
    ALL_10_TEMPORAL_VARIABLES,
    PREVIOUS_VALUE_COLS,
    CHANGE_COLS,
    RATE_COLS,
    ALL_MODEL_FEATURE_COLS,
)
from app.database import engine, init_db, reset_db_tables, get_session
from app.models import Patient, PatientVisit, TemporalPatientData, Prediction
from app.services.temporal_service import (
    TemporalFeatureService,
    NUMERICAL_VARIABLES,
    CATEGORICAL_VARIABLES,
    VARIABLE_METADATA,
    normalize_smoking_value,
    _safe_float,
    _safe_str,
)
from app.services.patient_service import (
    create_patient,
    create_visit,
    create_prediction,
    get_patient_by_id,
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def load_dataset(path: str | None = None) -> pd.DataFrame:
    path = path or DATASET_PATH
    if not os.path.exists(path):
        candidates = [
            path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "heart_disease_prediction_2026.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "backend", "data", "heart_disease_prediction_2026.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "heart_disease_prediction_2026.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    ext = Path(path).suffix.lower()
    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return normalize_columns(df)


def inspect_and_clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inspect dataset integrity and clean values according to the new dataset schema.
    """
    original_records = len(df)
    original_columns = len(df.columns)

    df = df.replace({"?": np.nan, "": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan}).dropna(how="all")
    empty_rows_removed = original_records - len(df)

    if "patient_id" not in df.columns:
        df["patient_id"] = [f"P{i+1:06d}" for i in range(len(df))]
    else:
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df = df[df["patient_id"].ne("") & df["patient_id"].ne("nan")].copy()

    # Determine date column
    if "visit_date" in df.columns:
        dt = pd.to_datetime(df["visit_date"], errors="coerce")
        df["visit_date"] = dt.fillna(datetime(2025, 1, 1))
    else:
        base_time = datetime(2025, 1, 1)
        df["visit_date"] = [base_time + timedelta(days=i % 365, hours=(i // 365) % 24) for i in range(len(df))]

    # Visit number
    if "visit_number" in df.columns:
        df["visit_number"] = pd.to_numeric(df["visit_number"], errors="coerce").fillna(1).astype(int)
    else:
        df["visit_number"] = 1

    # Target column
    target_col = "prediction" if "prediction" in df.columns else ("target" if "target" in df.columns else None)
    if target_col and target_col in df.columns:
        df["prediction"] = (pd.to_numeric(df[target_col], errors="coerce").fillna(0) > 0).astype(int)
    else:
        df["prediction"] = 0

    vc = df["patient_id"].value_counts()
    unique_patients = int(vc.size)
    multi_visit_patients = int((vc > 1).sum())
    single_visit_patients = int((vc == 1).sum())
    avg_visits = round(float(len(df)) / float(max(unique_patients, 1)), 2)

    report = {
        "Original records": original_records,
        "Original columns": original_columns,
        "Empty rows removed": empty_rows_removed,
        "Valid records after cleaning": len(df),
        "Unique patients": unique_patients,
        "Total visits": len(df),
        "Patients with multiple visits": multi_visit_patients,
        "Patients with one visit": single_visit_patients,
        "Average visits per patient": avg_visits,
        "Earliest visit": str(df["visit_date"].min()),
        "Latest visit": str(df["visit_date"].max()),
    }

    print("\nDATASET INTEGRITY & PATIENT INSPECTION REPORT")
    print("---------------------------------------------")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return df, report


def ingest_dataset_to_postgresql(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ingests the dataset into PostgreSQL with the exact three-table architecture:
      1. patients (unique patient records)
      2. patient_visits (permanent history of all visits)
      3. temporal_patient_data (ONLY the latest two visits per patient)
    """
    print("[Database] Resetting tables for fresh dataset ingestion...")
    reset_db_tables()

    # Sort dataset chronologically per patient
    df = df.sort_values(by=["patient_id", "visit_date", "visit_number"]).reset_index(drop=True)
    now = datetime.utcnow()

    # 1. Populate patients table
    print(f"[Database] Extracting unique patients from {len(df)} visits...")
    patient_grp = df.drop_duplicates(subset=["patient_id"], keep="first")
    patients_data = []
    for _, row in patient_grp.iterrows():
        pid = str(row["patient_id"])
        gender = str(row.get("gender", "")) if pd.notna(row.get("gender")) else None
        age = float(row.get("age")) if pd.notna(row.get("age")) else None
        patients_data.append({
            "patient_id": pid,
            "name": f"Patient {pid}",
            "gender": gender,
            "age": age,
            "created_at": now,
        })

    with engine.begin() as con:
        print(f"[Database] Inserting {len(patients_data)} unique patients into 'patients'...")
        batch_size = 10000
        for i in range(0, len(patients_data), batch_size):
            batch = patients_data[i:i + batch_size]
            con.execute(text("""
                INSERT INTO patients (patient_id, name, gender, age, created_at)
                VALUES (:patient_id, :name, :gender, :age, :created_at)
                ON CONFLICT (patient_id) DO NOTHING
            """), batch)

    # 2. Populate patient_visits table
    print(f"[Database] Inserting {len(df)} visits into 'patient_visits'...")
    visit_cols = BASELINE_FEATURE_COLS + ["patient_id", "visit_number", "visit_date", "prediction"]
    vis_df = df[[c for c in visit_cols if c in df.columns]].copy()
    vis_df["visit_timestamp"] = vis_df["visit_date"]
    vis_df["created_at"] = now
    vis_df["updated_at"] = now

    # Ensure smoking_status exists
    if "smoking" in vis_df.columns and "smoking_status" not in vis_df.columns:
        vis_df["smoking_status"] = vis_df["smoking"]

    records = vis_df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    with engine.begin() as con:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            con.execute(text("""
                INSERT INTO patient_visits (
                    patient_id, visit_number, visit_date, visit_timestamp,
                    age, gender, bmi, chest_pain_type, systolic_bp, diastolic_bp,
                    resting_heart_rate, max_heart_rate, cholesterol, hdl, ldl,
                    fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                    oldpeak, st_slope, num_major_vessels, thalassemia, smoking,
                    smoking_status, family_history, physical_activity, stress_level,
                    prediction, created_at, updated_at
                )
                VALUES (
                    :patient_id, :visit_number, :visit_date, :visit_timestamp,
                    :age, :gender, :bmi, :chest_pain_type, :systolic_bp, :diastolic_bp,
                    :resting_heart_rate, :max_heart_rate, :cholesterol, :hdl, :ldl,
                    :fasting_blood_sugar, :hba1c, :diabetes, :resting_ecg, :exercise_angina,
                    :oldpeak, :st_slope, :num_major_vessels, :thalassemia, :smoking,
                    :smoking_status, :family_history, :physical_activity, :stress_level,
                    :prediction, :created_at, :updated_at
                )
            """), batch)

    # 3. Populate temporal_patient_data (ONLY the latest two visits per patient)
    print("[Database] Populating 'temporal_patient_data' with ONLY the latest two visits per patient...")
    with engine.begin() as con:
        db_visits_df = pd.read_sql(
            "SELECT * FROM patient_visits ORDER BY patient_id, visit_date ASC, id ASC",
            con
        )

    db_records = db_visits_df.to_dict(orient="records")
    patient_visits_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in db_records:
        pid = str(r["patient_id"])
        if pid not in patient_visits_map:
            patient_visits_map[pid] = []
        patient_visits_map[pid].append(r)

    # Calculate snapshots for latest 2 visits per patient
    temporal_rows = []
    for pid, v_list in patient_visits_map.items():
        latest_v = v_list[-1]
        prev_v = v_list[-2] if len(v_list) > 1 else None

        snapshot = TemporalFeatureService.calculate_snapshot(
            current_data=latest_v,
            previous_data=prev_v,
            patient_id=str(pid),
            current_visit_id=latest_v["id"],
            previous_visit_id=prev_v["id"] if prev_v else None,
            current_visit_date=latest_v["visit_date"],
            previous_visit_date=prev_v["visit_date"] if prev_v else None,
        )

        row = {
            "patient_id": str(pid),
            "current_visit_id": snapshot["current_visit_id"],
            "previous_visit_id": snapshot["previous_visit_id"],
            "current_visit_date": snapshot["current_visit_date"],
            "previous_visit_date": snapshot["previous_visit_date"],
            "days_between_visits": snapshot["days_between_visits"],
            "created_at": now,
            "updated_at": now,
        }
        for var in NUMERICAL_VARIABLES:
            row[f"current_{var}"] = snapshot.get(f"current_{var}")
            row[f"previous_{var}"] = snapshot.get(f"previous_{var}")
            row[f"{var}_change"] = snapshot.get(f"{var}_change")
            row[f"{var}_rate"] = snapshot.get(f"{var}_rate")

        row["current_smoking"] = snapshot.get("current_smoking")
        row["previous_smoking"] = snapshot.get("previous_smoking")
        row["smoking_changed"] = snapshot.get("smoking_changed")

        row["current_physical_activity"] = snapshot.get("current_physical_activity")
        row["previous_physical_activity"] = snapshot.get("previous_physical_activity")
        row["physical_activity_changed"] = snapshot.get("physical_activity_changed")

        temporal_rows.append(row)

    with engine.begin() as con:
        for i in range(0, len(temporal_rows), batch_size):
            batch = temporal_rows[i:i + batch_size]
            con.execute(text("""
                INSERT INTO temporal_patient_data (
                    patient_id, current_visit_id, previous_visit_id,
                    current_visit_date, previous_visit_date, days_between_visits,
                    current_systolic_bp, previous_systolic_bp, systolic_bp_change, systolic_bp_rate,
                    current_diastolic_bp, previous_diastolic_bp, diastolic_bp_change, diastolic_bp_rate,
                    current_cholesterol, previous_cholesterol, cholesterol_change, cholesterol_rate,
                    current_ldl, previous_ldl, ldl_change, ldl_rate,
                    current_hdl, previous_hdl, hdl_change, hdl_rate,
                    current_bmi, previous_bmi, bmi_change, bmi_rate,
                    current_hba1c, previous_hba1c, hba1c_change, hba1c_rate,
                    current_resting_heart_rate, previous_resting_heart_rate, resting_heart_rate_change, resting_heart_rate_rate,
                    current_smoking, previous_smoking, smoking_changed,
                    current_physical_activity, previous_physical_activity, physical_activity_changed,
                    created_at, updated_at
                )
                VALUES (
                    :patient_id, :current_visit_id, :previous_visit_id,
                    :current_visit_date, :previous_visit_date, :days_between_visits,
                    :current_systolic_bp, :previous_systolic_bp, :systolic_bp_change, :systolic_bp_rate,
                    :current_diastolic_bp, :previous_diastolic_bp, :diastolic_bp_change, :diastolic_bp_rate,
                    :current_cholesterol, :previous_cholesterol, :cholesterol_change, :cholesterol_rate,
                    :current_ldl, :previous_ldl, :ldl_change, :ldl_rate,
                    :current_hdl, :previous_hdl, :hdl_change, :hdl_rate,
                    :current_bmi, :previous_bmi, :bmi_change, :bmi_rate,
                    :current_hba1c, :previous_hba1c, :hba1c_change, :hba1c_rate,
                    :current_resting_heart_rate, :previous_resting_heart_rate, :resting_heart_rate_change, :resting_heart_rate_rate,
                    :current_smoking, :previous_smoking, :smoking_changed,
                    :current_physical_activity, :previous_physical_activity, :physical_activity_changed,
                    :created_at, :updated_at
                )
                ON CONFLICT (patient_id) DO UPDATE SET
                    current_visit_id = EXCLUDED.current_visit_id,
                    previous_visit_id = EXCLUDED.previous_visit_id,
                    current_visit_date = EXCLUDED.current_visit_date,
                    previous_visit_date = EXCLUDED.previous_visit_date,
                    days_between_visits = EXCLUDED.days_between_visits,
                    current_systolic_bp = EXCLUDED.current_systolic_bp,
                    previous_systolic_bp = EXCLUDED.previous_systolic_bp,
                    systolic_bp_change = EXCLUDED.systolic_bp_change,
                    systolic_bp_rate = EXCLUDED.systolic_bp_rate,
                    current_diastolic_bp = EXCLUDED.current_diastolic_bp,
                    previous_diastolic_bp = EXCLUDED.previous_diastolic_bp,
                    diastolic_bp_change = EXCLUDED.diastolic_bp_change,
                    diastolic_bp_rate = EXCLUDED.diastolic_bp_rate,
                    current_cholesterol = EXCLUDED.current_cholesterol,
                    previous_cholesterol = EXCLUDED.previous_cholesterol,
                    cholesterol_change = EXCLUDED.cholesterol_change,
                    cholesterol_rate = EXCLUDED.cholesterol_rate,
                    current_ldl = EXCLUDED.current_ldl,
                    previous_ldl = EXCLUDED.previous_ldl,
                    ldl_change = EXCLUDED.ldl_change,
                    ldl_rate = EXCLUDED.ldl_rate,
                    current_hdl = EXCLUDED.current_hdl,
                    previous_hdl = EXCLUDED.previous_hdl,
                    hdl_change = EXCLUDED.hdl_change,
                    hdl_rate = EXCLUDED.hdl_rate,
                    current_bmi = EXCLUDED.current_bmi,
                    previous_bmi = EXCLUDED.previous_bmi,
                    bmi_change = EXCLUDED.bmi_change,
                    bmi_rate = EXCLUDED.bmi_rate,
                    current_hba1c = EXCLUDED.current_hba1c,
                    previous_hba1c = EXCLUDED.previous_hba1c,
                    hba1c_change = EXCLUDED.hba1c_change,
                    hba1c_rate = EXCLUDED.hba1c_rate,
                    current_resting_heart_rate = EXCLUDED.current_resting_heart_rate,
                    previous_resting_heart_rate = EXCLUDED.previous_resting_heart_rate,
                    resting_heart_rate_change = EXCLUDED.resting_heart_rate_change,
                    resting_heart_rate_rate = EXCLUDED.resting_heart_rate_rate,
                    current_smoking = EXCLUDED.current_smoking,
                    previous_smoking = EXCLUDED.previous_smoking,
                    smoking_changed = EXCLUDED.smoking_changed,
                    current_physical_activity = EXCLUDED.current_physical_activity,
                    previous_physical_activity = EXCLUDED.previous_physical_activity,
                    physical_activity_changed = EXCLUDED.physical_activity_changed,
                    updated_at = EXCLUDED.updated_at
            """), batch)

    print(f"[Database] Successfully populated PostgreSQL database:")
    print(f"  • patients: {len(patients_data)} rows")
    print(f"  • patient_visits: {len(records)} rows")
    print(f"  • temporal_patient_data: {len(temporal_rows)} rows")

    return {
        "patients_count": len(patients_data),
        "visits_count": len(records),
        "temporal_records_count": len(temporal_rows),
    }


def prepare_ml_features_from_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute previous visit values, temporal changes, and rates for EVERY visit in the dataset
    using strictly chronologically preceding visits of the SAME patient (zero data leakage).
    """
    df_sorted = df.sort_values(by=["patient_id", "visit_date", "visit_number"]).reset_index(drop=True)

    # Date differences
    df_sorted["prev_visit_date"] = df_sorted.groupby("patient_id")["visit_date"].shift(1)
    diff_days = (df_sorted["visit_date"] - df_sorted["prev_visit_date"]).dt.total_seconds() / 86400.0
    df_sorted["days_between_visits"] = np.where(diff_days > 0, diff_days, np.nan)

    # 8 Numerical features
    for var in CRITICAL_NUMERICAL_FEATURES:
        cur_s = pd.to_numeric(df_sorted[var], errors="coerce")
        prev_s = df_sorted.groupby("patient_id")[var].shift(1)
        prev_s = pd.to_numeric(prev_s, errors="coerce")

        df_sorted[f"prev_{var}"] = prev_s
        df_sorted[f"{var}_change"] = cur_s - prev_s
        df_sorted[f"delta_{var}"] = cur_s - prev_s
        df_sorted[f"{var}_rate"] = np.where(
            df_sorted["days_between_visits"] > 0,
            (cur_s - prev_s) / df_sorted["days_between_visits"],
            np.nan
        )

    # 2 Lifestyle features
    for var in CRITICAL_LIFESTYLE_FEATURES:
        cur_s = df_sorted[var].astype(str)
        prev_s = df_sorted.groupby("patient_id")[var].shift(1)
        df_sorted[f"prev_{var}"] = prev_s
        df_sorted[f"{var}_changed"] = np.where(
            prev_s.notna(),
            cur_s != prev_s.astype(str),
            np.nan
        )

    y = df_sorted["prediction"].astype(int)
    return df_sorted, y


def build_feature_preprocessor(feature_cols: List[str], df_sample: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build preprocessor for the specified list of feature columns.
    """
    num_cols = []
    cat_cols = []

    for col in feature_cols:
        if col in df_sample.columns:
            s = pd.to_numeric(df_sample[col], errors="coerce")
            if s.notna().mean() > 0.4:
                num_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            if any(term in col for term in ["_change", "_rate", "prev_", "delta_"]):
                num_cols.append(col)
            else:
                cat_cols.append(col)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, num_cols, cat_cols


def get_approved_models() -> Dict[str, Any]:
    """Return the approved model suite used for holdout and cross-fold evaluation."""
    models: Dict[str, Any] = {}

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=70,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )

    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        subsample=0.8,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )

    models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=70,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )

    models["Compact MLP Neural Network"] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.01,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        max_iter=100,
        random_state=RANDOM_STATE,
    )
    return models


def calculate_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray | None = None,
) -> Dict[str, Any]:
    """Calculate the full binary evaluation metric set used across the project."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    score = y_score if y_score is not None else y_pred

    try:
        auc = float(roc_auc_score(y_true, score))
    except Exception:
        auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, score))
    except Exception:
        pr_auc = 0.5

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": cm.tolist()},
    }


def _predict_scores(model: Any, X_data: np.ndarray) -> np.ndarray:
    try:
        return model.predict_proba(X_data)[:, 1]
    except Exception:
        try:
            return model.decision_function(X_data)
        except Exception:
            return model.predict(X_data)


def cross_validate_model_suite(
    X_resampled: np.ndarray,
    y_resampled: pd.Series | np.ndarray,
    suite_label: str,
    folds: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run stratified 5-fold validation and return summary rows plus per-fold metrics."""
    models = get_approved_models()
    y_array = np.asarray(y_resampled)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]
    fold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    print(f"\n[Validation] Running {folds}-fold validation for {suite_label}...")
    for name, model in models.items():
        per_model_metrics: Dict[str, List[float]] = {m: [] for m in metric_names}
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_resampled, y_array), start=1):
            fold_model = clone(model)
            fold_model.fit(X_resampled[train_idx], y_array[train_idx])
            pred = fold_model.predict(X_resampled[val_idx])
            score = _predict_scores(fold_model, X_resampled[val_idx])
            metrics = calculate_classification_metrics(y_array[val_idx], pred, score)

            for metric_name in metric_names:
                per_model_metrics[metric_name].append(float(metrics[metric_name]))

            fold_rows.append({
                "suite": suite_label,
                "model_name": name,
                "fold": fold_idx,
                **{m: metrics[m] for m in metric_names},
                "confusion_matrix": metrics["confusion_matrix"],
            })

        summary = {
            "suite": suite_label,
            "model_name": name,
            "model_type": "Deep Learning" if "MLP" in name else "ML Ensemble",
            "folds": folds,
        }
        for metric_name in metric_names:
            values = per_model_metrics[metric_name]
            summary[f"{metric_name}_mean"] = round(float(np.mean(values)), 4)
            summary[f"{metric_name}_std"] = round(float(np.std(values)), 4)
            summary[f"{metric_name}_min"] = round(float(np.min(values)), 4)
            summary[f"{metric_name}_max"] = round(float(np.max(values)), 4)
        summary["meets_95_accuracy_target"] = bool(summary["accuracy_mean"] >= 0.95)
        summary_rows.append(summary)
        print(
            f"  * [{suite_label}] {name:<26}: "
            f"CV Acc={summary['accuracy_mean']*100:.2f}% +/- {summary['accuracy_std']*100:.2f}% | "
            f"F1={summary['f1_mean']*100:.2f}% | ROC-AUC={summary['roc_auc_mean']:.4f}"
        )

    summary_rows.sort(key=lambda r: (r["roc_auc_mean"], r["f1_mean"], r["accuracy_mean"]), reverse=True)
    return summary_rows, fold_rows


def train_model_suite(
    X_train_res: np.ndarray,
    y_train_res: pd.Series | np.ndarray,
    X_test_proc: np.ndarray,
    y_test: pd.Series | np.ndarray,
    proc_feature_names: List[str],
    suite_label: str = "Enhanced"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Any, Dict[str, float]]:
    """
    Train and evaluate the 5 approved models on the prepared train/test arrays.
    """
    models = get_approved_models()

    comparison_rows = []
    trained_models = {}
    feature_importances: Dict[str, float] = {}

    print(f"\n[Training] Running {suite_label} Model Suite ({len(models)} approved models)...")
    for name, model in models.items():
        t_start = time.time()
        model.fit(X_train_res, y_train_res)
        train_time = round(time.time() - t_start, 2)
        trained_models[name] = model

        # Evaluate on test set
        pred = model.predict(X_test_proc)
        proba = _predict_scores(model, X_test_proc)
        metrics = calculate_classification_metrics(y_test, pred, proba)

        row = {
            "model_name": name,
            "model_type": "Deep Learning" if "MLP" in name else "ML Ensemble",
            **metrics,
            "training_time_seconds": train_time,
        }
        comparison_rows.append(row)
        print(f"  * [{suite_label}] {name:<26}: Acc={metrics['accuracy']*100:.2f}% | F1={metrics['f1']*100:.2f}% | ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} ({train_time}s)")

        # Feature importances from tree-based models
        if name in {"XGBoost", "Gradient Boosting", "LightGBM"} and not feature_importances:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                if len(imp) == len(proc_feature_names):
                    for fn_name, val in zip(proc_feature_names, imp):
                        feature_importances[fn_name] = round(float(val), 5)

    comp_df = pd.DataFrame(comparison_rows).sort_values(["roc_auc", "f1", "accuracy"], ascending=False)
    best_row = comp_df.iloc[0].to_dict()
    best_model = trained_models[best_row["model_name"]]

    return comparison_rows, best_row, best_model, feature_importances


def run_temporal_training_pipeline(dataset_path: str | None = None) -> Dict[str, Any]:
    """
    Run complete HeartSense ML/DL pipeline:
      - Clean & ingest new dataset into PostgreSQL
      - Prepare temporal features (zero leakage)
      - Apply SMOTE before train/test split
      - Evaluates models Before Temporal (Baseline 24 features) and After Temporal (Baseline + Previous + Changes + Rates)
      - Generates comparative analytics & persists all artifacts.
    """
    dataset_path = dataset_path or DATASET_PATH

    print("=" * 75)
    print("HEARTSENSE TEMPORAL ML/DL PIPELINE: BEFORE vs AFTER TEMPORAL DATA")
    print("=" * 75)

    # 1. Dataset Loading & Inspection
    df_raw = load_dataset(dataset_path)
    df_clean, prep_report = inspect_and_clean_dataset(df_raw)

    # 2. Ingest Dataset into PostgreSQL (patients, patient_visits, temporal_patient_data)
    db_stats = ingest_dataset_to_postgresql(df_clean)

    # 3. Generate ML Features for all visits (zero leakage)
    df_features, y_all = prepare_ml_features_from_dataset(df_clean)

    # -------------------------------------------------------------
    # PIPELINE 1: BEFORE TEMPORAL DATA USAGE (BASELINE 24 FEATURES)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. BEFORE TEMPORAL DATA USAGE (BASELINE CLINICAL MODEL)")
    print("=" * 50)
    X_baseline_df = df_features[BASELINE_FEATURE_COLS].copy()
    base_preprocessor, base_num_cols, base_cat_cols = build_feature_preprocessor(BASELINE_FEATURE_COLS, X_baseline_df)
    X_base_proc = base_preprocessor.fit_transform(X_baseline_df)
    base_proc_names = list(base_preprocessor.get_feature_names_out())

    # SMOTE applied before split on full baseline dataset
    smote_base = SMOTE(random_state=RANDOM_STATE)
    X_base_res, y_base_res = smote_base.fit_resample(X_base_proc, y_all)
    base_cv_summary, base_cv_folds = cross_validate_model_suite(
        X_base_res, y_base_res, suite_label="Baseline (Before Temporal)", folds=5
    )

    X_base_tr, X_base_te, y_base_tr, y_base_te = train_test_split(
        X_base_res, y_base_res, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_base_res
    )

    base_comparison, base_best_row, base_best_model, _ = train_model_suite(
        X_base_tr, y_base_tr, X_base_te, y_base_te, base_proc_names, suite_label="Baseline (Before Temporal)"
    )

    # -------------------------------------------------------------
    # PIPELINE 2: AFTER TEMPORAL DATA USAGE (BASELINE + PREVIOUS + CHANGES + RATES)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. AFTER TEMPORAL DATA USAGE (LONGITUDINAL ENHANCED MODEL)")
    print("=" * 50)
    model_features = [c for c in ALL_MODEL_FEATURE_COLS if c in df_features.columns]
    X_temporal_df = df_features[model_features].copy()
    temp_preprocessor, temp_num_cols, temp_cat_cols = build_feature_preprocessor(model_features, X_temporal_df)
    X_temp_proc = temp_preprocessor.fit_transform(X_temporal_df)
    temp_proc_names = list(temp_preprocessor.get_feature_names_out())

    # SMOTE applied before split on full temporal-enhanced dataset
    smote_before_dist = {int(k): int(v) for k, v in y_all.value_counts().items()}
    smote_temp = SMOTE(random_state=RANDOM_STATE)
    X_temp_res, y_temp_res = smote_temp.fit_resample(X_temp_proc, y_all)
    smote_after_dist = {int(k): int(v) for k, v in pd.Series(y_temp_res).value_counts().items()}
    temp_cv_summary, temp_cv_folds = cross_validate_model_suite(
        X_temp_res, y_temp_res, suite_label="Enhanced (After Temporal)", folds=5
    )

    smote_info = {
        "before": smote_before_dist,
        "after": smote_after_dist,
        "samples_before": len(y_all),
        "samples_after": len(y_temp_res),
        "synthetic_samples_created": len(y_temp_res) - len(y_all),
        "smote_stage": "Applied before train/test split on full feature space",
    }

    X_temp_tr, X_temp_te, y_temp_tr, y_temp_te = train_test_split(
        X_temp_res, y_temp_res, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_temp_res
    )

    temp_comparison, temp_best_row, temp_best_model, feature_importances = train_model_suite(
        X_temp_tr, y_temp_tr, X_temp_te, y_temp_te, temp_proc_names, suite_label="Enhanced (After Temporal)"
    )

    # -------------------------------------------------------------
    # 3. COMPARATIVE ANALYSIS (BEFORE VS AFTER TEMPORAL LIFT)
    # -------------------------------------------------------------
    comparative_analysis = []
    base_map = {r["model_name"]: r for r in base_comparison}
    base_cv_map = {r["model_name"]: r for r in base_cv_summary}
    cross_validation_comparison = []
    for temp_cv_row in temp_cv_summary:
        mname = temp_cv_row["model_name"]
        base_cv_row = base_cv_map.get(mname, temp_cv_row)
        cv_acc_lift = round(temp_cv_row["accuracy_mean"] - base_cv_row["accuracy_mean"], 4)
        cv_f1_lift = round(temp_cv_row["f1_mean"] - base_cv_row["f1_mean"], 4)
        cv_auc_lift = round(temp_cv_row["roc_auc_mean"] - base_cv_row["roc_auc_mean"], 4)
        cv_recall_lift = round(temp_cv_row["recall_mean"] - base_cv_row["recall_mean"], 4)
        cv_balanced_lift = round(temp_cv_row["balanced_accuracy_mean"] - base_cv_row["balanced_accuracy_mean"], 4)
        cv_mcc_lift = round(temp_cv_row["mcc_mean"] - base_cv_row["mcc_mean"], 4)
        cross_validation_comparison.append({
            "model_name": mname,
            "model_type": temp_cv_row["model_type"],
            "folds": temp_cv_row["folds"],
            "before_accuracy_mean": base_cv_row["accuracy_mean"],
            "before_accuracy_std": base_cv_row["accuracy_std"],
            "after_accuracy_mean": temp_cv_row["accuracy_mean"],
            "after_accuracy_std": temp_cv_row["accuracy_std"],
            "accuracy_lift": cv_acc_lift,
            "accuracy_lift_pct": f"{cv_acc_lift * 100:+.2f}%",
            "before_f1_mean": base_cv_row["f1_mean"],
            "after_f1_mean": temp_cv_row["f1_mean"],
            "f1_lift": cv_f1_lift,
            "before_roc_auc_mean": base_cv_row["roc_auc_mean"],
            "after_roc_auc_mean": temp_cv_row["roc_auc_mean"],
            "roc_auc_lift": cv_auc_lift,
            "before_recall_mean": base_cv_row["recall_mean"],
            "after_recall_mean": temp_cv_row["recall_mean"],
            "recall_lift": cv_recall_lift,
            "before_balanced_accuracy_mean": base_cv_row["balanced_accuracy_mean"],
            "after_balanced_accuracy_mean": temp_cv_row["balanced_accuracy_mean"],
            "balanced_accuracy_lift": cv_balanced_lift,
            "before_mcc_mean": base_cv_row["mcc_mean"],
            "after_mcc_mean": temp_cv_row["mcc_mean"],
            "mcc_lift": cv_mcc_lift,
            "after_meets_95_accuracy_target": temp_cv_row["meets_95_accuracy_target"],
        })

    for temp_row in temp_comparison:
        mname = temp_row["model_name"]
        b_row = base_map.get(mname, temp_row)
        acc_lift = round(temp_row["accuracy"] - b_row["accuracy"], 4)
        f1_lift = round(temp_row["f1"] - b_row["f1"], 4)
        auc_lift = round(temp_row["roc_auc"] - b_row["roc_auc"], 4)
        rec_lift = round(temp_row["recall"] - b_row["recall"], 4)

        comparative_analysis.append({
            "model_name": mname,
            "model_type": temp_row["model_type"],
            "before_accuracy": b_row["accuracy"],
            "after_accuracy": temp_row["accuracy"],
            "accuracy_lift": acc_lift,
            "accuracy_lift_pct": f"{acc_lift * 100:+.2f}%",
            "before_f1": b_row["f1"],
            "after_f1": temp_row["f1"],
            "f1_lift": f1_lift,
            "f1_lift_pct": f"{f1_lift * 100:+.2f}%",
            "before_roc_auc": b_row["roc_auc"],
            "after_roc_auc": temp_row["roc_auc"],
            "roc_auc_lift": auc_lift,
            "roc_auc_lift_val": f"{auc_lift:+.4f}",
            "before_recall": b_row["recall"],
            "after_recall": temp_row["recall"],
            "recall_lift": rec_lift,
            "clinical_impact": "Positive temporal trajectory gain" if auc_lift >= 0 else "Baseline parity",
        })

    # ROC Data for Best Enhanced Model
    try:
        y_proba_best = temp_best_model.predict_proba(X_temp_te)[:, 1]
    except Exception:
        y_proba_best = temp_best_model.predict(X_temp_te)

    fpr, tpr, _ = roc_curve(y_temp_te, y_proba_best)
    roc_data = {
        "fpr": [round(float(x), 4) for x in fpr],
        "tpr": [round(float(x), 4) for x in tpr],
        "roc_auc": temp_best_row["roc_auc"],
        "pr_auc": temp_best_row["pr_auc"],
    }

    # 4. Persist Artifacts
    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)

    artifact = {
        "model": temp_best_model,
        "model_name": temp_best_row["model_name"],
        "preprocessor": temp_preprocessor,
        "feature_columns": model_features,
        "num_cols": temp_num_cols,
        "cat_cols": temp_cat_cols,
        "temporal_variables": ALL_10_TEMPORAL_VARIABLES,
        "baseline_features": BASELINE_FEATURE_COLS,
        "critical_numerical": CRITICAL_NUMERICAL_FEATURES,
        "critical_lifestyle": CRITICAL_LIFESTYLE_FEATURES,
    }
    joblib.dump(artifact, os.path.join(MODEL_ARTIFACT_DIR, "temporal_model_artifact.joblib"))
    joblib.dump(temp_best_model, os.path.join(MODEL_ARTIFACT_DIR, "best_model.joblib"))

    pd.DataFrame(temp_comparison).to_csv(os.path.join(MODEL_ARTIFACT_DIR, "model_comparison.csv"), index=False)
    pd.DataFrame(comparative_analysis).to_csv(os.path.join(MODEL_ARTIFACT_DIR, "before_after_temporal_comparison.csv"), index=False)
    pd.DataFrame(base_cv_summary + temp_cv_summary).to_csv(
        os.path.join(MODEL_ARTIFACT_DIR, "cross_validation_summary.csv"), index=False
    )
    pd.DataFrame(base_cv_folds + temp_cv_folds).to_csv(
        os.path.join(MODEL_ARTIFACT_DIR, "cross_validation_folds.csv"), index=False
    )
    pd.DataFrame(cross_validation_comparison).to_csv(
        os.path.join(MODEL_ARTIFACT_DIR, "before_after_cross_validation_comparison.csv"), index=False
    )

    summary = {
        "preprocessing": prep_report,
        "database": db_stats,
        "temporal_statistics": {
            "total_patients": prep_report["Unique patients"],
            "total_visits": prep_report["Total visits"],
            "patients_with_multiple_visits": prep_report["Patients with multiple visits"],
            "patients_with_one_visit": prep_report["Patients with one visit"],
            "average_visits_per_patient": prep_report["Average visits per patient"],
            "maximum_visits_per_patient": int(df_clean.groupby("patient_id")["visit_number"].max().max() if "visit_number" in df_clean.columns else 1),
            "temporal_snapshots": db_stats["temporal_records_count"],
            "numeric_delta_availability": int((df_clean.groupby("patient_id")["visit_number"].count() > 1).sum()),
            "temporal_variables_count": 10,
            "numerical_delta_features_count": 8,
            "categorical_change_features_count": 2,
        },
        "feature_architecture": {
            "source_variables_count": 10,
            "source_variables": [VARIABLE_METADATA[v]["label"] for v in ALL_10_TEMPORAL_VARIABLES],
            "numerical_delta_features_count": 8,
            "numerical_delta_features": [f"{v}_change" for v in CRITICAL_NUMERICAL_FEATURES],
            "categorical_change_features_count": 2,
            "categorical_change_features": [f"{v}_changed" for v in CRITICAL_LIFESTYLE_FEATURES],
        },
        "smote": smote_info,
        "split": {
            "total_resampled_records": len(y_temp_res),
            "train_records": len(y_temp_tr),
            "test_records": len(y_temp_te),
            "test_ratio": TEST_SIZE,
            "smote_applied_before_split": True,
        },
        "before_temporal": {
            "models": base_comparison,
            "best_model": base_best_row,
            "cross_validation": base_cv_summary,
            "features_used": len(BASELINE_FEATURE_COLS),
        },
        "after_temporal": {
            "models": temp_comparison,
            "best_model": temp_best_row,
            "cross_validation": temp_cv_summary,
            "features_used": len(model_features),
        },
        "comparative_analysis": comparative_analysis,
        "cross_validation_comparison": cross_validation_comparison,
        "comparison": temp_comparison,
        "best_model": temp_best_row,
        "best_name": temp_best_row["model_name"],
        "roc": roc_data,
        "confusion_matrix": temp_best_row["confusion_matrix"],
        "feature_importances": feature_importances,
    }

    with open(os.path.join(MODEL_ARTIFACT_DIR, "temporal_pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(MODEL_ARTIFACT_DIR, "model_metrics.json"), "w") as f:
        json.dump(temp_best_row, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("BEFORE VS AFTER TEMPORAL ACCURACY & PERFORMANCE COMPARISON")
    print("=" * 70)
    for c in comparative_analysis:
        print(f"  * {c['model_name']:<26}: Baseline Acc={c['before_accuracy']*100:.2f}% -> Temporal Acc={c['after_accuracy']*100:.2f}% (Lift: {c['accuracy_lift_pct']}) | ROC-AUC: {c['before_roc_auc']:.4f} -> {c['after_roc_auc']:.4f} (Lift: {c['roc_auc_lift_val']})")

    print("\nTRAINING COMPLETE & ALL DUAL PIPELINE ARTIFACTS SAVED SUCCESSFULLY!")
    return summary

def predict_latest_for_patient(patient_id: str) -> Dict[str, Any]:
    """
    Predict CVD risk for a patient using:
      - Latest/current visit details from patient_visits
      - Already-calculated temporal features from temporal_patient_data

    Temporal features are NOT recalculated here.
    They are directly taken from temporal_patient_data.
    """

    artifact_path = os.path.join(
        MODEL_ARTIFACT_DIR,
        "temporal_model_artifact.joblib"
    )

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            "Model artifact not found. Please train the model first."
        )

    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    preprocessor = sanitize_preprocessor(artifact["preprocessor"])
    feature_cols = artifact["feature_columns"]

    session = get_session()

    try:
        # =====================================================
        # 1. GET LATEST VISIT FROM patient_visits
        # =====================================================
        visits_query = text("""
            SELECT *
            FROM patient_visits
            WHERE patient_id = :pid
            ORDER BY visit_date ASC, id ASC
        """)

        rows = session.execute(
            visits_query,
            {"pid": str(patient_id)}
        ).mappings().all()

        if not rows:
            raise ValueError(
                f"No patient visits found for patient_id: '{patient_id}'"
            )

        visits = [dict(r) for r in rows]

        latest_visit = visits[-1]

        # =====================================================
        # 2. GET TEMPORAL DATA
        # =====================================================
        temporal_query = text("""
            SELECT *
            FROM temporal_patient_data
            WHERE patient_id = :pid
            ORDER BY id DESC
            LIMIT 1
        """)

        temporal_row = session.execute(
            temporal_query,
            {"pid": str(patient_id)}
        ).mappings().first()

        if not temporal_row:
            raise ValueError(
                f"No temporal data found for patient_id: '{patient_id}'"
            )

        temporal_data = dict(temporal_row)


        num_cols = artifact.get("num_cols", [])
        cat_cols = artifact.get("cat_cols", [])

        row_dict = {}

        for c in feature_cols:

            val = None

            if c in temporal_data:
                val = temporal_data[c]

            elif c in latest_visit:
                val = latest_visit[c]

            if c in num_cols:

                if isinstance(val, bool):
                    row_dict[c] = 1.0 if val else 0.0

                elif isinstance(val, str):

                    v_low = val.strip().lower()

                    if v_low in {
                        "no",
                        "n",
                        "false",
                        "0"
                    }:
                        row_dict[c] = 0.0

                    elif v_low in {
                        "yes",
                        "y",
                        "true",
                        "1"
                    }:
                        row_dict[c] = 1.0

                    else:
                        try:
                            row_dict[c] = float(val)
                        except Exception:
                            row_dict[c] = np.nan

                else:
                    try:
                        row_dict[c] = (
                            float(val)
                            if val is not None
                            else np.nan
                        )
                    except Exception:
                        row_dict[c] = np.nan

            else:
                row_dict[c] = (
                    str(val)
                    if val is not None and str(val) != "nan"
                    else "missing"
                )


        pred_df = pd.DataFrame([row_dict])

        for c in num_cols:
            if c in pred_df.columns:
                pred_df[c] = pd.to_numeric(
                    pred_df[c],
                    errors="coerce"
                )

        for c in cat_cols:
            if c in pred_df.columns:
                pred_df[c] = pred_df[c].astype(str)


        proc_data = preprocessor.transform(pred_df)

        prediction_label = int(
            model.predict(proc_data)[0]
        )

        try:
            probabilities = model.predict_proba(
                proc_data
            )[0].tolist()

            risk_score = round(
                float(probabilities[1]),
                4
            )

        except Exception:
            risk_score = (
                1.0
                if prediction_label == 1
                else 0.0
            )


        risk_category = (
            "High Risk"
            if prediction_label == 1 or risk_score >= 0.5
            else "Low Risk"
        )


        temporal_features = {}

        for v in ALL_10_TEMPORAL_VARIABLES:

            if v in temporal_data:
                temporal_features[v] = temporal_data[v]

            else:
                temporal_features[v] = None


        num_details = []

        for v in CRITICAL_NUMERICAL_FEATURES:

            current_value = temporal_data.get(
                f"current_{v}",
                temporal_data.get(v)
            )

            previous_value = temporal_data.get(
                f"previous_{v}"
            )

            change_value = temporal_data.get(
                f"{v}_change"
            )

            rate_value = temporal_data.get(
                f"{v}_rate"
            )

            unit = VARIABLE_METADATA.get(
                v,
                {}
            ).get("unit", "")

            lbl = VARIABLE_METADATA.get(
                v,
                {}
            ).get("label", v)

            num_details.append({
                "feature": v,
                "label": lbl,
                "unit": unit,
                "current": current_value,
                "previous": previous_value,
                "change": change_value,
                "delta": change_value,
                "rate": rate_value,

                "status": (
                    "No previous visit available"
                    if previous_value is None
                    else (
                        f"+{change_value}"
                        if change_value is not None
                        and change_value > 0
                        else str(change_value)
                    )
                ),
            })

        # =====================================================
        # 10. CATEGORICAL DETAILS
        #
        # DIRECTLY FROM temporal_patient_data
        # =====================================================

        cat_details = []

        for v in CRITICAL_LIFESTYLE_FEATURES:

            current_value = temporal_data.get(
                f"current_{v}",
                temporal_data.get(v)
            )

            previous_value = temporal_data.get(
                f"previous_{v}"
            )

            changed_value = temporal_data.get(
                f"{v}_changed"
            )

            lbl = VARIABLE_METADATA.get(
                v,
                {}
            ).get("label", v)

            cat_details.append({
                "feature": v,
                "label": lbl,
                "current": current_value,
                "previous": previous_value,
                "changed": changed_value,

                "status": (
                    "No previous visit available"
                    if previous_value is None
                    else (
                        "Changed"
                        if changed_value
                        else "Unchanged"
                    )
                ),
            })


        timeline = (
            TemporalFeatureService
            .get_patient_temporal_timeline(
                session,
                str(patient_id)
            )
        )


        return {
            "patient_id": str(patient_id),

            "patient_code": str(patient_id),

            "prediction": {
                "prediction": prediction_label,
                "probability": risk_score,
                "risk_category": risk_category,
                "model_name": artifact["model_name"],
            },

            "risk_prediction": prediction_label,

            "risk_probability": risk_score,

            "risk_category": risk_category,

            "assessment_date": str(
                latest_visit.get(
                    "visit_date",
                    datetime.utcnow()
                )
            ),

            "model_name": artifact["model_name"],

            # Already calculated temporal data
            "temporal_data": temporal_data,

            "temporal_features": {
                "numerical": num_details,
                "categorical": cat_details,
            },

            "current_values": {
                v: temporal_data.get(
                    f"current_{v}",
                    temporal_data.get(v)
                )
                for v in ALL_10_TEMPORAL_VARIABLES
            },

            "previous_values": {
                v: temporal_data.get(
                    f"previous_{v}"
                )
                for v in ALL_10_TEMPORAL_VARIABLES
            },

            "has_previous_visit": (
                temporal_data.get("previous_visit_id")
                is not None
            ),

            "is_first_visit": (
                temporal_data.get("previous_visit_id")
                is None
            ),

            "number_of_visits": len(visits),

            "visit_number": len(visits),

            "timeline": timeline,
        }

    finally:
        session.close()

def insert_manual_visit(
    patient_id: str,
    clinical_data: Dict[str, Any],
    visit_date: datetime | None = None,
) -> Dict[str, Any]:
    """
    Insert a manual clinical visit for a patient:
      1. Create/get patient in 'patients' table.
      2. Find immediately preceding visit from 'patient_visits'.
      3. Insert new visit into 'patient_visits' (permanent history).
      4. Upsert 'temporal_patient_data' with ONLY latest 2 visits.
      5. Predict CVD risk using full temporal feature vector.
      6. Store prediction in 'predictions' table.
    """
    visit_date = visit_date or datetime.utcnow()
    session = get_session()
    try:
        pid = str(patient_id)

        # 1. Create / get patient
        patient = create_patient(
            session=session,
            patient_id=pid,
            name=clinical_data.get("name") or f"Patient {pid}",
            gender=clinical_data.get("gender"),
            age=clinical_data.get("age"),
        )

        # 2. Find immediately preceding visit
        prev_visit = TemporalFeatureService.get_immediately_previous_visit(
            session=session,
            patient_id=pid,
            before_date=visit_date,
        )

        # 3. Create permanent visit record
        visit_number = (session.query(PatientVisit).filter_by(patient_id=pid).count()) + 1
        new_visit = create_visit(
            session=session,
            patient_id=pid,
            visit_date=visit_date,
            clinical_values=clinical_data,
            visit_number=visit_number,
        )

        # 4. Upsert temporal_patient_data (represents ONLY latest 2 visits)
        temp_record = TemporalFeatureService.upsert_temporal_patient_data(
            session=session,
            patient_id=pid,
            current_visit=new_visit,
            previous_visit=prev_visit,
        )

        session.commit()

        # 5. Predict using temporal vector
        pred_res = predict_latest_for_patient(pid)

        # 6. Store prediction
        with get_session() as s2:
            create_prediction(
                session=s2,
                patient_id=pid,
                visit_id=new_visit.id,
                visit_date=visit_date,
                prediction=pred_res["risk_prediction"],
                probability=pred_res["risk_probability"],
                risk_level=pred_res["risk_category"],
                model_name=pred_res["model_name"],
            )
            s2.commit()

        return pred_res
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()



def load_visits_from_db(): pass
def load_mapping(): return {}
def generate_temporal_features(df): return df
