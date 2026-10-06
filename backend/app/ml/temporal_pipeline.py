from __future__ import annotations

import ast
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Python 3.14 ast compatibility
if not hasattr(ast, 'Num'):
    ast.Num = ast.Constant
if not hasattr(ast, 'Str'):
    ast.Str = ast.Constant
if not hasattr(ast, 'Bytes'):
    ast.Bytes = ast.Constant
if not hasattr(ast, 'NameConstant'):
    ast.NameConstant = ast.Constant

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer

if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = property(
        lambda self: np.asarray(self.statistics_).dtype if hasattr(self, 'statistics_') else np.float64
    )
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import text

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from app.config import (
    DATASET_PATH,
    MODEL_ARTIFACT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from app.database import engine, init_db

# ── CRITICAL CARDIOVASCULAR FEATURES ─────────────────────────────────────────
# Temporal features (previous_*, *_change, *_rate) are ONLY calculated for these!
CRITICAL_CARDIO_KEYS = [
    'systolic_bp',
    'diastolic_bp',
    'cholesterol',
    'bmi',
    'glucose',
    'weight',
]

ALIASES = {
    'patient_id': ['patient_id', 'patientid', 'patient_code', 'id', 'pid'],
    'visit_date': ['visit_date', 'visit_timestamp', 'date', 'timestamp', 'encounter_date', 'visit_dt'],
    'target': ['prediction', 'target', 'heart_disease', 'disease', 'outcome', 'label', 'cardio'],
    'systolic_bp': ['systolic_bp', 'systolic', 'trestbps', 'resting_blood_pressure', 'sys_bp', 'ap_hi'],
    'diastolic_bp': ['diastolic_bp', 'diastolic', 'dia_bp', 'ap_lo'],
    'cholesterol': ['cholesterol', 'chol', 'tot_chol', 'total_cholesterol'],
    'bmi': ['bmi', 'body_mass_index'],
    'glucose': ['glucose', 'blood_sugar', 'fasting_blood_sugar', 'fbs', 'gluc'],
    'weight': ['weight', 'body_weight', 'wt'],
    'resting_heart_rate': ['resting_heart_rate', 'resting_hr', 'pulse'],
    'max_heart_rate': ['max_heart_rate', 'thalach', 'max_hr', 'thalachh'],
    'hdl': ['hdl', 'hdl_cholesterol'],
    'ldl': ['ldl', 'ldl_cholesterol'],
    'hba1c': ['hba1c', 'glycated_hemoglobin'],
    'smoking': ['smoking', 'smoker', 'smoking_status', 'smoke'],
    'activity': ['physical_activity', 'activity', 'activity_level', 'active'],
    'diet': ['diet', 'diet_status'],
    'alcohol': ['alcohol', 'alcohol_status', 'alcohol_use', 'alco'],
    'stress': ['stress_level', 'stress'],
}

LEAKAGE_TERMS = ['prediction', 'target', 'outcome', 'label', 'probability', 'cardio_disease']


@dataclass
class DatasetMapping:
    dataset_path: str
    patient_id: str | None
    visit_date: str | None
    target: str | None
    numerical: list[str]
    categorical: list[str]
    clinical: list[str]
    lifestyle: list[str]
    critical_features: list[str]
    other_features: list[str]
    systolic_bp: str | None = None
    diastolic_bp: str | None = None
    cholesterol: str | None = None
    bmi: str | None = None
    glucose: str | None = None
    weight: str | None = None
    smoking: str | None = None
    physical_activity: str | None = None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    return df


def find_column(cols: list[str], names: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    for n in names:
        for c in cols:
            if n in c.lower():
                return c
    return None


def load_dataset(path: str | None = None) -> pd.DataFrame:
    path = path or DATASET_PATH
    if not os.path.exists(path):
        candidates = [
            path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "heart_disease_prediction_2026.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "heart_disease_prediction_2026.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "heart_disease_prediction_2026.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    ext = Path(path).suffix.lower()
    if ext in {'.xls', '.xlsx'}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return normalize_columns(df)


def analyze_dataset(df: pd.DataFrame, path: str) -> DatasetMapping:
    cols = list(df.columns)
    vals = {k: find_column(cols, v) for k, v in ALIASES.items()}
    exclude = {x for x in [vals['patient_id'], vals['visit_date'], vals['target']] if x}

    numerical = []
    categorical = []
    for c in cols:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.8:
            numerical.append(c)
        else:
            categorical.append(c)

    clinical = sorted(set(numerical + categorical))
    lifestyle_keys = {'smoking', 'activity', 'diet', 'alcohol', 'stress'}
    lifestyle = sorted([vals[k] for k in lifestyle_keys if vals.get(k) and vals[k] in cols])

    # Identify which critical features actually exist in the dataset
    # (Systolic BP, Diastolic BP, Total Cholesterol, BMI, and Glucose/Weight if present as continuous measurements)
    critical_features = []
    for k in ['systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi', 'weight']:
        actual_col = vals.get(k)
        if actual_col and actual_col in cols and actual_col not in critical_features:
            if pd.to_numeric(df[actual_col], errors='coerce').notna().mean() > 0.5:
                critical_features.append(actual_col)

    if 'glucose' in cols and 'glucose' not in critical_features:
        if pd.to_numeric(df['glucose'], errors='coerce').notna().mean() > 0.5:
            critical_features.append('glucose')

    # All other features are retained as current/latest values only
    other_features = [c for c in clinical if c not in critical_features and c not in exclude]

    mapping = DatasetMapping(
        dataset_path=path,
        patient_id=vals['patient_id'],
        visit_date=vals['visit_date'],
        target=vals['target'],
        numerical=numerical,
        categorical=categorical,
        clinical=clinical,
        lifestyle=lifestyle,
        critical_features=critical_features,
        other_features=other_features,
        systolic_bp=vals.get('systolic_bp'),
        diastolic_bp=vals.get('diastolic_bp'),
        cholesterol=vals.get('cholesterol'),
        bmi=vals.get('bmi'),
        glucose=vals.get('glucose'),
        weight=vals.get('weight'),
        smoking=vals.get('smoking'),
        physical_activity=vals.get('activity'),
    )

    print("\nDATASET ANALYSIS & FEATURE SEPARATION", flush=True)
    print("--------------------------------------", flush=True)
    print(f"Dataset path: {path}", flush=True)
    print(f"Total records: {len(df)}", flush=True)
    print(f"Total columns: {len(cols)}", flush=True)
    print(f"Patient ID column: {mapping.patient_id}", flush=True)
    print(f"Visit date column: {mapping.visit_date or 'Auto-generated baseline date'}", flush=True)
    print(f"Target column: {mapping.target}", flush=True)
    print(f"\nCRITICAL TEMPORAL FEATURES ({len(critical_features)}):")
    for c in critical_features:
        print(f"  • {c} (will generate previous_{c}, {c}_change, {c}_rate)", flush=True)
    print(f"\nOTHER CURRENT FEATURES ({len(other_features)}):")
    for c in other_features:
        print(f"  • {c} (retained as current/latest value ONLY)", flush=True)

    return mapping


def clean_dataset(df: pd.DataFrame, mapping: DatasetMapping) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    original_records = len(df)
    original_columns = len(df.columns)

    df = df.replace({'?': np.nan, '': np.nan, 'NA': np.nan, 'null': np.nan, 'None': np.nan}).dropna(how='all')
    empty_rows_removed = original_records - len(df)

    if not mapping.patient_id or mapping.patient_id not in df.columns:
        df['patient_id'] = [f"P{i+1:06d}" for i in range(len(df))]
        mapping.patient_id = 'patient_id'
    else:
        df[mapping.patient_id] = df[mapping.patient_id].astype(str).str.strip()
        df = df[df[mapping.patient_id].ne('') & df[mapping.patient_id].ne('nan')].copy()

    invalid_dates = 0
    if mapping.visit_date and mapping.visit_date in df.columns:
        dt = pd.to_datetime(df[mapping.visit_date], errors='coerce')
        invalid_dates = int(dt.isna().sum())
        df = df.loc[dt.notna()].copy()
        df['visit_date'] = dt.loc[dt.notna()].dt.tz_localize(None)
    else:
        base_time = datetime(2025, 1, 1)
        df['visit_date'] = [base_time + timedelta(days=i % 365, hours=(i // 365) % 24) for i in range(len(df))]
        mapping.visit_date = 'visit_date'

    before_dup = len(df)
    df = df.drop_duplicates(subset=[mapping.patient_id, 'visit_date'], keep='last').copy()
    duplicate_rows_removed = before_dup - len(df)

    missing_handled = int(df.isna().sum().sum())

    for c in mapping.numerical:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            med = s.median()
            if pd.isna(med):
                med = 0.0
            df[c] = s.fillna(med)

    for c in mapping.categorical:
        if c in df.columns:
            mode_val = df[c].dropna().mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
            df[c] = df[c].fillna(fill_val).astype(str).str.strip()

    if mapping.target and mapping.target in df.columns:
        df[mapping.target] = pd.to_numeric(df[mapping.target], errors='coerce')
        df = df[df[mapping.target].notna()].copy()
        df[mapping.target] = (df[mapping.target] > 0).astype(int)

    df = df.sort_values([mapping.patient_id, 'visit_date']).reset_index(drop=True)

    vc = df[mapping.patient_id].value_counts()
    unique_patients = int(vc.size)
    multi_visit_patients = int((vc > 1).sum())
    single_visit_patients = int((vc == 1).sum())

    report = {
        'Original records': original_records,
        'Original columns': original_columns,
        'Empty rows removed': empty_rows_removed,
        'Duplicate rows removed': duplicate_rows_removed,
        'Invalid dates removed/fixed': invalid_dates,
        'Missing values handled': missing_handled,
        'Valid records after cleaning': len(df),
        'Unique patients': unique_patients,
        'Patients with multiple visits': multi_visit_patients,
        'Patients with one visit': single_visit_patients,
        'Earliest visit': str(df['visit_date'].min()),
        'Latest visit': str(df['visit_date'].max()),
    }

    print("\nPREPROCESSING REPORT", flush=True)
    print("--------------------", flush=True)
    for k, v in report.items():
        print(f"{k}: {v}", flush=True)

    return df, report


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def ensure_temporal_tables():
    init_db()
    dialect = engine.dialect.name
    id_type = 'INTEGER PRIMARY KEY AUTOINCREMENT' if dialect == 'sqlite' else 'SERIAL PRIMARY KEY'
    raw_type = 'TEXT'

    with engine.begin() as con:
        # 1. Ensure patient_visits has all necessary columns
        if dialect != 'sqlite':
            con.execute(text('DROP TABLE IF EXISTS patient_visits CASCADE'))
            con.execute(text(f'''CREATE TABLE patient_visits (
                id {id_type},
                patient_id INTEGER,
                source_patient_id VARCHAR(100) NOT NULL,
                visit_timestamp TIMESTAMP NOT NULL,
                visit_date TIMESTAMP NOT NULL,
                age FLOAT,
                gender VARCHAR(50),
                bmi FLOAT,
                chest_pain_type VARCHAR(100),
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                resting_heart_rate FLOAT,
                max_heart_rate FLOAT,
                cholesterol FLOAT,
                hdl FLOAT,
                ldl FLOAT,
                fasting_blood_sugar VARCHAR(50),
                hba1c FLOAT,
                diabetes VARCHAR(50),
                resting_ecg VARCHAR(100),
                exercise_angina VARCHAR(50),
                oldpeak FLOAT,
                st_slope VARCHAR(50),
                num_major_vessels FLOAT,
                thalassemia VARCHAR(100),
                smoking VARCHAR(50),
                family_history VARCHAR(50),
                physical_activity VARCHAR(100),
                stress_level FLOAT,
                target FLOAT,
                raw_data {raw_type},
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )'''))
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_patient_visits_source_patient_id ON patient_visits (source_patient_id)'))
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_patient_visits_visit_date ON patient_visits (visit_date)'))
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_patient_visits_source_date ON patient_visits (source_patient_id, visit_date)'))

            # 2. Ensure temporal_patient_features has exact required structure
            con.execute(text('DROP TABLE IF EXISTS temporal_patient_features CASCADE'))
            con.execute(text(f'''CREATE TABLE temporal_patient_features (
                id {id_type},
                patient_id VARCHAR(100) NOT NULL,
                visit_date TIMESTAMP NOT NULL,
                previous_visit_date TIMESTAMP,
                days_since_previous_visit FLOAT,
                has_previous_visit INTEGER NOT NULL DEFAULT 0,
                visit_number INTEGER DEFAULT 1,
                
                -- Current Critical Features
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                cholesterol FLOAT,
                bmi FLOAT,
                
                -- Previous Critical Values (NULL on first visit)
                previous_systolic_bp FLOAT,
                previous_diastolic_bp FLOAT,
                previous_cholesterol FLOAT,
                previous_bmi FLOAT,
                
                -- Temporal Changes (NULL on first visit)
                systolic_bp_change FLOAT,
                diastolic_bp_change FLOAT,
                cholesterol_change FLOAT,
                bmi_change FLOAT,
                
                -- Rates of Change (NULL on first visit)
                systolic_bp_rate FLOAT,
                diastolic_bp_rate FLOAT,
                cholesterol_rate FLOAT,
                bmi_rate FLOAT,
                
                -- Other Features (Current values only)
                age FLOAT,
                gender VARCHAR(50),
                chest_pain_type VARCHAR(100),
                resting_heart_rate FLOAT,
                max_heart_rate FLOAT,
                hdl FLOAT,
                ldl FLOAT,
                fasting_blood_sugar VARCHAR(50),
                hba1c FLOAT,
                diabetes VARCHAR(50),
                resting_ecg VARCHAR(100),
                exercise_angina VARCHAR(50),
                oldpeak FLOAT,
                st_slope VARCHAR(50),
                num_major_vessels FLOAT,
                thalassemia VARCHAR(100),
                smoking VARCHAR(50),
                family_history VARCHAR(50),
                physical_activity VARCHAR(100),
                stress_level FLOAT,
                
                target FLOAT,
                features {raw_type} NOT NULL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )'''))
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_tpf_patient_id ON temporal_patient_features (patient_id)'))
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_tpf_visit_date ON temporal_patient_features (visit_date)'))
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_tpf_patient_date ON temporal_patient_features (patient_id, visit_date)'))

            # 3. Ensure predictions table exists and has necessary columns
            con.execute(text(f'''CREATE TABLE IF NOT EXISTS predictions (
                id {id_type},
                patient_id INTEGER,
                patient_code VARCHAR(100),
                visit_id INTEGER,
                visit_date TIMESTAMP,
                prediction INTEGER NOT NULL,
                probability FLOAT,
                risk_level VARCHAR(50),
                model_name VARCHAR(100),
                model_type VARCHAR(50) DEFAULT 'temporal',
                created_at TIMESTAMP
            )'''))
            alter_pred_cols = [
                ('patient_code', 'VARCHAR(100)'),
                ('visit_date', 'TIMESTAMP'),
                ('risk_level', 'VARCHAR(50)'),
                ('model_type', "VARCHAR(50) DEFAULT 'temporal'"),
            ]
            for col, typ in alter_pred_cols:
                try:
                    con.execute(text(f'ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {typ}'))
                except Exception:
                    pass
            con.execute(text('CREATE INDEX IF NOT EXISTS ix_predictions_patient_code ON predictions (patient_code)'))

    print("PostgreSQL connection: SUCCESS", flush=True)
    print("patient_visits table: READY", flush=True)
    print("temporal_patient_features table: READY", flush=True)
    print("predictions table: READY", flush=True)


def insert_patient_visits(df: pd.DataFrame, mapping: DatasetMapping) -> Dict[str, int]:
    ensure_temporal_tables()
    dialect = engine.dialect.name

    stats = {
        'records_already_existing': 0,
        'new_records_inserted': 0,
        'records_updated': 0,
        'duplicates_skipped': 0,
    }

    with engine.begin() as con:
        existing_count = con.execute(text('SELECT COUNT(*) FROM patient_visits WHERE source_patient_id IS NOT NULL')).scalar() or 0
        stats['records_already_existing'] = int(existing_count)

        patient_codes = df[mapping.patient_id].astype(str).unique().tolist()
        now = datetime.utcnow()

        p_batch = [{'c': p, 't': now} for p in patient_codes]
        if dialect == 'postgresql':
            con.execute(
                text('INSERT INTO patients (patient_code, created_at) VALUES (:c, :t) ON CONFLICT (patient_code) DO NOTHING'),
                p_batch
            )
        else:
            con.execute(
                text('INSERT OR IGNORE INTO patients (patient_code, created_at) VALUES (:c, :t)'),
                p_batch
            )

        con.execute(text('DELETE FROM predictions'))
        con.execute(text('DELETE FROM patient_visits'))

        id_rows = con.execute(text('SELECT id, patient_code FROM patients')).mappings().all()
        id_map = {str(r['patient_code']): r['id'] for r in id_rows}

        # Fast vectorized dataframe preparation
        ins_df = pd.DataFrame()
        pids = df[mapping.patient_id].astype(str)
        ins_df['patient_id'] = [id_map.get(pid, 1) for pid in pids]
        ins_df['source_patient_id'] = pids
        ins_df['visit_timestamp'] = df['visit_date']
        ins_df['visit_date'] = df['visit_date']

        for c in ['age', 'gender', 'bmi', 'chest_pain_type', 'systolic_bp', 'diastolic_bp',
                  'resting_heart_rate', 'max_heart_rate', 'cholesterol', 'hdl', 'ldl',
                  'fasting_blood_sugar', 'hba1c', 'diabetes', 'resting_ecg', 'exercise_angina',
                  'oldpeak', 'st_slope', 'num_major_vessels', 'thalassemia', 'smoking',
                  'family_history', 'physical_activity', 'stress_level']:
            ins_df[c] = df[c] if c in df.columns else None

        tcol = mapping.target if mapping.target and mapping.target in df.columns else ('target' if 'target' in df.columns else None)
        ins_df['target'] = df[tcol] if tcol else None
        ins_df['raw_data'] = df.to_json(orient='records', lines=True).splitlines()
        ins_df['created_at'] = now
        ins_df['updated_at'] = now

        records = ins_df.where(pd.notna(ins_df), None).to_dict(orient='records')

        batch_size = 10000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            con.execute(text('''
                INSERT INTO patient_visits (
                    patient_id, source_patient_id, visit_timestamp, visit_date,
                    age, gender, bmi, chest_pain_type, systolic_bp, diastolic_bp,
                    resting_heart_rate, max_heart_rate, cholesterol, hdl, ldl,
                    fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                    oldpeak, st_slope, num_major_vessels, thalassemia, smoking,
                    family_history, physical_activity, stress_level, target, raw_data,
                    created_at, updated_at
                )
                VALUES (
                    :patient_id, :source_patient_id, :visit_timestamp, :visit_date,
                    :age, :gender, :bmi, :chest_pain_type, :systolic_bp, :diastolic_bp,
                    :resting_heart_rate, :max_heart_rate, :cholesterol, :hdl, :ldl,
                    :fasting_blood_sugar, :hba1c, :diabetes, :resting_ecg, :exercise_angina,
                    :oldpeak, :st_slope, :num_major_vessels, :thalassemia, :smoking,
                    :family_history, :physical_activity, :stress_level, :target, :raw_data,
                    :created_at, :updated_at
                )
            '''), batch)

        stats['new_records_inserted'] = len(records)

    print("\nDATABASE INGESTION ANALYSIS", flush=True)
    print("---------------------------", flush=True)
    print(f"Records already existing: {stats['records_already_existing']}", flush=True)
    print(f"New records inserted: {stats['new_records_inserted']}", flush=True)

    return stats


def load_visits_from_db(patient_id: str | None = None) -> pd.DataFrame:
    where = 'WHERE source_patient_id = :p' if patient_id else ''
    params = {'p': str(patient_id)} if patient_id else {}

    with engine.begin() as con:
        rows = con.execute(
            text(f'SELECT * FROM patient_visits {where} ORDER BY source_patient_id, visit_date ASC'),
            params
        ).mappings().all()

    if not rows:
        return pd.DataFrame()

    out = []
    for r in rows:
        d = dict(r)
        d['patient_id'] = str(d.get('source_patient_id') or d.get('patient_id'))
        d['visit_date'] = pd.to_datetime(d['visit_date'])
        d.pop('raw_data', None)
        out.append(d)

    df = normalize_columns(pd.DataFrame(out))
    if 'source_patient_id' in df.columns:
        df['patient_id'] = df['source_patient_id'].astype(str)
    return df


def generate_temporal_features(visits: pd.DataFrame, mapping: DatasetMapping) -> pd.DataFrame:
    """
    Generate temporal features ONLY for critical cardiovascular features.
    """
    if visits.empty:
        return pd.DataFrame()

    df = normalize_columns(visits).copy()
    pcol = 'patient_id' if 'patient_id' in df.columns else (mapping.patient_id if mapping.patient_id in df.columns else 'source_patient_id')
    tcol = mapping.target if mapping.target and mapping.target in df.columns else ('target' if 'target' in df.columns else 'prediction')

    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df = df[df['visit_date'].notna()].sort_values([pcol, 'visit_date']).reset_index(drop=True)

    # Patient and Visit Metadata
    temporal_df = pd.DataFrame()
    temporal_df['patient_id'] = df[pcol].astype(str)
    temporal_df['visit_date'] = df['visit_date']

    # Previous visit date & days since previous visit
    prev_date = df.groupby(pcol)['visit_date'].shift(1)
    temporal_df['previous_visit_date'] = prev_date

    days_diff = (df['visit_date'] - prev_date).dt.total_seconds() / 86400.0
    temporal_df['days_since_previous_visit'] = days_diff

    # First visit indicator: 0 for first visit, 1 for subsequent visits
    has_prev = prev_date.notna().astype(int)
    temporal_df['has_previous_visit'] = has_prev
    temporal_df['visit_number'] = df.groupby(pcol).cumcount() + 1

    # Identify critical features present
    critical_cols = [c for c in mapping.critical_features if c in df.columns]
    if not critical_cols:
        for k in ['systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi']:
            if k in df.columns:
                critical_cols.append(k)

    safe_days = days_diff.fillna(1.0).replace(0, 1.0)

    # ── 1. Critical Temporal Features (Current, Previous, Change, Rate) ──────
    for feat in critical_cols:
        cur_series = pd.to_numeric(df[feat], errors='coerce')
        temporal_df[feat] = cur_series

        # Previous value (NaN on first visit)
        prev_series = df.groupby(pcol)[feat].shift(1)
        prev_num = pd.to_numeric(prev_series, errors='coerce')
        temporal_df[f'previous_{feat}'] = prev_num

        # Temporal change: current - previous (NaN on first visit)
        change_series = cur_series - prev_num
        temporal_df[f'{feat}_change'] = change_series

        # Rate of change: change / days (NaN on first visit)
        rate_series = np.where(has_prev == 1, change_series / safe_days, np.nan)
        temporal_df[f'{feat}_rate'] = rate_series

    # ── 2. Other Features: Current / Latest values ONLY ───────────────────────
    exclude_cols = {pcol, 'patient_id', 'source_patient_id', 'visit_date', 'previous_visit_date',
                    'visit_timestamp', 'id', 'created_at', 'updated_at', 'raw_data', tcol, 'target'}
    exclude_cols.update(critical_cols)

    for col in df.columns:
        if col in exclude_cols:
            continue
        temporal_df[col] = df[col]

    # Target column
    if tcol in df.columns:
        temporal_df['target'] = pd.to_numeric(df[tcol], errors='coerce')
    elif 'target' in df.columns:
        temporal_df['target'] = pd.to_numeric(df['target'], errors='coerce')
    else:
        temporal_df['target'] = None

    total_visits = len(temporal_df)
    unique_pats = temporal_df['patient_id'].nunique()
    multi_visits = int((temporal_df.groupby('patient_id').size() > 1).sum())

    print("\nTEMPORAL FEATURE EXTRACTION COMPLETE", flush=True)
    print("------------------------------------", flush=True)
    print(f"Total records processed: {total_visits}", flush=True)
    print(f"Unique patients: {unique_pats}", flush=True)
    print(f"Patients with multiple visits: {multi_visits}", flush=True)
    print(f"Critical temporal features: {critical_cols}", flush=True)
    print(f"Generated columns ({len(temporal_df.columns)}): {list(temporal_df.columns)}", flush=True)

    return temporal_df


def save_temporal_features(temporal_df: pd.DataFrame):
    if temporal_df.empty:
        return

    with engine.begin() as con:
        con.execute(text('DELETE FROM temporal_patient_features'))

        feature_cols = [c for c in temporal_df.columns if c not in {'patient_id', 'visit_date', 'target'}]
        json_lines = temporal_df[feature_cols].to_json(orient='records', lines=True).splitlines()

        now = datetime.utcnow()
        save_df = temporal_df.copy()
        save_df['features'] = json_lines
        save_df['created_at'] = now
        save_df['updated_at'] = now

        # Ensure all columns exist in save_df
        req_cols = [
            'patient_id', 'visit_date', 'previous_visit_date', 'days_since_previous_visit',
            'has_previous_visit', 'visit_number', 'systolic_bp', 'diastolic_bp', 'cholesterol',
            'bmi', 'previous_systolic_bp', 'previous_diastolic_bp', 'previous_cholesterol',
            'previous_bmi', 'systolic_bp_change', 'diastolic_bp_change', 'cholesterol_change',
            'bmi_change', 'systolic_bp_rate', 'diastolic_bp_rate', 'cholesterol_rate',
            'bmi_rate', 'age', 'gender', 'chest_pain_type', 'resting_heart_rate', 'max_heart_rate',
            'hdl', 'ldl', 'fasting_blood_sugar', 'hba1c', 'diabetes', 'resting_ecg', 'exercise_angina',
            'oldpeak', 'st_slope', 'num_major_vessels', 'thalassemia', 'smoking', 'family_history',
            'physical_activity', 'stress_level', 'target', 'features', 'created_at', 'updated_at'
        ]
        for c in req_cols:
            if c not in save_df.columns:
                save_df[c] = None

        db_df = save_df[req_cols]
        records = db_df.where(pd.notna(db_df), None).to_dict(orient='records')

        batch_size = 10000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            con.execute(text('''
                INSERT INTO temporal_patient_features (
                    patient_id, visit_date, previous_visit_date, days_since_previous_visit,
                    has_previous_visit, visit_number, systolic_bp, diastolic_bp, cholesterol,
                    bmi, previous_systolic_bp, previous_diastolic_bp, previous_cholesterol,
                    previous_bmi, systolic_bp_change, diastolic_bp_change, cholesterol_change,
                    bmi_change, systolic_bp_rate, diastolic_bp_rate, cholesterol_rate,
                    bmi_rate, age, gender, chest_pain_type, resting_heart_rate, max_heart_rate,
                    hdl, ldl, fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                    oldpeak, st_slope, num_major_vessels, thalassemia, smoking, family_history,
                    physical_activity, stress_level, target, features, created_at, updated_at
                )
                VALUES (
                    :patient_id, :visit_date, :previous_visit_date, :days_since_previous_visit,
                    :has_previous_visit, :visit_number, :systolic_bp, :diastolic_bp, :cholesterol,
                    :bmi, :previous_systolic_bp, :previous_diastolic_bp, :previous_cholesterol,
                    :previous_bmi, :systolic_bp_change, :diastolic_bp_change, :cholesterol_change,
                    :bmi_change, :systolic_bp_rate, :diastolic_bp_rate, :cholesterol_rate,
                    :bmi_rate, :age, :gender, :chest_pain_type, :resting_heart_rate, :max_heart_rate,
                    :hdl, :ldl, :fasting_blood_sugar, :hba1c, :diabetes, :resting_ecg, :exercise_angina,
                    :oldpeak, :st_slope, :num_major_vessels, :thalassemia, :smoking, :family_history,
                    :physical_activity, :stress_level, :target, :features, :created_at, :updated_at
                )
            '''), batch)


def prepare_feature_matrices(temporal_df: pd.DataFrame, mapping: DatasetMapping) -> Dict[str, Any]:
    """
    Prepare both Static (baseline) and Temporal feature matrices.
    """
    y = pd.to_numeric(temporal_df['target'], errors='coerce')
    valid_idx = y.notna()
    data = temporal_df.loc[valid_idx].copy()
    y = y.loc[valid_idx].astype(int)

    blocked_cols = {'patient_id', 'visit_date', 'previous_visit_date', 'target', 'features'}
    leakage_cols = {c for c in data.columns if c != 'target' and any(term in c.lower() for term in LEAKAGE_TERMS)}

    # All candidate feature columns
    all_feature_cols = [c for c in data.columns if c not in blocked_cols and c not in leakage_cols]

    # Temporal feature set: contains current + critical previous + critical changes + critical rates + has_prev + days_since
    temporal_features = all_feature_cols

    # Static feature set: contains ONLY current features (no previous_*, *_change, *_rate, has_previous_visit, days_since)
    temporal_only_cols = {c for c in temporal_features if c.startswith('previous_') or c.endswith('_change') or c.endswith('_rate') or c in {'has_previous_visit', 'days_since_previous_visit', 'visit_number'}}
    static_features = [c for c in temporal_features if c not in temporal_only_cols]

    # Build typed DataFrames
    X_temporal = data[temporal_features].copy()
    X_static = data[static_features].copy()

    # Identify numeric and categorical features for temporal matrix
    temp_num = []
    temp_cat = []
    for c in X_temporal.columns:
        s = pd.to_numeric(X_temporal[c], errors='coerce')
        if s.notna().mean() > 0.6:
            X_temporal[c] = s
            temp_num.append(c)
        else:
            X_temporal[c] = X_temporal[c].astype(str).fillna('Unknown')
            temp_cat.append(c)

    # Identify numeric and categorical features for static matrix
    stat_num = []
    stat_cat = []
    for c in X_static.columns:
        s = pd.to_numeric(X_static[c], errors='coerce')
        if s.notna().mean() > 0.6:
            X_static[c] = s
            stat_num.append(c)
        else:
            X_static[c] = X_static[c].astype(str).fillna('Unknown')
            stat_cat.append(c)

    critical_changes = [c for c in X_temporal.columns if c.endswith('_change')]
    critical_rates = [c for c in X_temporal.columns if c.endswith('_rate')]
    critical_prevs = [c for c in X_temporal.columns if c.startswith('previous_')]

    groups = {
        'temporal_all_features': list(X_temporal.columns),
        'temporal_numeric_features': temp_num,
        'temporal_categorical_features': temp_cat,
        'static_all_features': list(X_static.columns),
        'static_numeric_features': stat_num,
        'static_categorical_features': stat_cat,
        'current_features': [c for c in X_temporal.columns if c not in temporal_only_cols],
        'critical_previous_features': critical_prevs,
        'critical_change_features': critical_changes,
        'critical_rate_features': critical_rates,
        'time_indicators': [c for c in ['has_previous_visit', 'days_since_previous_visit', 'visit_number'] if c in X_temporal.columns],
    }

    print("\nMODEL FEATURE ARCHITECTURE", flush=True)
    print("--------------------------", flush=True)
    print(f"Static Model Features ({len(groups['static_all_features'])}): {groups['static_all_features']}", flush=True)
    print(f"\nTemporal Model Features ({len(groups['temporal_all_features'])}):", flush=True)
    print(f"  • Current Features ({len(groups['current_features'])})", flush=True)
    print(f"  • Critical Previous Values ({len(critical_prevs)}): {critical_prevs}", flush=True)
    print(f"  • Critical Feature Changes ({len(critical_changes)}): {critical_changes}", flush=True)
    print(f"  • Critical Feature Rates ({len(critical_rates)}): {critical_rates}", flush=True)
    print(f"  • Temporal Indicators ({len(groups['time_indicators'])}): {groups['time_indicators']}", flush=True)

    return {
        'X_temporal': X_temporal,
        'X_static': X_static,
        'y': y,
        'groups': groups,
    }


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append((
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]),
            cat_cols
        ))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def apply_smote_before_split(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    counts_before = {int(k): int(v) for k, v in y.value_counts().to_dict().items()}
    min_class_count = min(counts_before.values()) if counts_before else 0

    if len(counts_before) < 2 or min_class_count < 2:
        return X, y, {
            'before': counts_before,
            'after': counts_before,
            'samples_before': len(y),
            'samples_after': len(y),
            'synthetic_samples_created': 0,
        }

    k_neigh = min(5, min_class_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neigh)
    X_res, y_res = smote.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)

    counts_after = {int(k): int(v) for k, v in y_res.value_counts().to_dict().items()}
    synthetic_created = len(y_res) - len(y)

    return X_res, y_res, {
        'before': counts_before,
        'after': counts_after,
        'samples_before': len(y),
        'samples_after': len(y_res),
        'synthetic_samples_created': synthetic_created,
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type_label: str = "Temporal",
    artifact_dir: str = MODEL_ARTIFACT_DIR
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Any]:
    """
    Train and evaluate the 6 approved ML models on training and test splits.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=45, max_depth=12, random_state=RANDOM_STATE, n_jobs=1),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=45,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=1
        )
    else:
        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=40, max_depth=4, random_state=RANDOM_STATE)

    sample_size = min(8000, len(X_train))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), sample_size, replace=False)
    X_sample, y_sample = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
    test_sample_size = min(2000, len(X_test))
    test_sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), test_sample_size, replace=False)
    X_test_sample, y_test_sample = X_test.iloc[test_sample_idx], y_test.iloc[test_sample_idx]

    models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=1)

    comparison_rows = []
    trained_models = {}

    print(f"\n--- Training {model_type_label} Models ---", flush=True)

    for name, model in models.items():
        t_start = time.time()
        if name == 'KNN' and len(X_train) > 15000:
            model.fit(X_sample, y_sample)
            pred = model.predict(X_test_sample)
            eval_y = y_test_sample
            eval_X = X_test_sample
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            eval_y = y_test
            eval_X = X_test

        trained_models[name] = model

        try:
            proba = model.predict_proba(eval_X)
            score = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
            auc = roc_auc_score(eval_y, score)
        except Exception:
            auc = np.nan

        cm = confusion_matrix(eval_y, pred)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            cv = float(cross_val_score(model, X_train[:2500], y_train[:2500], cv=3, scoring='accuracy').mean())
        except Exception:
            cv = np.nan

        row = {
            'model_name': name,
            'model_type': model_type_label.lower(),
            'accuracy': round(float(accuracy_score(eval_y, pred)), 4),
            'precision': round(float(precision_score(eval_y, pred, average='weighted', zero_division=0)), 4),
            'recall': round(float(recall_score(eval_y, pred, average='weighted', zero_division=0)), 4),
            'specificity': round(float(specificity), 4),
            'f1': round(float(f1_score(eval_y, pred, average='weighted', zero_division=0)), 4),
            'roc_auc': None if pd.isna(auc) else round(float(auc), 4),
            'cv_score': None if pd.isna(cv) else round(cv, 4),
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'matrix': cm.tolist()},
            'training_time_seconds': round(time.time() - t_start, 2),
        }
        comparison_rows.append(row)
        print(f"{name} ({model_type_label}): Accuracy={row['accuracy']:.4f}, F1={row['f1']:.4f}, ROC-AUC={row['roc_auc']}", flush=True)

    comp_df = pd.DataFrame(comparison_rows).sort_values(['roc_auc', 'f1', 'recall'], ascending=False, na_position='last')
    best_row = comp_df.iloc[0].to_dict()
    best_model = trained_models[best_row['model_name']]

    return comparison_rows, best_row, best_model


def train_static_and_temporal_pipelines(
    matrices: Dict[str, Any],
    artifact_dir: str = MODEL_ARTIFACT_DIR
) -> Dict[str, Any]:
    """
    Train and compare BOTH Static and Temporal model pipelines.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    groups = matrices['groups']
    y = matrices['y']

    # ── 1. STATIC PIPELINE ───────────────────────────────────────────────────
    X_static = matrices['X_static']
    preprocessor_static = build_preprocessor(groups['static_numeric_features'], groups['static_categorical_features'])
    X_static_proc = preprocessor_static.fit_transform(X_static)
    static_feat_names = list(preprocessor_static.get_feature_names_out())
    X_static_df = pd.DataFrame(X_static_proc, columns=static_feat_names)

    X_stat_smote, y_stat_smote, smote_stat_info = apply_smote_before_split(X_static_df, y)
    X_tr_stat, X_te_stat, y_tr_stat, y_te_stat = train_test_split(
        X_stat_smote, y_stat_smote, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_stat_smote if y_stat_smote.nunique() > 1 else None
    )

    static_comp, static_best_row, static_best_model = train_and_evaluate_models(
        X_tr_stat, X_te_stat, y_tr_stat, y_te_stat, model_type_label="Static", artifact_dir=artifact_dir
    )

    # ── 2. TEMPORAL PIPELINE ─────────────────────────────────────────────────
    X_temporal = matrices['X_temporal']
    preprocessor_temp = build_preprocessor(groups['temporal_numeric_features'], groups['temporal_categorical_features'])
    X_temp_proc = preprocessor_temp.fit_transform(X_temporal)
    temp_feat_names = list(preprocessor_temp.get_feature_names_out())
    X_temp_df = pd.DataFrame(X_temp_proc, columns=temp_feat_names)

    X_temp_smote, y_temp_smote, smote_temp_info = apply_smote_before_split(X_temp_df, y)
    X_tr_temp, X_te_temp, y_tr_temp, y_te_temp = train_test_split(
        X_temp_smote, y_temp_smote, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_temp_smote if y_temp_smote.nunique() > 1 else None
    )

    temporal_comp, temporal_best_row, temporal_best_model = train_and_evaluate_models(
        X_tr_temp, X_te_temp, y_tr_temp, y_te_temp, model_type_label="Temporal", artifact_dir=artifact_dir
    )

    # ── 3. ROC Curve for Best Temporal Model ──────────────────────────────────
    try:
        y_proba_best = temporal_best_model.predict_proba(X_te_temp)[:, 1]
    except Exception:
        y_proba_best = temporal_best_model.predict(X_te_temp)

    fpr, tpr, _ = roc_curve(y_te_temp, y_proba_best)
    roc_data = {'fpr': [round(float(x), 4) for x in fpr], 'tpr': [round(float(x), 4) for x in tpr]}

    # ── 4. Save Artifacts ────────────────────────────────────────────────────
    # Temporal Model Artifact (Primary)
    temp_artifact = {
        'model': temporal_best_model,
        'model_name': temporal_best_row['model_name'],
        'model_type': 'temporal',
        'preprocessor': preprocessor_temp,
        'raw_feature_columns': groups['temporal_all_features'],
        'processed_feature_columns': temp_feat_names,
        'feature_groups': groups,
    }
    joblib.dump(temp_artifact, os.path.join(artifact_dir, 'temporal_model_artifact.joblib'))
    joblib.dump(temporal_best_model, os.path.join(artifact_dir, 'best_model.joblib'))

    # Static Model Artifact (Baseline)
    static_artifact = {
        'model': static_best_model,
        'model_name': static_best_row['model_name'],
        'model_type': 'static',
        'preprocessor': preprocessor_static,
        'raw_feature_columns': groups['static_all_features'],
        'processed_feature_columns': static_feat_names,
        'feature_groups': groups,
    }
    joblib.dump(static_artifact, os.path.join(artifact_dir, 'static_model_artifact.joblib'))

    # Comparison CSV & Metrics
    all_comp_df = pd.DataFrame(temporal_comp + static_comp)
    all_comp_df.to_csv(os.path.join(artifact_dir, 'model_comparison.csv'), index=False)

    static_vs_temporal = {
        'temporal_best': temporal_best_row,
        'static_best': static_best_row,
        'temporal_comparison': temporal_comp,
        'static_comparison': static_comp,
        'accuracy_gain': round(temporal_best_row['accuracy'] - static_best_row['accuracy'], 4),
        'f1_gain': round(temporal_best_row['f1'] - static_best_row['f1'], 4),
        'roc_auc_gain': round((temporal_best_row['roc_auc'] or 0) - (static_best_row['roc_auc'] or 0), 4),
    }

    with open(os.path.join(artifact_dir, 'static_vs_temporal_comparison.json'), 'w') as f:
        json.dump(static_vs_temporal, f, indent=2, default=str)

    with open(os.path.join(artifact_dir, 'model_metrics.json'), 'w') as f:
        json.dump(temporal_best_row, f, indent=2, default=str)

    with open(os.path.join(artifact_dir, 'selected_features.json'), 'w') as f:
        json.dump({
            'selected_features': groups['temporal_all_features'],
            'feature_groups': groups,
            'static_features': groups['static_all_features'],
        }, f, indent=2, default=str)

    split_info = {
        'train_size': len(X_tr_temp),
        'test_size': len(X_te_temp),
        'total': len(X_temp_smote),
        'test_ratio': TEST_SIZE,
    }

    with open(os.path.join(artifact_dir, 'pipeline_info.json'), 'w') as f:
        json.dump({
            'smote': smote_temp_info,
            'split': split_info,
            'best_model': temporal_best_row['model_name'],
            'feature_selection': {'selected': groups['temporal_all_features'], 'k': len(groups['temporal_all_features'])},
        }, f, indent=2, default=str)

    with open(os.path.join(artifact_dir, 'roc_data.json'), 'w') as f:
        json.dump(roc_data, f, indent=2)

    with open(os.path.join(artifact_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(temporal_best_row['confusion_matrix'], f, indent=2)

    print("\nSTATIC VS TEMPORAL MODEL EVALUATION COMPARISON", flush=True)
    print("-----------------------------------------------", flush=True)
    print(f"Static Model Best:   {static_best_row['model_name']} | Accuracy: {static_best_row['accuracy']*100:.2f}% | F1: {static_best_row['f1']*100:.2f}% | ROC-AUC: {static_best_row['roc_auc']}", flush=True)
    print(f"Temporal Model Best: {temporal_best_row['model_name']} | Accuracy: {temporal_best_row['accuracy']*100:.2f}% | F1: {temporal_best_row['f1']*100:.2f}% | ROC-AUC: {temporal_best_row['roc_auc']}", flush=True)
    print(f"Accuracy Gain from Temporal Engineering: {static_vs_temporal['accuracy_gain']*100:+.2f}%", flush=True)

    return {
        'temporal_comparison': temporal_comp,
        'static_comparison': static_comp,
        'temporal_best': temporal_best_row,
        'static_best': static_best_row,
        'static_vs_temporal': static_vs_temporal,
        'smote': smote_temp_info,
        'split': split_info,
        'roc_data': roc_data,
        'confusion_matrix': temporal_best_row['confusion_matrix'],
    }


def compute_temporal_statistics(temporal_df: pd.DataFrame) -> Dict[str, Any]:
    if temporal_df.empty:
        return {
            'total_patients': 0,
            'total_visits': 0,
            'patients_with_multiple_visits': 0,
            'average_visits_per_patient': 0,
            'minimum_visits': 0,
            'maximum_visits': 0,
            'date_range': {'earliest': None, 'latest': None},
        }

    vc = temporal_df.groupby('patient_id').size()
    return {
        'total_patients': int(temporal_df['patient_id'].nunique()),
        'total_visits': int(len(temporal_df)),
        'patients_with_multiple_visits': int((vc > 1).sum()),
        'average_visits_per_patient': round(float(vc.mean()), 2),
        'minimum_visits': int(vc.min()),
        'maximum_visits': int(vc.max()),
        'date_range': {
            'earliest': str(temporal_df['visit_date'].min()),
            'latest': str(temporal_df['visit_date'].max()),
        },
    }


def run_temporal_training_pipeline(dataset_path: str | None = None) -> Dict[str, Any]:
    dataset_path = dataset_path or DATASET_PATH

    print("=" * 60, flush=True)
    print("OBJECTIVE 3: DYNAMIC CVD RISK MODEL (TEMPORAL PATIENT DATA)", flush=True)
    print("=" * 60, flush=True)

    # 1. Dataset Loading & Inspection
    print("\n1. DATASET LOADING\n------------------", flush=True)
    df = load_dataset(dataset_path)
    mapping = analyze_dataset(df, dataset_path)
    print(f"Loaded: {dataset_path} ({len(df)} rows, {len(df.columns)} cols)", flush=True)

    # 2. Data Cleaning
    print("\n2. DATA PREPROCESSING & CLEANING\n---------------------------------", flush=True)
    clean_df, prep_report = clean_dataset(df, mapping)

    # 3. PostgreSQL Ingestion
    print("\n3. POSTGRESQL VISIT INGESTION\n------------------------------", flush=True)
    db_stats = insert_patient_visits(clean_df, mapping)

    # 4. Critical Temporal Feature Engineering
    print("\n4. CRITICAL TEMPORAL FEATURE ENGINEERING\n-----------------------------------------", flush=True)
    visits = load_visits_from_db()
    temporal_df = generate_temporal_features(visits, mapping)
    save_temporal_features(temporal_df)

    # 5. Prepare Feature Matrices & Train Static + Temporal Models
    print("\n5. MODEL TRAINING & COMPARISON (STATIC VS TEMPORAL)\n---------------------------------------------------", flush=True)
    matrices = prepare_feature_matrices(temporal_df, mapping)
    train_results = train_static_and_temporal_pipelines(matrices)

    temporal_stats = compute_temporal_statistics(temporal_df)

    summary = {
        'mapping': asdict(mapping),
        'preprocessing': prep_report,
        'database': db_stats,
        'temporal_stats': temporal_stats,
        'model_features': matrices['groups'],
        **train_results,
    }

    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(MODEL_ARTIFACT_DIR, 'temporal_pipeline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(MODEL_ARTIFACT_DIR, 'dataset_mapping.json'), 'w') as f:
        json.dump(asdict(mapping), f, indent=2, default=str)

    print("\n6. TRAINING COMPLETE & ARTIFACTS PERSISTED\n-------------------------------------------", flush=True)
    print(f"Artifact directory: {MODEL_ARTIFACT_DIR}", flush=True)

    return summary


def load_mapping() -> DatasetMapping:
    map_path = os.path.join(MODEL_ARTIFACT_DIR, 'dataset_mapping.json')
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            return DatasetMapping(**json.load(f))
    df = load_dataset(DATASET_PATH)
    return analyze_dataset(df, DATASET_PATH)


def predict_latest_for_patient(patient_id: str) -> Dict[str, Any]:
    mapping = load_mapping()
    visits = load_visits_from_db(str(patient_id))
    if visits.empty:
        raise ValueError(f"No patient visits found for patient_id: {patient_id}")

    temporal_df = generate_temporal_features(visits, mapping)
    latest_record = temporal_df.sort_values('visit_date').iloc[-1]

    artifact_path = os.path.join(MODEL_ARTIFACT_DIR, 'temporal_model_artifact.joblib')
    if not os.path.exists(artifact_path):
        raise FileNotFoundError("Temporal model artifact not found. Run backend/train.py first.")

    artifact = joblib.load(artifact_path)
    feature_cols = artifact['raw_feature_columns']
    preprocessor = artifact['preprocessor']
    model = artifact['model']

    input_dict = {col: latest_record.get(col, np.nan) for col in feature_cols}
    input_df = pd.DataFrame([input_dict])

    for c in input_df.columns:
        if c in artifact['feature_groups']['temporal_numeric_features']:
            input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
        else:
            input_df[c] = input_df[c].astype(str).fillna('Unknown')

    X_trans = preprocessor.transform(input_df)
    pred = int(model.predict(X_trans)[0])
    try:
        proba = float(model.predict_proba(X_trans)[0][1])
    except Exception:
        proba = float(pred)

    # Separate feature groups for client response
    critical_changes = {}
    critical_rates = {}
    current_values = {}
    previous_values = {}
    other_values = {}

    for k, v in latest_record.to_dict().items():
        if k == 'target':
            continue
        safe_v = _json_safe(v)
        if k.endswith('_change'):
            critical_changes[k] = safe_v
        elif k.endswith('_rate'):
            critical_rates[k] = safe_v
        elif k.startswith('previous_'):
            previous_values[k.replace('previous_', '')] = safe_v
        elif k in mapping.critical_features:
            current_values[k] = safe_v
        elif k in {'days_since_previous_visit', 'has_previous_visit', 'visit_number', 'visit_date', 'previous_visit_date'}:
            pass
        else:
            other_values[k] = safe_v

    timeline = []
    for _, row in temporal_df.iterrows():
        timeline.append({k: _json_safe(v) for k, v in row.to_dict().items() if k != 'target'})

    has_prev_flag = int(latest_record.get('has_previous_visit', 0))

    return {
        'patient_id': str(patient_id),
        'risk_prediction': pred,
        'risk_probability': round(proba, 4),
        'risk_level': 'Higher Cardiovascular Risk' if pred == 1 else 'Lower Cardiovascular Risk',
        'visit_number': int(latest_record.get('visit_number', 1)),
        'number_of_visits': int(len(visits)),
        'has_previous_visit': bool(has_prev_flag == 1),
        'is_first_visit': bool(has_prev_flag == 0),
        'days_since_previous_visit': _json_safe(latest_record.get('days_since_previous_visit')),
        'previous_visit_date': _json_safe(latest_record.get('previous_visit_date')),
        'current_values': current_values,
        'previous_values': previous_values,
        'critical_changes': critical_changes,
        'critical_rates': critical_rates,
        'other_current_values': other_values,
        'temporal_features': {
            **critical_changes,
            **critical_rates,
            'days_since_previous_visit': _json_safe(latest_record.get('days_since_previous_visit')),
            'has_previous_visit': has_prev_flag,
        },
        'timeline': timeline,
    }


def insert_manual_visit(
    patient_id: str,
    clinical_data: Dict[str, Any],
    visit_date: datetime | None = None,
    target: float | None = None
) -> Dict[str, Any]:
    mapping = load_mapping()
    if visit_date is None and 'visit_date' in clinical_data and clinical_data['visit_date']:
        try:
            visit_date = pd.to_datetime(clinical_data['visit_date']).to_pydatetime()
        except Exception:
            visit_date = datetime.utcnow()
    else:
        visit_date = visit_date or datetime.utcnow()

    norm_data = normalize_columns(pd.DataFrame([clinical_data])).iloc[0].to_dict()
    raw_payload = {k: _json_safe(v) for k, v in norm_data.items()}
    raw_payload['patient_id'] = str(patient_id)
    raw_payload['visit_date'] = visit_date.isoformat()

    now = datetime.utcnow()
    with engine.begin() as con:
        p_row = con.execute(text('SELECT id FROM patients WHERE patient_code = :c'), {'c': str(patient_id)}).fetchone()
        if p_row:
            db_pid = p_row[0]
        else:
            con.execute(text('INSERT INTO patients (patient_code, created_at) VALUES (:c, :t)'), {'c': str(patient_id), 't': now})
            db_pid = con.execute(text('SELECT id FROM patients WHERE patient_code = :c'), {'c': str(patient_id)}).fetchone()[0]

        ex = con.execute(
            text('SELECT id FROM patient_visits WHERE source_patient_id = :p AND visit_date = :d'),
            {'p': str(patient_id), 'd': visit_date}
        ).fetchone()

        visit_cols = {
            'patient_id': db_pid,
            'source_patient_id': str(patient_id),
            'visit_timestamp': visit_date,
            'visit_date': visit_date,
            'age': float(norm_data['age']) if 'age' in norm_data and pd.notna(norm_data['age']) else None,
            'gender': str(norm_data['gender']) if 'gender' in norm_data and pd.notna(norm_data['gender']) else None,
            'bmi': float(norm_data['bmi']) if 'bmi' in norm_data and pd.notna(norm_data['bmi']) else None,
            'chest_pain_type': str(norm_data['chest_pain_type']) if 'chest_pain_type' in norm_data and pd.notna(norm_data['chest_pain_type']) else None,
            'systolic_bp': float(norm_data['systolic_bp']) if 'systolic_bp' in norm_data and pd.notna(norm_data['systolic_bp']) else None,
            'diastolic_bp': float(norm_data['diastolic_bp']) if 'diastolic_bp' in norm_data and pd.notna(norm_data['diastolic_bp']) else None,
            'resting_heart_rate': float(norm_data['resting_heart_rate']) if 'resting_heart_rate' in norm_data and pd.notna(norm_data['resting_heart_rate']) else None,
            'max_heart_rate': float(norm_data['max_heart_rate']) if 'max_heart_rate' in norm_data and pd.notna(norm_data['max_heart_rate']) else None,
            'cholesterol': float(norm_data['cholesterol']) if 'cholesterol' in norm_data and pd.notna(norm_data['cholesterol']) else None,
            'hdl': float(norm_data['hdl']) if 'hdl' in norm_data and pd.notna(norm_data['hdl']) else None,
            'ldl': float(norm_data['ldl']) if 'ldl' in norm_data and pd.notna(norm_data['ldl']) else None,
            'fasting_blood_sugar': str(norm_data['fasting_blood_sugar']) if 'fasting_blood_sugar' in norm_data and pd.notna(norm_data['fasting_blood_sugar']) else None,
            'hba1c': float(norm_data['hba1c']) if 'hba1c' in norm_data and pd.notna(norm_data['hba1c']) else None,
            'diabetes': str(norm_data['diabetes']) if 'diabetes' in norm_data and pd.notna(norm_data['diabetes']) else None,
            'resting_ecg': str(norm_data['resting_ecg']) if 'resting_ecg' in norm_data and pd.notna(norm_data['resting_ecg']) else None,
            'exercise_angina': str(norm_data['exercise_angina']) if 'exercise_angina' in norm_data and pd.notna(norm_data['exercise_angina']) else None,
            'oldpeak': float(norm_data['oldpeak']) if 'oldpeak' in norm_data and pd.notna(norm_data['oldpeak']) else None,
            'st_slope': str(norm_data['st_slope']) if 'st_slope' in norm_data and pd.notna(norm_data['st_slope']) else None,
            'num_major_vessels': float(norm_data['num_major_vessels']) if 'num_major_vessels' in norm_data and pd.notna(norm_data['num_major_vessels']) else None,
            'thalassemia': str(norm_data['thalassemia']) if 'thalassemia' in norm_data and pd.notna(norm_data['thalassemia']) else None,
            'smoking': str(norm_data['smoking']) if 'smoking' in norm_data and pd.notna(norm_data['smoking']) else None,
            'family_history': str(norm_data['family_history']) if 'family_history' in norm_data and pd.notna(norm_data['family_history']) else None,
            'physical_activity': str(norm_data['physical_activity']) if 'physical_activity' in norm_data and pd.notna(norm_data['physical_activity']) else None,
            'stress_level': float(norm_data['stress_level']) if 'stress_level' in norm_data and pd.notna(norm_data['stress_level']) else None,
            'target': target,
            'raw_data': json.dumps(raw_payload),
            'updated_at': now,
        }

        if ex:
            set_clause = ", ".join([f"{k} = :{k}" for k in visit_cols.keys() if k not in {'patient_id', 'source_patient_id'}])
            con.execute(text(f'UPDATE patient_visits SET {set_clause} WHERE id = {ex[0]}'), visit_cols)
        else:
            visit_cols['created_at'] = now
            cols_str = ", ".join(visit_cols.keys())
            vals_str = ", ".join([f":{k}" for k in visit_cols.keys()])
            con.execute(text(f'INSERT INTO patient_visits ({cols_str}) VALUES ({vals_str})'), visit_cols)

    # Recompute temporal features for this patient
    patient_visits = load_visits_from_db(str(patient_id))
    patient_temporal = generate_temporal_features(patient_visits, mapping)

    with engine.begin() as con:
        con.execute(text('DELETE FROM temporal_patient_features WHERE patient_id = :p'), {'p': str(patient_id)})

    save_temporal_features(patient_temporal)

    pred_res = predict_latest_for_patient(str(patient_id))

    # Save to predictions table
    with engine.begin() as con:
        con.execute(text('''
            INSERT INTO predictions (patient_code, visit_date, prediction, probability, risk_level, model_name, model_type, created_at)
            VALUES (:p, :d, :pred, :prob, :rl, :m, 'temporal', :c)
        '''), {
            'p': str(patient_id),
            'd': visit_date,
            'pred': pred_res['risk_prediction'],
            'prob': pred_res['risk_probability'],
            'rl': pred_res['risk_level'],
            'm': 'Dynamic Temporal Risk Model',
            'c': now,
        })

    return pred_res
