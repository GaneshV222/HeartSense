"""
HeartSense Configuration Module

Centralizes all application settings using environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent

# Search and load .env from multiple candidate locations
for env_path in [
    BASE_DIR / ".env",
    WORKSPACE_DIR / ".env",
    BASE_DIR.parent.parent / ".env",
    Path.cwd() / ".env",
]:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

# Paths
DATASET_PATH = os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "heart_disease_prediction_2026.csv"))
if not os.path.exists(DATASET_PATH):
    for candidate in [
        BASE_DIR / "data" / "heart_disease_prediction_2026.csv",
        WORKSPACE_DIR / "heart_disease_prediction_2026.csv",
        WORKSPACE_DIR / "backend" / "data" / "heart_disease_prediction_2026.csv",
    ]:
        if candidate.exists():
            DATASET_PATH = str(candidate)
            break

MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", str(BASE_DIR / "artifacts"))
if not os.path.exists(MODEL_ARTIFACT_DIR):
    os.makedirs(MODEL_ARTIFACT_DIR, exist_ok=True)

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:VishnuS%402023@127.0.0.1:5432/heartsense"
)

# ML Pipeline Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SMOTE_RANDOM_STATE = 42
FEATURE_SELECTION_K = "all"

# Target Column in New Dataset
TARGET_NAME = "prediction"

# Baseline 24 Clinical & Lifestyle Features in New Dataset
BASELINE_FEATURE_COLS = [
    "age",
    "gender",
    "bmi",
    "chest_pain_type",
    "systolic_bp",
    "diastolic_bp",
    "resting_heart_rate",
    "max_heart_rate",
    "cholesterol",
    "hdl",
    "ldl",
    "fasting_blood_sugar",
    "hba1c",
    "diabetes",
    "resting_ecg",
    "exercise_angina",
    "oldpeak",
    "st_slope",
    "num_major_vessels",
    "thalassemia",
    "smoking",
    "family_history",
    "physical_activity",
    "stress_level",
]

# Critical Features for Temporal Analysis
CRITICAL_NUMERICAL_FEATURES = [
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "ldl",
    "hdl",
    "bmi",
    "hba1c",
    "resting_heart_rate",
]

CRITICAL_LIFESTYLE_FEATURES = [
    "smoking",
    "physical_activity",
]

ALL_10_TEMPORAL_VARIABLES = CRITICAL_NUMERICAL_FEATURES + CRITICAL_LIFESTYLE_FEATURES

PREVIOUS_VALUE_COLS = [f"prev_{col}" for col in ALL_10_TEMPORAL_VARIABLES]
CHANGE_COLS = [f"{col}_change" for col in CRITICAL_NUMERICAL_FEATURES] + [f"{col}_changed" for col in CRITICAL_LIFESTYLE_FEATURES]
RATE_COLS = [f"{col}_rate" for col in CRITICAL_NUMERICAL_FEATURES]

# All ML Model Feature Columns (Baseline + Previous + Changes + Rates)
ALL_MODEL_FEATURE_COLS = (
    BASELINE_FEATURE_COLS
    + PREVIOUS_VALUE_COLS
    + CHANGE_COLS
    + RATE_COLS
)
