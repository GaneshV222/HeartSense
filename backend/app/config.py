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
    "postgresql+psycopg2://postgres:VishnuS%402023@localhost:5432/heartsense"
)

# ML Pipeline
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SMOTE_RANDOM_STATE = 42
FEATURE_SELECTION_K = "all"
