"""
HeartSense Configuration Module

Centralizes all application settings using environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "dataset.csv"))
MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", str(BASE_DIR / "artifacts"))

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cvd_prediction.db")

# ── ML Pipeline ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SMOTE_RANDOM_STATE = 42
FEATURE_SELECTION_K = 10  # number of top features to select (or "all")

# ── Cleveland Heart Disease Dataset Feature Schema ─────────────────────────────
# These are the 13 clinical features from the UCI Cleveland dataset.
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_NAME = "target"

# Human-readable labels for the Streamlit UI
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)",
    "restecg": "Resting ECG Results (0-2)",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=Yes, 0=No)",
    "oldpeak": "ST Depression (oldpeak)",
    "slope": "Slope of Peak Exercise ST (0-2)",
    "ca": "Number of Major Vessels (0-3)",
    "thal": "Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)",
}

# Numerical features (used for temporal comparison)
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical features
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# ── Model Configuration ───────────────────────────────────────────────────────
# The 6 models to train, compare, and evaluate.
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "SVM",
    "KNN",
    "Naive Bayes",
]

# Best-model selection priority:
#   1. ROC-AUC   (primary)
#   2. Recall    (secondary – detecting CVD is critical)
#   3. F1-Score  (tertiary)
BEST_MODEL_CRITERIA = ["roc_auc", "recall", "f1"]
