"""
HeartSense – Data Preprocessing

Handles:
  • Missing-value imputation
  • Data-type conversion
  • Clean clinical ranges
  • Feature / target separation
  • Consistent feature ordering
"""

import numpy as np
import pandas as pd
from app.config import (
    BASELINE_FEATURE_COLS,
    TARGET_NAME,
    CRITICAL_NUMERICAL_FEATURES,
    CRITICAL_LIFESTYLE_FEATURES,
    ALL_MODEL_FEATURE_COLS,
)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict of dataset characteristics."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe(),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the cardiovascular dataset according to the new schema.
    """
    df = df.copy()

    # Replace stray empty strings and missing symbols
    df = df.replace({"?": np.nan, "": np.nan, "NA": np.nan, "null": np.nan, "None": np.nan})

    # Ensure target column is 0/1 integer
    target_col = "prediction" if "prediction" in df.columns else ("target" if "target" in df.columns else None)
    if target_col and target_col in df.columns:
        df["prediction"] = (pd.to_numeric(df[target_col], errors="coerce").fillna(0) > 0).astype(int)

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Normal"
                df[col] = df[col].fillna(mode_val)

    # Plausible clinical range clipping
    clip_ranges = {
        "age": (1, 120),
        "systolic_bp": (50, 250),
        "diastolic_bp": (30, 150),
        "cholesterol": (50, 600),
        "ldl": (20, 400),
        "hdl": (10, 150),
        "bmi": (10, 70),
        "resting_heart_rate": (30, 220),
        "max_heart_rate": (40, 250),
        "hba1c": (3, 20),
        "oldpeak": (0, 10),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    return df


def separate_features_target(df: pd.DataFrame):
    """
    Split a cleaned dataframe into X (features) and y (target).
    """
    available_features = [f for f in ALL_MODEL_FEATURE_COLS if f in df.columns]
    if not available_features:
        available_features = [f for f in BASELINE_FEATURE_COLS if f in df.columns]

    X = df[available_features].copy()
    y = df["prediction"].copy() if "prediction" in df.columns else (df["target"].copy() if "target" in df.columns else None)
    return X, y


def preprocess_single_patient(patient_dict: dict, selected_features: list[str]) -> pd.DataFrame:
    """
    Prepare a single patient's data dictionary for model inference.
    """
    row = {}
    for feat in selected_features:
        row[feat] = patient_dict.get(feat, np.nan)

    df = pd.DataFrame([row])
    return df
