"""
HeartSense – Data Preprocessing

Handles:
 • Missing-value imputation
 • Data-type conversion
 • Invalid-value cleaning
 • Feature / target separation
 • Consistent feature ordering

All transformations are deterministic and can be applied to both the
training batch and to single-patient prediction inputs.
"""

import numpy as np
import pandas as pd
from app.config import FEATURE_NAMES, TARGET_NAME, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict useful for the Streamlit dashboard."""
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
    Clean the raw Cleveland Heart Disease dataframe.

    Steps
    -----
    1. Replace ``?`` with ``NaN`` (already handled at load, but defensive).
    2. Cast all columns to numeric where possible.
    3. Drop duplicate rows.
    4. For the target column, binarize: values > 0 → 1 (disease present).
    5. Impute missing numerical values with column median.
    6. Impute missing categorical values with column mode.
    7. Clip numerical values to plausible clinical ranges.
    """
    df = df.copy()

    # 1. Replace stray '?' strings
    df = df.replace("?", np.nan)

    # 2. Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Drop exact duplicates
    df = df.drop_duplicates()

    # 4. Binarize target (Cleveland uses 0-4; we need 0/1)
    if TARGET_NAME in df.columns:
        df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)

    # 5 & 6. Impute missing values
    for col in df.columns:
        if df[col].isnull().any():
            if col in NUMERICAL_FEATURES:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)

    # 7. Clip to plausible clinical ranges (safety net)
    clip_ranges = {
        "age": (1, 120),
        "trestbps": (50, 250),
        "chol": (50, 600),
        "thalach": (50, 250),
        "oldpeak": (0, 10),
        "ca": (0, 3),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    return df


def separate_features_target(df: pd.DataFrame):
    """
    Split a cleaned dataframe into X (features) and y (target).

    Returns
    -------
    X : pd.DataFrame   – feature columns in FEATURE_NAMES order
    y : pd.Series       – target column
    """
    # Ensure consistent column ordering
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[available_features].copy()
    y = df[TARGET_NAME].copy() if TARGET_NAME in df.columns else None
    return X, y


def preprocess_single_patient(patient_dict: dict, selected_features: list[str]) -> pd.DataFrame:
    """
    Prepare a single patient's data for prediction.

    Parameters
    ----------
    patient_dict : dict
        Keys are feature names, values are the raw input values.
    selected_features : list[str]
        The features the trained model expects (after feature selection).

    Returns
    -------
    pd.DataFrame with one row in the correct column order.
    """
    row = {}
    for feat in FEATURE_NAMES:
        val = patient_dict.get(feat, 0)
        row[feat] = pd.to_numeric(val, errors="coerce")

    df = pd.DataFrame([row])

    # Apply same clipping as training
    clip_ranges = {
        "age": (1, 120),
        "trestbps": (50, 250),
        "chol": (50, 600),
        "thalach": (50, 250),
        "oldpeak": (0, 10),
        "ca": (0, 3),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Select only the features used by the model
    df = df[[f for f in selected_features if f in df.columns]]
    return df
