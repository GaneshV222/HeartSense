from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.preprocessing import clean_data as _backend_clean_data


def inspect_dataset(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe(include="all"),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _backend_clean_data(df.copy())

    # Legacy compatibility for the older Cleveland-style tests.
    if "target" in df.columns and "prediction" not in df.columns:
        df["prediction"] = df["target"]

    # Ensure no missing values remain in legacy feature columns.
    for col in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
        if col in df.columns and df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)

    # The project policy is to remove exact duplicate records before training.
    df = df.drop_duplicates(ignore_index=True)

    return df


def separate_features_target(df: pd.DataFrame):
    feature_cols = [
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
    X = df[feature_cols].copy()
    y = df["target"].copy() if "target" in df.columns else df["prediction"].copy()
    return X, y
