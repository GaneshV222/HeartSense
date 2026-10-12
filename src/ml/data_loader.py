from __future__ import annotations

import pandas as pd

from backend.app.ml.data_loader import load_dataset as _backend_load_dataset


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _map_string_values(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map({str(k).strip().lower(): v for k, v in mapping.items()}).fillna(
        pd.to_numeric(series, errors="coerce")
    )


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the canonical CSV and normalize it to the legacy Cleveland feature schema."""
    df = _backend_load_dataset(path).copy()

    # Canonical naming used by the current application.
    if "gender" in df.columns and "sex" not in df.columns:
        df["sex"] = df["gender"].map({
            "Male": 1,
            "male": 1,
            "MALE": 1,
            "Female": 0,
            "female": 0,
            "FEMALE": 0,
        }).fillna(pd.to_numeric(df["gender"].replace({"Male": 1, "Female": 0, "male": 1, "female": 0}), errors="coerce"))

    if "chest_pain_type" in df.columns and "cp" not in df.columns:
        cp_map = {
            "Typical Angina": 0,
            "typical angina": 0,
            "Typical angina": 0,
            "Atypical Angina": 1,
            "atypical angina": 1,
            "Non-anginal Pain": 2,
            "Non Anginal Pain": 2,
            "non anginal pain": 2,
            "Asymptomatic": 3,
            "asymptomatic": 3,
        }
        df["cp"] = df["chest_pain_type"].map(cp_map).fillna(
            pd.to_numeric(df["chest_pain_type"], errors="coerce")
        )

    if "systolic_bp" in df.columns and "trestbps" not in df.columns:
        df["trestbps"] = _coerce_numeric(df["systolic_bp"])

    if "cholesterol" in df.columns and "chol" not in df.columns:
        df["chol"] = _coerce_numeric(df["cholesterol"])

    if "fasting_blood_sugar" in df.columns and "fbs" not in df.columns:
        df["fbs"] = df["fasting_blood_sugar"].map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}).fillna(pd.to_numeric(df["fasting_blood_sugar"], errors="coerce").clip(0, 1))

    if "resting_ecg" in df.columns and "restecg" not in df.columns:
        restecg_map = {
            "Normal": 0,
            "normal": 0,
            "NORMAL": 0,
            "ST-T Wave Abnormality": 1,
            "ST-T wave abnormality": 1,
            "Left Ventricular Hypertrophy": 2,
            "left ventricular hypertrophy": 2,
        }
        df["restecg"] = df["resting_ecg"].map(restecg_map).fillna(
            pd.to_numeric(df["resting_ecg"], errors="coerce")
        )

    if "max_heart_rate" in df.columns and "thalach" not in df.columns:
        df["thalach"] = _coerce_numeric(df["max_heart_rate"])

    if "exercise_angina" in df.columns and "exang" not in df.columns:
        df["exang"] = df["exercise_angina"].map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}).fillna(pd.to_numeric(df["exercise_angina"], errors="coerce").clip(0, 1))

    if "st_slope" in df.columns and "slope" not in df.columns:
        slope_map = {"Up": 0, "up": 0, "Flat": 1, "flat": 1, "Down": 2, "down": 2}
        df["slope"] = df["st_slope"].map(slope_map).fillna(pd.to_numeric(df["st_slope"], errors="coerce"))

    if "num_major_vessels" in df.columns and "ca" not in df.columns:
        df["ca"] = _coerce_numeric(df["num_major_vessels"])

    if "thalassemia" in df.columns and "thal" not in df.columns:
        thal_map = {
            "Normal": 1,
            "normal": 1,
            "Reversible Defect": 3,
            "Reversible defect": 3,
            "Fixed Defect": 2,
            "Fixed defect": 2,
        }
        df["thal"] = df["thalassemia"].map(thal_map).fillna(pd.to_numeric(df["thalassemia"], errors="coerce"))

    if "prediction" in df.columns and "target" not in df.columns:
        df["target"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(0).astype(int)

    # Some test fixtures and legacy code expect the classic Cleveland feature set
    # and a binary target column to be present on the DataFrame itself.
    for col_name in [
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
        "target",
    ]:
        if col_name in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
            if col_name not in df.columns:
                df[col_name] = pd.NA

    legacy_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    selected = [col for col in legacy_columns if col in df.columns]
    return df[selected].copy()
