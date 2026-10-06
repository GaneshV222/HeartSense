"""
HeartSense – Data Loader

Loads the cardiovascular dataset from CSV / Excel.
"""

import os
from pathlib import Path
import pandas as pd
from app.config import DATASET_PATH


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """
    Load the cardiovascular dataset and return a clean DataFrame.
    """
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

    # Normalize columns
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    print(f"[DataLoader] Loaded {len(df)} rows, {df.shape[1]} columns from {path}")
    return df
