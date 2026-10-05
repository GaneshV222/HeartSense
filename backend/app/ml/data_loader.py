"""
HeartSense – Data Loader

Loads the Cleveland Heart Disease Dataset from CSV.
Downloads from UCI if the local file is missing.
"""

import os
import urllib.request
import pandas as pd
from app.config import DATASET_PATH, FEATURE_NAMES, TARGET_NAME


UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def download_dataset(dest_path: str = DATASET_PATH) -> str:
    """Download the processed Cleveland dataset from UCI if not present."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"[DataLoader] Downloading Cleveland dataset to {dest_path} …")
        urllib.request.urlretrieve(UCI_URL, dest_path)
        # The UCI file has no header – add one
        cols = FEATURE_NAMES + [TARGET_NAME]
        df = pd.read_csv(dest_path, header=None, names=cols, na_values="?")
        df.to_csv(dest_path, index=False)
        print("[DataLoader] Download complete.")
    return dest_path


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """
    Load the heart-disease CSV and return a DataFrame with the standard
    column names.  If *path* is ``None`` the configured ``DATASET_PATH``
    is used and the file is downloaded automatically when missing.
    """
    path = path or DATASET_PATH
    if not os.path.exists(path):
        download_dataset(path)

    df = pd.read_csv(path, na_values="?")

    # If the file has no header row (raw UCI), assign column names
    expected_cols = FEATURE_NAMES + [TARGET_NAME]
    if list(df.columns) != expected_cols and df.shape[1] == len(expected_cols):
        df.columns = expected_cols

    print(f"[DataLoader] Loaded {len(df)} rows, {df.shape[1]} columns from {path}")
    return df
