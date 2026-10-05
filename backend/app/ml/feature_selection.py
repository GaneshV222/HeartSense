"""
HeartSense – Feature Selection

Uses SelectKBest with mutual_info_classif to rank and select the most
informative features.

Pipeline position:  Preprocessing → SMOTE → **Feature Selection** → Train-Test Split
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from app.config import FEATURE_SELECTION_K, MODEL_ARTIFACT_DIR


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int | str = FEATURE_SELECTION_K,
):
    """
    Select the top-*k* features using mutual information.

    Parameters
    ----------
    X : pd.DataFrame  – feature matrix (post-SMOTE)
    y : pd.Series     – target vector
    k : int | str     – number of features to keep, or ``"all"``

    Returns
    -------
    X_selected : pd.DataFrame  – reduced feature matrix
    selector   : SelectKBest   – fitted selector (for later transform)
    info       : dict          – original, selected, and removed features + scores
    """
    if k == "all" or k >= X.shape[1]:
        k = X.shape[1]

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)

    mask = selector.get_support()
    selected = list(X.columns[mask])
    removed = list(X.columns[~mask])

    scores = dict(zip(X.columns, selector.scores_))

    X_selected = X[selected].copy()

    info = {
        "original_features": list(X.columns),
        "selected_features": selected,
        "removed_features": removed,
        "feature_scores": {k: round(float(v), 4) for k, v in scores.items()},
        "k": k,
    }

    print(f"[FeatureSelection] Kept {len(selected)}/{X.shape[1]} features: {selected}")
    return X_selected, selector, info


def save_feature_artifacts(selector, info: dict, artifact_dir: str = MODEL_ARTIFACT_DIR):
    """Persist the fitted selector and metadata."""
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(selector, os.path.join(artifact_dir, "feature_selector.joblib"))
    with open(os.path.join(artifact_dir, "selected_features.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[FeatureSelection] Artifacts saved to {artifact_dir}")


def load_selected_features(artifact_dir: str = MODEL_ARTIFACT_DIR) -> list[str]:
    """Load the list of selected feature names from saved JSON."""
    path = os.path.join(artifact_dir, "selected_features.json")
    with open(path, "r") as f:
        info = json.load(f)
    return info["selected_features"]


def load_feature_selector(artifact_dir: str = MODEL_ARTIFACT_DIR):
    """Load the fitted SelectKBest object."""
    return joblib.load(os.path.join(artifact_dir, "feature_selector.joblib"))
