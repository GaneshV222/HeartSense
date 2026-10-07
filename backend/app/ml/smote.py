"""
HeartSense – SMOTE Module

Applies Synthetic Minority Oversampling Technique to balance classes.

CRITICAL: SMOTE is applied BEFORE the train-test split, as specified
by the project pipeline:

    Preprocessing → SMOTE → Feature Selection → Train-Test Split
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from app.config import SMOTE_RANDOM_STATE


def get_class_distribution(y) -> dict:
    """Return a dict mapping class label → count."""
    if isinstance(y, pd.Series):
        return y.value_counts().to_dict()
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = SMOTE_RANDOM_STATE):
    """
    Apply SMOTE to balance the dataset.

    Converts non-numeric columns to a numeric one-hot representation before oversampling,
    so categorical feature sets remain compatible with sklearn's SMOTE pipeline.
    """
    dist_before = get_class_distribution(y)

    X_prepared = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if isinstance(X_prepared, pd.DataFrame):
        non_numeric = list(X_prepared.select_dtypes(exclude=[np.number]).columns)
        if non_numeric:
            X_prepared = pd.get_dummies(X_prepared, columns=non_numeric, dtype=float)
        X_prepared = X_prepared.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_prepared, y)

    X_resampled = pd.DataFrame(X_res, columns=X_prepared.columns)
    y_resampled = pd.Series(y_res, name=y.name)

    dist_after = get_class_distribution(y_resampled)

    info = {
        "before": dist_before,
        "after": dist_after,
        "samples_before": len(y),
        "samples_after": len(y_resampled),
    }

    print(f"[SMOTE] Before: {dist_before}  ->  After: {dist_after}")
    return X_resampled, y_resampled, info
