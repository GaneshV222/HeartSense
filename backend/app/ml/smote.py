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

    Parameters
    ----------
    X : pd.DataFrame – feature matrix
    y : pd.Series    – target vector

    Returns
    -------
    X_resampled : pd.DataFrame
    y_resampled : pd.Series
    info : dict   – contains 'before' and 'after' distributions
    """
    dist_before = get_class_distribution(y)

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    # Keep DataFrame format with column names
    X_resampled = pd.DataFrame(X_res, columns=X.columns)
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
