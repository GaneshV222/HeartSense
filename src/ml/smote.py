from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE


def get_class_distribution(y) -> dict:
    if isinstance(y, pd.Series):
        return y.value_counts().to_dict()
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    dist_before = get_class_distribution(y)
    X_prepared = X.copy()
    X_prepared = X_prepared.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_prepared, y)
    X_resampled = pd.DataFrame(X_res, columns=X_prepared.columns)
    y_resampled = pd.Series(y_res, name=y.name)
    info = {
        "before": dist_before,
        "after": get_class_distribution(y_resampled),
        "samples_before": len(y),
        "samples_after": len(y_resampled),
    }
    return X_resampled, y_resampled, info
