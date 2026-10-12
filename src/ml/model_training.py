"""Legacy compatibility layer for the project test suite.

This repository keeps the canonical backend under backend/app, but older code and
compatibility tests import the model training API from the root-level src package.
The legacy contract expects the six-model registry used by the historical test suite,
including XGBoost.
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:  # compatibility support for environments with XGBoost installed
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - fallback when xgboost is unavailable
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

from src.config import RANDOM_STATE


def get_base_models() -> Dict[str, object]:
    """Return the legacy six-model registry expected by the test suite."""
    return {
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
        ),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=RANDOM_STATE, probability=True, class_weight="balanced")),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
        "Naive Bayes": GaussianNB(),
    }
