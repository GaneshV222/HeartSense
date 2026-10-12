"""
HeartSense – Temporal Module

Exposes the authoritative TemporalFeatureService and helper functions
for computing and inspecting temporal clinical changes.
"""

from typing import Any, Dict, List, Optional
from app.services.temporal_service import (
    TemporalFeatureService,
    NUMERICAL_VARIABLES,
    CATEGORICAL_VARIABLES,
    VARIABLE_METADATA,
)


def _feature_name_aliases() -> Dict[str, str]:
    """Allow the legacy tests to query the canonical temporal fields by their dataset aliases."""
    return {
        "trestbps": "systolic_bp",
        "chol": "cholesterol",
        "thalach": "resting_heart_rate",
        "age": "age",
        "oldpeak": "oldpeak",
        "bmi": "bmi",
        "hdl": "hdl",
        "ldl": "ldl",
        "hba1c": "hba1c",
    }


def compute_temporal_changes(current_values: dict, previous_values: Optional[dict]) -> List[dict]:
    """Compatibility wrapper that accepts old Cleveland-style field names and modern canonical names."""
    if current_values is None:
        return []
    if previous_values is None:
        previous_values = {}

    alias_map = {
        "age": ["age"],
        "sex": ["sex", "gender"],
        "cp": ["cp", "chest_pain_type"],
        "trestbps": ["trestbps", "systolic_bp"],
        "chol": ["chol", "cholesterol"],
        "fbs": ["fbs", "fasting_blood_sugar"],
        "restecg": ["restecg", "resting_ecg"],
        "thalach": ["thalach", "max_heart_rate", "resting_heart_rate"],
        "exang": ["exang", "exercise_angina"],
        "oldpeak": ["oldpeak"],
        "slope": ["slope", "st_slope"],
        "ca": ["ca", "num_major_vessels"],
        "thal": ["thal", "thalassemia"],
    }

    def resolve_value(values: dict, feature: str):
        for candidate in alias_map.get(feature, [feature]):
            if candidate in values:
                return values[candidate]
        return None

    changes: List[dict] = []
    for feature in [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal",
    ]:
        current_val = resolve_value(current_values, feature)
        previous_val = resolve_value(previous_values, feature)
        if current_val is None or previous_val is None:
            continue

        if feature in {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}:
            change_val = int(current_val != previous_val)
            direction = "Changed" if change_val else "Stable"
        else:
            change_val = current_val - previous_val
            direction = "Increased" if change_val > 0 else ("Decreased" if change_val < 0 else "Stable")

        changes.append({
            "feature": feature,
            "label": feature,
            "previous": previous_val,
            "current": current_val,
            "change": change_val,
            "direction": direction,
            "canonical_feature": feature,
        })

    return changes


def summarize_risk_changes(changes: Optional[List[dict]]) -> dict:
    """Summarize changes into worsening vs improving indicators with legacy-compatible keys."""
    if not changes:
        return {"concerning": [], "improving": [], "neutral": [], "stable": [], "total_changes": 0}

    concerning = []
    improving = []
    neutral = []
    stable = []

    for c in changes:
        delta = c.get("change")
        if delta is None and c.get("changed") is None:
            continue
        feat = c.get("feature", "")
        if isinstance(delta, (int, float)):
            if delta > 0:
                if feat in {"trestbps", "chol", "thalach"}:
                    if feat == "thalach":
                        concerning.append(c)
                    else:
                        concerning.append(c)
                elif feat == "thalach":
                    concerning.append(c)
                else:
                    neutral.append(c)
            elif delta < 0:
                if feat == "trestbps":
                    improving.append(c)
                elif feat == "thalach":
                    concerning.append(c)
                else:
                    improving.append(c)
            else:
                stable.append(c)
                neutral.append(c)
        else:
            if c.get("direction") == "Stable":
                stable.append(c)
                neutral.append(c)
            else:
                neutral.append(c)

    summary = {
        "concerning": concerning,
        "improving": improving,
        "neutral": neutral,
        "stable": stable,
        "total_changes": len(changes),
    }
    return summary
