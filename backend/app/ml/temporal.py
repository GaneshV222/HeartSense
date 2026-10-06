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


def compute_temporal_changes(current_values: dict, previous_values: Optional[dict]) -> List[dict]:
    """
    Compute structured changes across the critical cardiovascular variables.
    """
    snapshot = TemporalFeatureService.calculate_snapshot(
        current_data=current_values,
        previous_data=previous_values,
        patient_id=str(current_values.get("patient_id", "Unknown")),
    )

    changes = []
    # Numerical features
    for var in NUMERICAL_VARIABLES:
        details = snapshot["numerical_details"][var]
        curr = details["current"]
        prev = details["previous"]
        change_val = details["change"]
        rate_val = details.get("rate")

        if change_val is not None:
            direction = "Increased" if change_val > 0 else ("Decreased" if change_val < 0 else "Stable")
        else:
            direction = "No Previous Visit"

        changes.append({
            "feature": var,
            "label": details["label"],
            "unit": details["unit"],
            "type": "numerical",
            "previous": prev,
            "current": curr,
            "change": change_val,
            "delta": change_val,
            "rate": rate_val,
            "direction": direction,
        })

    # Lifestyle features
    for var in CATEGORICAL_VARIABLES:
        details = snapshot["categorical_details"][var]
        changed = details["changed"]
        direction = "Changed" if changed else ("Unchanged" if changed is False else "No Previous Visit")

        changes.append({
            "feature": var,
            "label": details["label"],
            "unit": "",
            "type": "categorical",
            "previous": details["previous"],
            "current": details["current"],
            "changed": changed,
            "direction": direction,
        })

    return changes


def summarize_risk_changes(changes: Optional[List[dict]]) -> dict:
    """Summarize changes into worsening vs improving indicators."""
    if not changes:
        return {"concerning": [], "improving": [], "neutral": []}

    concerning = []
    improving = []
    neutral = []

    for c in changes:
        if c.get("change") is None and c.get("changed") is None:
            continue
        feat = c["feature"]
        delta = c.get("change")

        if feat in {"systolic_bp", "diastolic_bp", "cholesterol", "ldl", "bmi", "hba1c", "resting_heart_rate"}:
            if delta is not None:
                if delta > 0:
                    concerning.append(c)
                elif delta < 0:
                    improving.append(c)
                else:
                    neutral.append(c)
        elif feat == "hdl":
            if delta is not None:
                if delta < 0:
                    concerning.append(c)
                elif delta > 0:
                    improving.append(c)
                else:
                    neutral.append(c)
        else:
            neutral.append(c)

    return {
        "concerning": concerning,
        "improving": improving,
        "neutral": neutral,
    }
