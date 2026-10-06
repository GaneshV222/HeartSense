"""
HeartSense – Temporal Module

Exposes the authoritative TemporalFeatureService and helper functions
for computing and inspecting temporal clinical changes.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from app.services.temporal_service import (
    TemporalFeatureService,
    NUMERICAL_VARIABLES,
    CATEGORICAL_VARIABLES,
    ALL_10_TEMPORAL_VARIABLES,
    NUMERICAL_DELTA_FIELDS,
    CATEGORICAL_CHANGED_FIELDS,
    VARIABLE_METADATA,
)


def compute_temporal_changes(current_values: dict, previous_values: Optional[dict]) -> List[dict]:
    """
    Compute structured changes across the 10 core temporal variables.
    """
    snapshot = TemporalFeatureService.calculate_snapshot(
        current_data=current_values,
        previous_data=previous_values,
        patient_id=str(current_values.get("patient_id", "Unknown")),
    )

    changes = []
    # 8 Numerical
    for var in NUMERICAL_VARIABLES:
        details = snapshot["numerical_details"][var]
        curr = details["current"]
        prev = details["previous"]
        delta = details["delta"]

        if delta is not None:
            direction = "Increased" if delta > 0 else ("Decreased" if delta < 0 else "Stable")
        else:
            direction = "No Previous Visit"

        changes.append({
            "feature": var,
            "label": details["label"],
            "unit": details["unit"],
            "type": "numerical",
            "previous": prev,
            "current": curr,
            "change": delta,
            "delta": delta,
            "direction": direction,
        })

    # 2 Categorical
    for var in CATEGORICAL_VARIABLES:
        details = snapshot["categorical_details"][var]
        changes.append({
            "feature": var,
            "label": details["label"],
            "unit": "",
            "type": "categorical",
            "previous": details["previous"],
            "current": details["current"],
            "changed": details["changed"],
            "direction": "Changed" if details["changed"] else ("Unchanged" if details["changed"] is False else "No Previous Visit"),
        })

    return changes
