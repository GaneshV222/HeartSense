"""
HeartSense – Temporal Feature Extraction

Compares a patient's current visit with their most recent previous visit
to calculate changes and determine direction for each numerical feature.
"""

import pandas as pd
from app.config import FEATURE_NAMES, FEATURE_LABELS, NUMERICAL_FEATURES


def compute_temporal_changes(current_values: dict, previous_values: dict) -> list[dict]:
    """
    Compare current vs. previous clinical values.

    Parameters
    ----------
    current_values  : dict  – feature_name → value for the current visit
    previous_values : dict  – feature_name → value for the previous visit

    Returns
    -------
    list[dict] with keys: feature, label, previous, current, change, direction
    """
    changes = []
    for feat in FEATURE_NAMES:
        if feat not in current_values or feat not in previous_values:
            continue

        curr = float(current_values[feat])
        prev = float(previous_values[feat])
        diff = round(curr - prev, 2)

        if diff > 0:
            direction = "Increased"
        elif diff < 0:
            direction = "Decreased"
        else:
            direction = "Stable"

        changes.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "previous": prev,
            "current": curr,
            "change": diff,
            "direction": direction,
        })
    return changes


def summarize_risk_changes(changes: list[dict]) -> dict:
    """
    Produce a human-readable summary of concerning / improving changes.

    Concerning increases:  trestbps ↑, chol ↑, oldpeak ↑, fbs ↑
    Concerning decreases:  thalach ↓
    """
    concerning_up = {"trestbps", "chol", "oldpeak", "fbs"}
    concerning_down = {"thalach"}

    concerning = []
    improving = []
    stable = []

    for c in changes:
        feat = c["feature"]
        d = c["direction"]
        if d == "Stable":
            stable.append(c)
        elif (feat in concerning_up and d == "Increased") or \
             (feat in concerning_down and d == "Decreased"):
            concerning.append(c)
        elif (feat in concerning_up and d == "Decreased") or \
             (feat in concerning_down and d == "Increased"):
            improving.append(c)
        else:
            # Generic features – just note the change
            stable.append(c)

    return {
        "concerning": concerning,
        "improving": improving,
        "stable": stable,
        "total_changes": len(changes),
    }
