"""
Tests for temporal feature extraction and dynamic risk assessment.
"""

import pytest
from src.ml.temporal import compute_temporal_changes, summarize_risk_changes


class TestTemporalChanges:
    """Tests for temporal feature comparison."""

    def test_compute_changes_basic(self):
        current = {"age": 56, "trestbps": 145, "chol": 225, "thalach": 140, "oldpeak": 2.5}
        previous = {"age": 55, "trestbps": 120, "chol": 200, "thalach": 160, "oldpeak": 1.0}

        changes = compute_temporal_changes(current, previous)
        assert len(changes) > 0

        # Check trestbps
        bp_change = next(c for c in changes if c["feature"] == "trestbps")
        assert bp_change["previous"] == 120
        assert bp_change["current"] == 145
        assert bp_change["change"] == 25
        assert bp_change["direction"] == "Increased"

    def test_direction_decreased(self):
        current = {"thalach": 120}
        previous = {"thalach": 160}

        changes = compute_temporal_changes(current, previous)
        hr_change = next(c for c in changes if c["feature"] == "thalach")
        assert hr_change["direction"] == "Decreased"
        assert hr_change["change"] == -40

    def test_direction_stable(self):
        current = {"chol": 200}
        previous = {"chol": 200}

        changes = compute_temporal_changes(current, previous)
        chol_change = next(c for c in changes if c["feature"] == "chol")
        assert chol_change["direction"] == "Stable"
        assert chol_change["change"] == 0

    def test_missing_feature_skipped(self):
        current = {"trestbps": 130}
        previous = {"chol": 200}  # different features

        changes = compute_temporal_changes(current, previous)
        # Neither should appear since they don't overlap properly
        for c in changes:
            assert c["feature"] not in ["trestbps", "chol"] or \
                   (c["feature"] in current and c["feature"] in previous)


class TestRiskSummary:
    """Tests for risk change summarization."""

    def test_concerning_bp_increase(self):
        changes = [
            {"feature": "trestbps", "label": "BP", "previous": 120,
             "current": 160, "change": 40, "direction": "Increased"},
        ]
        summary = summarize_risk_changes(changes)
        assert len(summary["concerning"]) == 1
        assert summary["concerning"][0]["feature"] == "trestbps"

    def test_concerning_thalach_decrease(self):
        changes = [
            {"feature": "thalach", "label": "Heart Rate", "previous": 170,
             "current": 120, "change": -50, "direction": "Decreased"},
        ]
        summary = summarize_risk_changes(changes)
        assert len(summary["concerning"]) == 1

    def test_improving_bp_decrease(self):
        changes = [
            {"feature": "trestbps", "label": "BP", "previous": 160,
             "current": 120, "change": -40, "direction": "Decreased"},
        ]
        summary = summarize_risk_changes(changes)
        assert len(summary["improving"]) == 1

    def test_stable_classified_correctly(self):
        changes = [
            {"feature": "chol", "label": "Cholesterol", "previous": 200,
             "current": 200, "change": 0, "direction": "Stable"},
        ]
        summary = summarize_risk_changes(changes)
        assert len(summary["stable"]) == 1

    def test_chronological_ordering(self):
        """Verify that temporal comparison works for multiple features."""
        current = {
            "age": 56, "sex": 1, "cp": 2, "trestbps": 150, "chol": 250,
            "fbs": 1, "restecg": 1, "thalach": 130, "exang": 1,
            "oldpeak": 3.0, "slope": 1, "ca": 2, "thal": 2,
        }
        previous = {
            "age": 55, "sex": 1, "cp": 2, "trestbps": 120, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 1, "thal": 1,
        }
        changes = compute_temporal_changes(current, previous)
        summary = summarize_risk_changes(changes)

        # There should be several concerning changes
        assert summary["total_changes"] == 13
        assert len(summary["concerning"]) > 0
