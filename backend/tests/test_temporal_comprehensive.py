"""
HeartSense – Comprehensive 12-Point Temporal & Database Test Suite

Validates Section 41 of the Master Codex Specification:
  TEST 1:  First visit (previous = NULL, delta = NULL)
  TEST 2:  Second visit (delta = current - previous)
  TEST 3:  Third visit (previous = second visit, NOT first visit)
  TEST 4:  Two different patients (Patient A never uses Patient B's values)
  TEST 5:  Smoking: No -> Yes, changed = true
  TEST 6:  Smoking: Yes -> Yes, changed = false
  TEST 7:  Physical Activity: High -> Low, changed = true
  TEST 8:  Current = previous, delta = 0
  TEST 9:  Missing previous visit, delta = NULL
  TEST 10: Future visit must not influence an earlier visit (no future leakage)
  TEST 11: Database persistence (Create visit -> commit -> new session -> record exists)
  TEST 12: No SQLite fallback when PostgreSQL is configured
"""

import sys
import os
import uuid
import unittest
from datetime import datetime, timedelta
from sqlalchemy import text

# Add backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
except Exception:
    pass

from app.services.temporal_service import (
    TemporalFeatureService,
    NUMERICAL_VARIABLES,
    CATEGORICAL_VARIABLES,
    ALL_10_TEMPORAL_VARIABLES,
)
from app.database import engine, get_session, verify_db_connection
from app.ml.temporal_pipeline import insert_manual_visit, predict_latest_for_patient
from app.models import PatientTemporalFeature, PatientVisit, Patient


def cleanup_test_patients(patient_codes: list[str]):
    with engine.begin() as con:
        for code in patient_codes:
            con.execute(text("DELETE FROM predictions WHERE patient_code = :p"), {"p": code})
            con.execute(text("DELETE FROM patient_temporal_features WHERE patient_id = :p"), {"p": code})
            con.execute(text("DELETE FROM patient_visits WHERE source_patient_id = :p"), {"p": code})
            con.execute(text("DELETE FROM patients WHERE patient_code = :p"), {"p": code})


class TestTemporal12Points(unittest.TestCase):
    """Explicit verification of TEST 1 to TEST 12."""

    def test_01_first_visit_null_deltas(self):
        """TEST 1: First visit. Expected: previous = NULL, delta = NULL."""
        v1_data = {
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "cholesterol": 190,
            "ldl": 110,
            "hdl": 50,
            "bmi": 24.5,
            "hba1c": 5.4,
            "resting_heart_rate": 72,
            "smoking_status": "No",
            "physical_activity": "Moderate",
        }
        snapshot = TemporalFeatureService.calculate_snapshot(
            current_data=v1_data,
            previous_data=None,
            patient_id="TEST_P1",
        )

        assert snapshot["is_first_visit"] is True
        assert snapshot["has_previous_visit"] is False

        # All 8 previous and deltas must be None
        for var in NUMERICAL_VARIABLES:
            assert snapshot[f"previous_{var}"] is None, f"Expected previous_{var} to be None for first visit"
            assert snapshot[f"delta_{var}"] is None, f"Expected delta_{var} to be None for first visit"

        # Categorical changes must be None
        assert snapshot["previous_smoking_status"] is None
        assert snapshot["smoking_status_changed"] is None
        assert snapshot["previous_physical_activity"] is None
        assert snapshot["physical_activity_changed"] is None

    def test_02_second_visit_exact_delta(self):
        """TEST 2: Second visit. Expected: delta = current - previous."""
        v1 = {"systolic_bp": 120, "cholesterol": 200, "bmi": 24.0, "smoking_status": "No", "physical_activity": "Moderate"}
        v2 = {"systolic_bp": 135, "cholesterol": 225, "bmi": 26.5, "smoking_status": "Yes", "physical_activity": "Low"}

        snapshot = TemporalFeatureService.calculate_snapshot(
            current_data=v2,
            previous_data=v1,
            patient_id="TEST_P2",
        )

        assert snapshot["is_first_visit"] is False
        assert snapshot["has_previous_visit"] is True
        assert snapshot["previous_systolic_bp"] == 120
        assert snapshot["current_systolic_bp"] == 135
        assert snapshot["delta_systolic_bp"] == 15.0  # 135 - 120

        assert snapshot["previous_cholesterol"] == 200
        assert snapshot["current_cholesterol"] == 225
        assert snapshot["delta_cholesterol"] == 25.0  # 225 - 200

        assert snapshot["previous_bmi"] == 24.0
        assert snapshot["current_bmi"] == 26.5
        assert snapshot["delta_bmi"] == 2.5  # 26.5 - 24.0

    def test_03_third_visit_references_second_not_first(self):
        """TEST 3: Third visit. Expected: previous = second visit, NOT first visit."""
        pid = f"TEST_3VISIT_{uuid.uuid4().hex[:6]}"
        cleanup_test_patients([pid])
        try:
            # Visit 1
            insert_manual_visit(pid, {"systolic_bp": 120, "cholesterol": 180}, visit_date=datetime(2025, 1, 10))
            # Visit 2
            insert_manual_visit(pid, {"systolic_bp": 130, "cholesterol": 200}, visit_date=datetime(2025, 3, 10))
            # Visit 3
            res_v3 = insert_manual_visit(pid, {"systolic_bp": 145, "cholesterol": 230}, visit_date=datetime(2025, 6, 10))

            snap = res_v3["temporal_snapshot"]
            assert snap["current_systolic_bp"] == 145
            assert snap["previous_systolic_bp"] == 130, "Previous BP for visit 3 must be Visit 2 (130), NOT Visit 1 (120)"
            assert snap["delta_systolic_bp"] == 15.0  # 145 - 130, NOT 145 - 120

            assert snap["current_cholesterol"] == 230
            assert snap["previous_cholesterol"] == 200, "Previous Cholesterol for visit 3 must be Visit 2 (200), NOT Visit 1 (180)"
            assert snap["delta_cholesterol"] == 30.0  # 230 - 200
        finally:
            cleanup_test_patients([pid])

    def test_04_patient_separation_no_cross_patient_leakage(self):
        """TEST 4: Two different patients. Expected: Patient A never uses Patient B's values."""
        pid_a = f"TEST_PA_{uuid.uuid4().hex[:6]}"
        pid_b = f"TEST_PB_{uuid.uuid4().hex[:6]}"
        cleanup_test_patients([pid_a, pid_b])
        try:
            # Patient A has a visit with BP = 180
            insert_manual_visit(pid_a, {"systolic_bp": 180, "cholesterol": 260}, visit_date=datetime(2025, 1, 1))

            # Patient B has their FIRST visit with BP = 120
            res_b = insert_manual_visit(pid_b, {"systolic_bp": 120, "cholesterol": 190}, visit_date=datetime(2025, 2, 1))

            snap_b = res_b["temporal_snapshot"]
            assert snap_b["is_first_visit"] is True, "Patient B first visit must not reference Patient A"
            assert snap_b["previous_systolic_bp"] is None, "Patient B must have NULL previous_systolic_bp, never Patient A's 180"
            assert snap_b["delta_systolic_bp"] is None, "Patient B must have NULL delta_systolic_bp"
        finally:
            cleanup_test_patients([pid_a, pid_b])

    def test_05_smoking_changed_no_to_yes(self):
        """TEST 5: Smoking: No -> Yes, changed = true."""
        v1 = {"smoking_status": "No"}
        v2 = {"smoking_status": "Yes"}
        snap = TemporalFeatureService.calculate_snapshot(v2, v1, "P_SMK_1")
        assert snap["previous_smoking_status"] == "No"
        assert snap["current_smoking_status"] == "Yes"
        assert snap["smoking_status_changed"] is True

    def test_06_smoking_unchanged_yes_to_yes(self):
        """TEST 6: Smoking: Yes -> Yes, changed = false."""
        v1 = {"smoking_status": "Yes"}
        v2 = {"smoking_status": "Yes"}
        snap = TemporalFeatureService.calculate_snapshot(v2, v1, "P_SMK_2")
        assert snap["previous_smoking_status"] == "Yes"
        assert snap["current_smoking_status"] == "Yes"
        assert snap["smoking_status_changed"] is False

    def test_07_physical_activity_changed_high_to_low(self):
        """TEST 7: Physical Activity: High -> Low, changed = true."""
        v1 = {"physical_activity": "High"}
        v2 = {"physical_activity": "Low"}
        snap = TemporalFeatureService.calculate_snapshot(v2, v1, "P_ACT_1")
        assert snap["previous_physical_activity"] == "High"
        assert snap["current_physical_activity"] == "Low"
        assert snap["physical_activity_changed"] is True

    def test_08_current_equals_previous_delta_zero(self):
        """TEST 8: Current = previous. Expected: delta = 0."""
        v1 = {"systolic_bp": 120, "cholesterol": 200}
        v2 = {"systolic_bp": 120, "cholesterol": 200}
        snap = TemporalFeatureService.calculate_snapshot(v2, v1, "P_ZERO")
        assert snap["delta_systolic_bp"] == 0.0
        assert snap["delta_cholesterol"] == 0.0

    def test_09_missing_previous_visit_is_null_not_zero(self):
        """TEST 9: Missing previous visit. Expected: delta = NULL (not zero!)."""
        v1 = {"systolic_bp": 130}
        snap = TemporalFeatureService.calculate_snapshot(v1, None, "P_NULL")
        assert snap["delta_systolic_bp"] is None
        assert snap["delta_systolic_bp"] != 0.0, "Missing previous visit must be None/NULL, NEVER 0.0"

    def test_10_no_future_leakage(self):
        """TEST 10: Future visit must not influence an earlier visit."""
        pid = f"TEST_LEAK_{uuid.uuid4().hex[:6]}"
        cleanup_test_patients([pid])
        try:
            # First visit in Jan
            res_v1 = insert_manual_visit(pid, {"systolic_bp": 120}, visit_date=datetime(2025, 1, 1))
            # Later visit in July with higher BP
            res_v2 = insert_manual_visit(pid, {"systolic_bp": 160}, visit_date=datetime(2025, 7, 1))

            # Query historical visit 1 snapshot directly from database
            session = get_session()
            try:
                timeline = TemporalFeatureService.get_patient_temporal_timeline(session, pid)
                assert len(timeline) == 2
                v1_snap = timeline[0]["snapshot"]
                assert v1_snap["current_systolic_bp"] == 120
                assert v1_snap["previous_systolic_bp"] is None, "Earlier visit 1 must not know about visit 2"
                assert v1_snap["delta_systolic_bp"] is None
            finally:
                session.close()
        finally:
            cleanup_test_patients([pid])

    def test_11_database_persistence_across_sessions(self):
        """TEST 11: Database persistence. Create visit -> reconnect -> temporal record still exists."""
        pid = f"TEST_PERSIST_{uuid.uuid4().hex[:6]}"
        cleanup_test_patients([pid])
        try:
            insert_manual_visit(pid, {"systolic_bp": 135, "cholesterol": 210}, visit_date=datetime(2025, 2, 1))

            # Open a completely fresh session (simulating backend restart)
            fresh_session = get_session()
            try:
                ptf_row = fresh_session.execute(
                    text("SELECT * FROM patient_temporal_features WHERE patient_id = :p"),
                    {"p": pid}
                ).mappings().first()

                assert ptf_row is not None, "Temporal record must be persisted in PostgreSQL table"
                assert ptf_row["patient_id"] == pid
                assert ptf_row["current_systolic_bp"] == 135
                assert ptf_row["current_cholesterol"] == 210
            finally:
                fresh_session.close()
        finally:
            cleanup_test_patients([pid])

    def test_12_no_sqlite_fallback_postgresql_verified(self):
        """TEST 12: No SQLite fallback when PostgreSQL is configured."""
        dialect = engine.dialect.name
        assert dialect == "postgresql", f"Production database must be PostgreSQL, found: {dialect}"
        assert verify_db_connection() is True, "PostgreSQL connection must be active and verified"


if __name__ == "__main__":
    unittest.main(verbosity=2)
