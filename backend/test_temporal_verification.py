"""
Comprehensive Verification and Acceptance Test for Temporal Cardiovascular Risk Prediction
Tests the exact 4-visit progression for P001, database records, calculations, API responses, and analytics.
"""

import sys
import os
import uuid
from datetime import datetime, timedelta

# Adjust path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml.temporal_pipeline import (
    predict_latest_for_patient,
    insert_manual_visit,
)
from app.database import engine, get_session
from app.models import Patient, PatientVisit, TemporalPatientData
from sqlalchemy import text
from fastapi.testclient import TestClient
from app.main import app


def cleanup_test_data(patient_ids: list[str]):
    with engine.begin() as con:
        for pid in patient_ids:
            con.execute(text("DELETE FROM predictions WHERE patient_id = :p"), {"p": pid})
            con.execute(text("DELETE FROM temporal_patient_data WHERE patient_id = :p"), {"p": pid})
            con.execute(text("DELETE FROM patient_visits WHERE patient_id = :p"), {"p": pid})
            con.execute(text("DELETE FROM patients WHERE patient_id = :p"), {"p": pid})


def run_acceptance_tests():
    print("=" * 70)
    print("STARTING TEMPORAL PATIENT ACCEPTANCE TESTS (P001 V1 -> V2 -> V3 -> V4)")
    print("=" * 70)

    uid = uuid.uuid4().hex[:6]
    test_pid = f"P_TEST_{uid}"
    cleanup_test_data([test_pid])

    session = get_session()

    try:
        # -------------------------------------------------------------
        # 1. VISIT 1: Baseline Visit (First Visit)
        # -------------------------------------------------------------
        print(f"\n[STEP 1] Adding Visit 1 for Patient {test_pid}...")
        v1_date = datetime(2025, 1, 15, 10, 0, 0)
        v1_data = {
            "age": 55,
            "gender": "Male",
            "bmi": 26.5,
            "chest_pain_type": "Typical Angina",
            "systolic_bp": 130.0,
            "diastolic_bp": 85.0,
            "resting_heart_rate": 72.0,
            "max_heart_rate": 150.0,
            "cholesterol": 210.0,
            "hdl": 50.0,
            "ldl": 130.0,
            "fasting_blood_sugar": "False",
            "hba1c": 5.6,
            "diabetes": "False",
            "resting_ecg": "Normal",
            "exercise_angina": "False",
            "oldpeak": 0.8,
            "st_slope": "Flat",
            "num_major_vessels": 0,
            "thalassemia": "Normal",
            "smoking": "False",
            "family_history": "False",
            "physical_activity": "Moderate",
            "stress_level": 4.0,
        }
        res_v1 = insert_manual_visit(test_pid, v1_data, visit_date=v1_date)

        # Verify DB state after Visit 1
        session.expire_all()
        p_count = session.query(Patient).filter_by(patient_id=test_pid).count()
        v_count = session.query(PatientVisit).filter_by(patient_id=test_pid).count()
        t_records = session.query(TemporalPatientData).filter_by(patient_id=test_pid).all()

        assert p_count == 1, f"Expected 1 patient in patients, got {p_count}"
        assert v_count == 1, f"Expected 1 visit in patient_visits, got {v_count}"
        assert len(t_records) == 1, f"Expected 1 row in temporal_patient_data, got {len(t_records)}"

        t1 = t_records[0]
        assert t1.previous_visit_id is None, f"Visit 1 previous_visit_id must be None, got {t1.previous_visit_id}"
        assert t1.previous_systolic_bp is None, f"Visit 1 previous_systolic_bp must be None, got {t1.previous_systolic_bp}"
        assert t1.systolic_bp_change is None, f"Visit 1 systolic_bp_change must be None, got {t1.systolic_bp_change}"
        assert t1.current_systolic_bp == 130.0, f"Expected current_systolic_bp 130.0, got {t1.current_systolic_bp}"
        assert res_v1["is_first_visit"] is True, "is_first_visit must be True for visit 1"
        print("  [OK] [PASSED] Visit 1: patients = 1, patient_visits = 1, temporal_patient_data = NULL previous + V1 current")

        # -------------------------------------------------------------
        # 2. VISIT 2: Second Visit (Calculate V2 - V1)
        # -------------------------------------------------------------
        print(f"\n[STEP 2] Adding Visit 2 for Patient {test_pid}...")
        v2_date = datetime(2025, 6, 15, 10, 0, 0)
        v2_data = {
            **v1_data,
            "age": 55,
            "bmi": 27.5,              # +1.0
            "systolic_bp": 142.0,     # +12.0
            "diastolic_bp": 90.0,     # +5.0
            "cholesterol": 235.0,     # +25.0
            "ldl": 148.0,             # +18.0
            "hdl": 46.0,              # -4.0
            "hba1c": 6.1,             # +0.5
            "resting_heart_rate": 78.0, # +6.0
            "smoking": "True",        # Changed from False to True
            "physical_activity": "Low", # Changed from Moderate to Low
        }
        res_v2 = insert_manual_visit(test_pid, v2_data, visit_date=v2_date)

        # Verify DB state after Visit 2
        session.expire_all()
        p_count = session.query(Patient).filter_by(patient_id=test_pid).count()
        v_count = session.query(PatientVisit).filter_by(patient_id=test_pid).count()
        t_records = session.query(TemporalPatientData).filter_by(patient_id=test_pid).all()

        assert p_count == 1, f"Expected 1 patient in patients, got {p_count}"
        assert v_count == 2, f"Expected 2 visits in patient_visits, got {v_count}"
        assert len(t_records) == 1, f"Expected exactly 1 row in temporal_patient_data, got {len(t_records)}"

        t2 = t_records[0]
        assert t2.previous_visit_id is not None, "Visit 2 previous_visit_id must NOT be None"
        assert t2.previous_systolic_bp == 130.0, f"Expected previous_systolic_bp 130.0, got {t2.previous_systolic_bp}"
        assert t2.current_systolic_bp == 142.0, f"Expected current_systolic_bp 142.0, got {t2.current_systolic_bp}"
        assert t2.systolic_bp_change == 12.0, f"Expected systolic_bp_change 12.0 (142 - 130), got {t2.systolic_bp_change}"
        assert t2.cholesterol_change == 25.0, f"Expected cholesterol_change 25.0 (235 - 210), got {t2.cholesterol_change}"
        assert t2.bmi_change == 1.0, f"Expected bmi_change 1.0 (27.5 - 26.5), got {t2.bmi_change}"
        assert t2.smoking_changed is True, "Smoking should be changed"
        assert t2.physical_activity_changed is True, "Physical activity should be changed"
        assert res_v2["is_first_visit"] is False, "is_first_visit must be False for visit 2"
        print("  [OK] [PASSED] Visit 2: patient_visits = V1,V2; temporal_patient_data = V1,V2; change = V2 - V1 (+12 BP, +25 Chol)")

        # -------------------------------------------------------------
        # 3. VISIT 3: Third Visit (Calculate V3 - V2)
        # -------------------------------------------------------------
        print(f"\n[STEP 3] Adding Visit 3 for Patient {test_pid}...")
        v3_date = datetime(2026, 1, 15, 10, 0, 0)
        v3_data = {
            **v2_data,
            "age": 56,
            "bmi": 28.0,              # V3 - V2 = 28.0 - 27.5 = +0.5
            "systolic_bp": 155.0,     # V3 - V2 = 155.0 - 142.0 = +13.0
            "diastolic_bp": 96.0,     # V3 - V2 = 96.0 - 90.0 = +6.0
            "cholesterol": 250.0,     # V3 - V2 = 250.0 - 235.0 = +15.0
            "ldl": 160.0,             # V3 - V2 = 160.0 - 148.0 = +12.0
            "hdl": 42.0,              # V3 - V2 = 42.0 - 46.0 = -4.0
            "hba1c": 6.5,             # V3 - V2 = 6.5 - 6.1 = +0.4
            "resting_heart_rate": 82.0, # V3 - V2 = 82.0 - 78.0 = +4.0
            "smoking": "True",        # V3 == V2 (Unchanged)
            "physical_activity": "Low", # V3 == V2 (Unchanged)
        }
        res_v3 = insert_manual_visit(test_pid, v3_data, visit_date=v3_date)

        # Verify DB state after Visit 3
        session.expire_all()
        p_count = session.query(Patient).filter_by(patient_id=test_pid).count()
        v_count = session.query(PatientVisit).filter_by(patient_id=test_pid).count()
        t_records = session.query(TemporalPatientData).filter_by(patient_id=test_pid).all()

        assert p_count == 1, f"Expected 1 patient in patients, got {p_count}"
        assert v_count == 3, f"Expected 3 visits in patient_visits, got {v_count}"
        assert len(t_records) == 1, f"Expected exactly 1 row in temporal_patient_data, got {len(t_records)}"

        t3 = t_records[0]
        assert t3.previous_systolic_bp == 142.0, f"Expected previous_systolic_bp 142.0 (V2), got {t3.previous_systolic_bp}"
        assert t3.current_systolic_bp == 155.0, f"Expected current_systolic_bp 155.0 (V3), got {t3.current_systolic_bp}"
        assert t3.systolic_bp_change == 13.0, f"Expected systolic_bp_change 13.0 (V3 - V2: 155 - 142), got {t3.systolic_bp_change}"
        assert t3.cholesterol_change == 15.0, f"Expected cholesterol_change 15.0 (V3 - V2: 250 - 235), got {t3.cholesterol_change}"
        assert t3.bmi_change == 0.5, f"Expected bmi_change 0.5 (V3 - V2: 28.0 - 27.5), got {t3.bmi_change}"
        assert t3.smoking_changed is False, "Smoking should be unchanged between V2 and V3"
        print("  [OK] [PASSED] Visit 3: patient_visits = V1,V2,V3; temporal_patient_data = V2,V3; change = V3 - V2 (+13 BP, +15 Chol)")

        # -------------------------------------------------------------
        # 4. VISIT 4: Fourth Visit (Calculate V4 - V3)
        # -------------------------------------------------------------
        print(f"\n[STEP 4] Adding Visit 4 for Patient {test_pid}...")
        v4_date = datetime(2026, 10, 15, 10, 0, 0)
        v4_data = {
            **v3_data,
            "age": 57,
            "bmi": 26.0,              # V4 - V3 = 26.0 - 28.0 = -2.0 (Improvement)
            "systolic_bp": 128.0,     # V4 - V3 = 128.0 - 155.0 = -27.0 (Improvement)
            "diastolic_bp": 82.0,     # V4 - V3 = 82.0 - 96.0 = -14.0
            "cholesterol": 195.0,     # V4 - V3 = 195.0 - 250.0 = -55.0 (Improvement)
            "ldl": 115.0,             # V4 - V3 = 115.0 - 160.0 = -45.0
            "hdl": 52.0,              # V4 - V3 = 52.0 - 42.0 = +10.0
            "hba1c": 5.5,             # V4 - V3 = 5.5 - 6.5 = -1.0
            "resting_heart_rate": 70.0, # V4 - V3 = 70.0 - 82.0 = -12.0
            "smoking": "False",       # Changed from True to False (Quit smoking)
            "physical_activity": "High", # Changed from Low to High
        }
        res_v4 = insert_manual_visit(test_pid, v4_data, visit_date=v4_date)

        # Verify DB state after Visit 4
        session.expire_all()
        p_count = session.query(Patient).filter_by(patient_id=test_pid).count()
        v_count = session.query(PatientVisit).filter_by(patient_id=test_pid).count()
        t_records = session.query(TemporalPatientData).filter_by(patient_id=test_pid).all()

        assert p_count == 1, f"Expected 1 patient in patients, got {p_count}"
        assert v_count == 4, f"Expected 4 visits in patient_visits, got {v_count}"
        assert len(t_records) == 1, f"Expected exactly 1 row in temporal_patient_data, got {len(t_records)}"

        t4 = t_records[0]
        assert t4.previous_systolic_bp == 155.0, f"Expected previous_systolic_bp 155.0 (V3), got {t4.previous_systolic_bp}"
        assert t4.current_systolic_bp == 128.0, f"Expected current_systolic_bp 128.0 (V4), got {t4.current_systolic_bp}"
        assert t4.systolic_bp_change == -27.0, f"Expected systolic_bp_change -27.0 (V4 - V3: 128 - 155), got {t4.systolic_bp_change}"
        assert t4.cholesterol_change == -55.0, f"Expected cholesterol_change -55.0 (V4 - V3: 195 - 250), got {t4.cholesterol_change}"
        assert t4.bmi_change == -2.0, f"Expected bmi_change -2.0 (V4 - V3: 26.0 - 28.0), got {t4.bmi_change}"
        assert t4.smoking_changed is True, "Smoking should be changed (Quit smoking)"
        assert t4.physical_activity_changed is True, "Physical activity should be changed"
        print("  [OK] [PASSED] Visit 4: patient_visits = V1,V2,V3,V4; temporal_patient_data = V3,V4; change = V4 - V3 (-27 BP, -55 Chol)")

        # -------------------------------------------------------------
        # 5. TEST FASTAPI CLIENT ENDPOINTS
        # -------------------------------------------------------------
        print("\n[STEP 5] Testing FastAPI Endpoints for complete workflow...")
        client = TestClient(app)

        # 5a. Patient visits API
        resp = client.get(f"/api/patients/{test_pid}/visits")
        assert resp.status_code == 200, f"Failed visits API: {resp.text}"
        vis_resp = resp.json()
        assert vis_resp["total_visits"] == 4, f"Expected 4 visits, got {vis_resp['total_visits']}"
        print(f"  [OK] GET /api/patients/{test_pid}/visits returned {vis_resp['total_visits']} timeline visits")

        # 5b. Patient temporal profile API
        resp = client.get(f"/api/patients/{test_pid}/temporal")
        assert resp.status_code == 200, f"Failed temporal API: {resp.text}"
        prof_resp = resp.json()
        assert prof_resp["visit_number"] == 4
        assert prof_resp["current_values"]["systolic_bp"] == 128.0
        assert prof_resp["previous_values"]["systolic_bp"] == 155.0
        print(f"  [OK] GET /api/patients/{test_pid}/temporal returned prediction {prof_resp['risk_prediction']} (Category: {prof_resp['risk_category']})")

        # 5c. Analytics API
        resp = client.get("/api/analytics")
        assert resp.status_code == 200, f"Failed analytics API: {resp.text}"
        ana = resp.json()
        assert "model_comparison" in ana
        assert "best_model" in ana
        assert "temporal_statistics" in ana
        print(f"  [OK] GET /api/analytics returned Best Model: {ana['best_model']['name']} (Acc: {ana['best_model']['accuracy']*100:.2f}%)")

        print("\n" + "=" * 70)
        print("ALL ACCEPTANCE TESTS COMPLETED AND FULLY PASSED!")
        print("=" * 70)

    finally:
        cleanup_test_data([test_pid])
        session.close()


if __name__ == "__main__":
    run_acceptance_tests()
