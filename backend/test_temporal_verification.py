"""
Comprehensive End-to-End Verification Test for Temporal Patient Risk Assessment
"""
import sys
import os
import uuid
from datetime import datetime

# Adjust path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml.temporal_pipeline import (
    predict_latest_for_patient,
    insert_manual_visit,
    load_visits_from_db,
    generate_temporal_features,
    engine,
)
from sqlalchemy import text
from fastapi.testclient import TestClient
from app.main import app

def cleanup_test_data(patient_codes):
    with engine.begin() as con:
        for code in patient_codes:
            con.execute(text("DELETE FROM temporal_patient_features WHERE patient_id = :p"), {"p": code})
            con.execute(text("DELETE FROM patient_visits WHERE source_patient_id = :p"), {"p": code})
            con.execute(text("DELETE FROM patients WHERE patient_code = :p"), {"p": code})

def run_tests():
    print("==================================================")
    print("STARTING TEMPORAL PATIENT PIPELINE VERIFICATION")
    print("==================================================")

    uid = uuid.uuid4().hex[:6]
    test_pid = f"TEST_LONG_{uid}"
    new_pid = f"TEST_NEW_{uid}"
    cleanup_test_data([test_pid, new_pid])

    # 1. Test Existing Patient from initial dataset
    print("\n[TEST 1] Testing Existing Patient Lookup (Patient ID: 45263)...")
    res_existing = predict_latest_for_patient("45263")
    assert res_existing.get("risk_prediction") in (0, 1), f"Existing patient prediction failed: {res_existing}"
    print(f"  Patient ID: {res_existing['patient_id']}")
    print(f"  Risk Prediction: {res_existing['risk_prediction']} (Prob: {res_existing.get('risk_probability')}%)")
    print(f"  Visit Number: {res_existing['visit_number']}")
    print(f"  Total Visits in DB: {res_existing['number_of_visits']}")
    print("  [PASSED] Test 1!")

    # 2. Test Longitudinal Patient - Visit 1 (Baseline)
    print(f"\n[TEST 2] Inserting Visit 1 (Baseline) for Patient {test_pid}...")
    v1_data = {
        "age": 52,
        "gender": "Male",
        "chest_pain_type": "Typical Angina",
        "resting_heart_rate": 72,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "cholesterol": 190,
        "fasting_blood_sugar": "False",
        "resting_ecg": "Normal",
        "max_heart_rate": 160,
        "exercise_angina": 0,
        "oldpeak": 0.5,
        "st_slope": "Upsloping",
        "num_major_vessels": 0,
        "thalassemia": "Normal",
        "bmi": 24.5,
        "smoking": 0,
        "diabetes": 0,
        "physical_activity": "Moderate",
        "family_history": 0,
        "stress_level": 3,
        "hdl": 55,
        "ldl": 110,
        "hba1c": 5.4,
        "visit_date": "2025-01-15T09:00:00"
    }
    res_v1 = insert_manual_visit(test_pid, v1_data)
    print(f"  Visit 1 Number: {res_v1['visit_number']}")
    print(f"  Risk Prediction: {res_v1['risk_prediction']} (Prob: {res_v1['risk_probability']}%)")
    assert res_v1["visit_number"] == 1, f"Visit 1 should have visit_number == 1, got {res_v1['visit_number']}"
    assert res_v1["is_first_visit"] is True, "First visit must have is_first_visit == True"
    print("  [PASSED] Test 2 Passed!")

    # 3. Test Longitudinal Patient - Visit 2 (Follow-up with Delta & Trend)
    print(f"\n[TEST 3] Inserting Visit 2 (Follow-up + Deterioration) for Patient {test_pid}...")
    v2_data = {
        "age": 53,
        "gender": "Male",
        "chest_pain_type": "Atypical Angina",
        "resting_heart_rate": 84,
        "systolic_bp": 150,    # +30 delta
        "diastolic_bp": 95,     # +15 delta
        "cholesterol": 245,     # +55 delta
        "fasting_blood_sugar": "True",
        "resting_ecg": "ST-T wave abnormality",
        "max_heart_rate": 140,
        "exercise_angina": 1,
        "oldpeak": 1.8,
        "st_slope": "Flat",
        "num_major_vessels": 1,
        "thalassemia": "Reversible Defect",
        "bmi": 28.2,            # +3.7 delta
        "smoking": 1,           # changed from 0 to 1
        "diabetes": 1,
        "physical_activity": "Low",
        "family_history": 0,
        "stress_level": 8,
        "hdl": 40,
        "ldl": 160,
        "hba1c": 6.8,
        "visit_date": "2025-07-20T10:30:00"
    }
    res_v2 = insert_manual_visit(test_pid, v2_data)
    print(f"  Visit 2 Number: {res_v2['visit_number']}")
    print(f"  Risk Prediction: {res_v2['risk_prediction']} (Prob: {res_v2['risk_probability']}%)")
    
    tf = res_v2["temporal_features"]
    print(f"  Delta Systolic BP: {tf.get('delta_systolic_bp')} mmHg (Current: {res_v2.get('current_values', {}).get('systolic_bp')}, Previous: {res_v2.get('previous_values', {}).get('systolic_bp')})")
    print(f"  Delta Cholesterol: {tf.get('delta_cholesterol')} mg/dL")
    print(f"  Delta BMI: {tf.get('delta_bmi')}")
    print(f"  Days Since Previous Visit: {tf.get('days_since_previous_visit')} days")
    print(f"  Trend Info: {res_v2.get('trend_information')}")
    
    # Assertions for temporal accuracy
    assert tf.get("delta_systolic_bp") == 30.0, f"Expected delta_systolic_bp 30.0, got {tf.get('delta_systolic_bp')}"
    assert tf.get("delta_cholesterol") == 55.0, f"Expected delta_cholesterol 55.0, got {tf.get('delta_cholesterol')}"
    assert tf.get("days_since_previous_visit") > 0, "Days since previous visit must be positive"
    assert res_v2["is_first_visit"] is False, "Second visit must have is_first_visit == False"
    print("  [PASSED] Test 3 Passed! Temporal features calculated accurately.")

    # 4. Test FastAPI HTTP Endpoints
    print("\n[TEST 4] Testing FastAPI Endpoints via TestClient...")
    client = TestClient(app)

    # 4a. Health endpoint
    resp = client.get("/api/health")
    print(f"  GET /api/health -> Status {resp.status_code}")
    assert resp.status_code == 200

    # 4b. Predict Existing Patient (Workflow A)
    resp = client.post("/api/assessment/predict", json={"patient_id": test_pid})
    print(f"  POST /api/assessment/predict (Workflow A) -> Status {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_code"] == test_pid
    assert "temporal_features" in data
    assert "timeline" in data
    assert len(data["timeline"]) == 2
    print(f"    Returned {len(data['timeline'])} visits in timeline for {test_pid}")

    # 4c. Predict New Patient / Visit (Workflow B)
    resp = client.post("/api/assessment/predict", json={
        "patient_id": new_pid,
        "patient_name": "Jane Test",
        "age": 45,
        "gender": "Female",
        "systolic_bp": 118,
        "diastolic_bp": 78,
        "cholesterol": 180,
        "bmi": 22.0
    })
    print(f"  POST /api/assessment/predict (Workflow B) -> Status {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_code"] == new_pid
    assert data["visit_number"] == 1
    assert data["is_first_visit"] is True

    # 4d. Patient Temporal Profile Endpoint
    resp = client.get(f"/api/patients/{test_pid}/temporal")
    print(f"  GET /api/patients/{test_pid}/temporal -> Status {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert "timeline" in data
    assert "trend_information" in data

    # 4e. Analytics Endpoint
    resp = client.get("/api/analytics")
    print(f"  GET /api/analytics -> Status {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_comparison" in data
    assert "best_model" in data
    assert "temporal_statistics" in data
    assert "smote_analysis" in data
    print(f"    Best Model: {data['best_model']['name']} (Accuracy: {data['best_model']['accuracy'] * 100:.2f}%)")
    print(f"    Total Patients in DB: {data['temporal_statistics']['total_patients']}")
    print(f"    Total Visits in DB: {data['temporal_statistics']['total_visits']}")
    print(f"    SMOTE Original Records: {data['smote_analysis']['records_before']}")
    print(f"    SMOTE Balanced Records: {data['smote_analysis']['records_after']}")
    print("  [PASSED] Test 4 Passed!")

    # Cleanup test data
    cleanup_test_data([test_pid, new_pid])

    print("\n==================================================")
    print("ALL 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
