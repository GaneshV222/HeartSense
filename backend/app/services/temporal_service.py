"""
HeartSense – TemporalFeatureService

The single authoritative temporal feature calculation and persistence service.
Encapsulates all logic for the 10 core temporal clinical variables:
  - 8 Numerical: Systolic BP, Diastolic BP, Cholesterol, LDL, HDL, BMI, HbA1c, Resting Heart Rate (current, previous, delta = current - previous)
  - 2 Categorical: Smoking Status, Physical Activity (current, previous, changed boolean flag)

Rules:
  - Previous values MUST come from the immediately preceding chronological visit of the SAME patient_id.
  - If a patient has only 1 visit (first visit): previous = NULL, delta = NULL, changed = NULL/false.
  - Missing previous visit is NEVER treated as delta = 0.
  - Historical snapshots are never overwritten; every visit produces a new temporal snapshot.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

# Exact 10 temporal variables
NUMERICAL_VARIABLES = [
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "ldl",
    "hdl",
    "bmi",
    "hba1c",
    "resting_heart_rate",
]

CATEGORICAL_VARIABLES = [
    "smoking_status",
    "physical_activity",
]

ALL_10_TEMPORAL_VARIABLES = NUMERICAL_VARIABLES + CATEGORICAL_VARIABLES

NUMERICAL_DELTA_FIELDS = [f"delta_{var}" for var in NUMERICAL_VARIABLES]
CATEGORICAL_CHANGED_FIELDS = [f"{var}_changed" for var in CATEGORICAL_VARIABLES]

VARIABLE_METADATA = {
    "systolic_bp": {"label": "Systolic Blood Pressure", "unit": "mmHg", "type": "numerical"},
    "diastolic_bp": {"label": "Diastolic Blood Pressure", "unit": "mmHg", "type": "numerical"},
    "cholesterol": {"label": "Total Cholesterol", "unit": "mg/dL", "type": "numerical"},
    "ldl": {"label": "LDL Cholesterol", "unit": "mg/dL", "type": "numerical"},
    "hdl": {"label": "HDL Cholesterol", "unit": "mg/dL", "type": "numerical"},
    "bmi": {"label": "Body Mass Index (BMI)", "unit": "kg/m²", "type": "numerical"},
    "hba1c": {"label": "Glycated Hemoglobin (HbA1c)", "unit": "%", "type": "numerical"},
    "resting_heart_rate": {"label": "Resting Heart Rate", "unit": "bpm", "type": "numerical"},
    "smoking_status": {"label": "Smoking Status", "unit": "", "type": "categorical"},
    "physical_activity": {"label": "Physical Activity Level", "unit": "", "type": "categorical"},
}


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f  # check for NaN
    except (ValueError, TypeError):
        return None


def _safe_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"none", "null", "nan", ""}:
        return None
    return s


def normalize_smoking_value(val: Any) -> Optional[str]:
    """Normalize smoking inputs like 1, 0, 'True', 'False', 'Yes', 'No'."""
    s = _safe_str(val)
    if s is None:
        return None
    low = s.lower()
    if low in {"1", "true", "yes", "smoker"}:
        return "Yes"
    if low in {"0", "false", "no", "non-smoker"}:
        return "No"
    return s.capitalize()


class TemporalFeatureService:
    """
    Authoritative service for computing, storing, and retrieving longitudinal temporal features.
    """

    @staticmethod
    def calculate_snapshot(
        current_data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]],
        patient_id: str,
        visit_id: Optional[int] = None,
        assessment_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the exact 10 temporal clinical features given a current visit and
        an optional immediately preceding visit for the SAME patient.
        """
        assessment_date = assessment_date or datetime.utcnow()
        is_first_visit = (previous_data is None or len(previous_data) == 0)

        snapshot: Dict[str, Any] = {
            "patient_id": str(patient_id),
            "visit_id": visit_id,
            "assessment_date": assessment_date,
            "is_first_visit": is_first_visit,
            "has_previous_visit": not is_first_visit,
        }

        # ── 1. Calculate 8 Numerical Deltas ──────────────────────────────────
        numerical_details: Dict[str, Dict[str, Optional[float]]] = {}
        for var in NUMERICAL_VARIABLES:
            curr_raw = current_data.get(var)
            curr_val = _safe_float(curr_raw)
            snapshot[f"current_{var}"] = curr_val

            if is_first_visit or previous_data is None:
                prev_val = None
                delta_val = None
            else:
                prev_raw = previous_data.get(var)
                prev_val = _safe_float(prev_raw)
                if curr_val is not None and prev_val is not None:
                    delta_val = round(curr_val - prev_val, 2)
                else:
                    delta_val = None

            snapshot[f"previous_{var}"] = prev_val
            snapshot[f"delta_{var}"] = delta_val

            numerical_details[var] = {
                "previous": prev_val,
                "current": curr_val,
                "delta": delta_val,
                "unit": VARIABLE_METADATA[var]["unit"],
                "label": VARIABLE_METADATA[var]["label"],
            }

        # ── 2. Calculate 2 Categorical Changes ────────────────────────────────
        categorical_details: Dict[str, Dict[str, Any]] = {}

        # 9. Smoking Status
        curr_smoke_raw = current_data.get("smoking_status") or current_data.get("smoking")
        curr_smoke = normalize_smoking_value(curr_smoke_raw)
        snapshot["current_smoking_status"] = curr_smoke

        if is_first_visit or previous_data is None:
            prev_smoke = None
            smoke_changed = None
        else:
            prev_smoke_raw = previous_data.get("smoking_status") or previous_data.get("smoking")
            prev_smoke = normalize_smoking_value(prev_smoke_raw)
            if curr_smoke is not None and prev_smoke is not None:
                smoke_changed = bool(curr_smoke.lower() != prev_smoke.lower())
            else:
                smoke_changed = None

        snapshot["previous_smoking_status"] = prev_smoke
        snapshot["smoking_status_changed"] = smoke_changed
        categorical_details["smoking_status"] = {
            "previous": prev_smoke,
            "current": curr_smoke,
            "changed": smoke_changed,
            "label": VARIABLE_METADATA["smoking_status"]["label"],
        }

        # 10. Physical Activity
        curr_act = _safe_str(current_data.get("physical_activity"))
        if curr_act:
            curr_act = curr_act.capitalize()
        snapshot["current_physical_activity"] = curr_act

        if is_first_visit or previous_data is None:
            prev_act = None
            act_changed = None
        else:
            prev_act = _safe_str(previous_data.get("physical_activity"))
            if prev_act:
                prev_act = prev_act.capitalize()
            if curr_act is not None and prev_act is not None:
                act_changed = bool(curr_act.lower() != prev_act.lower())
            else:
                act_changed = None

        snapshot["previous_physical_activity"] = prev_act
        snapshot["physical_activity_changed"] = act_changed
        categorical_details["physical_activity"] = {
            "previous": prev_act,
            "current": curr_act,
            "changed": act_changed,
            "label": VARIABLE_METADATA["physical_activity"]["label"],
        }

        snapshot["numerical_details"] = numerical_details
        snapshot["categorical_details"] = categorical_details

        return snapshot

    @staticmethod
    def get_immediately_previous_visit(
        session: Session,
        patient_id: str,
        before_date: Optional[datetime] = None,
        before_visit_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the immediately preceding visit for the EXACT SAME patient_id.
        ORDER BY visit_date DESC, id DESC LIMIT 1.
        Never compares across different patients.
        """
        query = """
            SELECT id, source_patient_id, visit_date, visit_timestamp,
                   age, gender, bmi, chest_pain_type, systolic_bp, diastolic_bp,
                   resting_heart_rate, max_heart_rate, cholesterol, hdl, ldl,
                   fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                   oldpeak, st_slope, num_major_vessels, thalassemia, smoking,
                   smoking_status, family_history, physical_activity, stress_level, target
            FROM patient_visits
            WHERE source_patient_id = :pid
        """
        params: Dict[str, Any] = {"pid": str(patient_id)}

        if before_date:
            query += " AND visit_date < :bdate"
            params["bdate"] = before_date
        elif before_visit_id:
            query += " AND id < :bvid"
            params["bvid"] = before_visit_id

        query += " ORDER BY visit_date DESC, id DESC LIMIT 1"

        row = session.execute(text(query), params).mappings().first()
        if row:
            return dict(row)
        return None

    @staticmethod
    def persist_temporal_snapshot(
        session: Session,
        snapshot: Dict[str, Any]
    ) -> int:
        """
        Persist temporal snapshot into patient_temporal_features table.
        Safe insert: does NOT delete or overwrite previous snapshots.
        """
        insert_query = text("""
            INSERT INTO patient_temporal_features (
                patient_id, visit_id, assessment_date,
                previous_systolic_bp, current_systolic_bp, delta_systolic_bp,
                previous_diastolic_bp, current_diastolic_bp, delta_diastolic_bp,
                previous_cholesterol, current_cholesterol, delta_cholesterol,
                previous_ldl, current_ldl, delta_ldl,
                previous_hdl, current_hdl, delta_hdl,
                previous_bmi, current_bmi, delta_bmi,
                previous_hba1c, current_hba1c, delta_hba1c,
                previous_resting_heart_rate, current_resting_heart_rate, delta_resting_heart_rate,
                previous_smoking_status, current_smoking_status, smoking_status_changed,
                previous_physical_activity, current_physical_activity, physical_activity_changed,
                created_at, updated_at
            )
            VALUES (
                :patient_id, :visit_id, :assessment_date,
                :previous_systolic_bp, :current_systolic_bp, :delta_systolic_bp,
                :previous_diastolic_bp, :current_diastolic_bp, :delta_diastolic_bp,
                :previous_cholesterol, :current_cholesterol, :delta_cholesterol,
                :previous_ldl, :current_ldl, :delta_ldl,
                :previous_hdl, :current_hdl, :delta_hdl,
                :previous_bmi, :current_bmi, :delta_bmi,
                :previous_hba1c, :current_hba1c, :delta_hba1c,
                :previous_resting_heart_rate, :current_resting_heart_rate, :delta_resting_heart_rate,
                :previous_smoking_status, :current_smoking_status, :smoking_status_changed,
                :previous_physical_activity, :current_physical_activity, :physical_activity_changed,
                :created_at, :updated_at
            )
            RETURNING id
        """)

        now = datetime.utcnow()
        params = {
            "patient_id": str(snapshot["patient_id"]),
            "visit_id": snapshot.get("visit_id"),
            "assessment_date": snapshot.get("assessment_date") or now,
            "previous_systolic_bp": snapshot.get("previous_systolic_bp"),
            "current_systolic_bp": snapshot.get("current_systolic_bp"),
            "delta_systolic_bp": snapshot.get("delta_systolic_bp"),
            "previous_diastolic_bp": snapshot.get("previous_diastolic_bp"),
            "current_diastolic_bp": snapshot.get("current_diastolic_bp"),
            "delta_diastolic_bp": snapshot.get("delta_diastolic_bp"),
            "previous_cholesterol": snapshot.get("previous_cholesterol"),
            "current_cholesterol": snapshot.get("current_cholesterol"),
            "delta_cholesterol": snapshot.get("delta_cholesterol"),
            "previous_ldl": snapshot.get("previous_ldl"),
            "current_ldl": snapshot.get("current_ldl"),
            "delta_ldl": snapshot.get("delta_ldl"),
            "previous_hdl": snapshot.get("previous_hdl"),
            "current_hdl": snapshot.get("current_hdl"),
            "delta_hdl": snapshot.get("delta_hdl"),
            "previous_bmi": snapshot.get("previous_bmi"),
            "current_bmi": snapshot.get("current_bmi"),
            "delta_bmi": snapshot.get("delta_bmi"),
            "previous_hba1c": snapshot.get("previous_hba1c"),
            "current_hba1c": snapshot.get("current_hba1c"),
            "delta_hba1c": snapshot.get("delta_hba1c"),
            "previous_resting_heart_rate": snapshot.get("previous_resting_heart_rate"),
            "current_resting_heart_rate": snapshot.get("current_resting_heart_rate"),
            "delta_resting_heart_rate": snapshot.get("delta_resting_heart_rate"),
            "previous_smoking_status": snapshot.get("previous_smoking_status"),
            "current_smoking_status": snapshot.get("current_smoking_status"),
            "smoking_status_changed": snapshot.get("smoking_status_changed"),
            "previous_physical_activity": snapshot.get("previous_physical_activity"),
            "current_physical_activity": snapshot.get("current_physical_activity"),
            "physical_activity_changed": snapshot.get("physical_activity_changed"),
            "created_at": now,
            "updated_at": now,
        }

        new_id = session.execute(insert_query, params).scalar()
        session.commit()
        return new_id or 0

    @staticmethod
    def get_patient_temporal_timeline(
        session: Session,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve complete chronological visit timeline for patient_id,
        calculating or retrieving the temporal snapshot for each visit.
        """
        visits_query = text("""
            SELECT id, source_patient_id, visit_date, visit_timestamp,
                   age, gender, bmi, chest_pain_type, systolic_bp, diastolic_bp,
                   resting_heart_rate, max_heart_rate, cholesterol, hdl, ldl,
                   fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                   oldpeak, st_slope, num_major_vessels, thalassemia, smoking,
                   smoking_status, family_history, physical_activity, stress_level, target
            FROM patient_visits
            WHERE source_patient_id = :pid
            ORDER BY visit_date ASC, id ASC
        """)
        rows = session.execute(visits_query, {"pid": str(patient_id)}).mappings().all()
        if not rows:
            return []

        timeline = []
        for i, row in enumerate(rows):
            curr_v = dict(row)
            prev_v = dict(rows[i - 1]) if i > 0 else None
            snapshot = TemporalFeatureService.calculate_snapshot(
                current_data=curr_v,
                previous_data=prev_v,
                patient_id=str(patient_id),
                visit_id=curr_v.get("id"),
                assessment_date=curr_v.get("visit_date"),
            )
            timeline.append({
                "visit_number": i + 1,
                "visit_id": curr_v.get("id"),
                "visit_date": curr_v["visit_date"].isoformat() if curr_v.get("visit_date") else None,
                "clinical_values": curr_v,
                "snapshot": snapshot,
                "is_first_visit": (i == 0),
            })
        return timeline

    @staticmethod
    def get_temporal_coverage_analytics(session: Session) -> Dict[str, Any]:
        """
        Compute true longitudinal coverage statistics across PostgreSQL.
        """
        total_patients = session.execute(text("SELECT COUNT(*) FROM patients")).scalar() or 0
        total_visits = session.execute(text("SELECT COUNT(*) FROM patient_visits")).scalar() or 0

        multi_res = session.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT source_patient_id FROM patient_visits
                GROUP BY source_patient_id HAVING COUNT(*) > 1
            ) s
        """)).scalar() or 0

        single_res = session.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT source_patient_id FROM patient_visits
                GROUP BY source_patient_id HAVING COUNT(*) = 1
            ) s
        """)).scalar() or 0

        max_visits = session.execute(text("""
            SELECT COALESCE(MAX(cnt), 1) FROM (
                SELECT COUNT(*) as cnt FROM patient_visits GROUP BY source_patient_id
            ) s
        """)).scalar() or 1

        avg_visits = round(float(total_visits) / float(max(total_patients, 1)), 2)

        temporal_snapshots_count = session.execute(text(
            "SELECT COUNT(*) FROM patient_temporal_features"
        )).scalar() or 0

        numeric_deltas_available = session.execute(text("""
            SELECT COUNT(*) FROM patient_temporal_features
            WHERE delta_systolic_bp IS NOT NULL
        """)).scalar() or 0

        return {
            "total_patients": int(total_patients),
            "total_visits": int(total_visits),
            "patients_with_multiple_visits": int(multi_res),
            "patients_with_one_visit": int(single_res),
            "average_visits_per_patient": float(avg_visits),
            "maximum_visits_per_patient": int(max_visits),
            "temporal_snapshots": int(temporal_snapshots_count),
            "numeric_delta_availability": int(numeric_deltas_available),
            "temporal_variables_count": 10,
            "numerical_delta_features_count": 8,
            "categorical_change_features_count": 2,
        }
