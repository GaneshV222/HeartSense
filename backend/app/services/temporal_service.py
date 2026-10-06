"""
HeartSense – TemporalFeatureService

The single authoritative temporal feature calculation, persistence, and retrieval service.
Encapsulates all logic for the critical cardiovascular features:
  - 8 Numerical: Systolic BP, Diastolic BP, Cholesterol, LDL, HDL, BMI, HbA1c, Resting Heart Rate
    (Current, Previous, Change = Current - Previous, Rate = Change / days_between_visits)
  - 2 Lifestyle: Smoking, Physical Activity
    (Current, Previous, Changed boolean flag)

Rules:
  - temporal_patient_data contains ONLY the latest two visits per patient.
  - Previous values MUST come from the immediately preceding chronological visit of the SAME patient.
  - If a patient has only 1 visit (first visit): previous = NULL, change = NULL, rate = NULL, changed = NULL.
  - Missing previous visit is NEVER treated as change = 0.
  - Upsert strategy ensures exactly one row per patient in temporal_patient_data.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import PatientVisit, TemporalPatientData
from app.config import (
    CRITICAL_NUMERICAL_FEATURES,
    CRITICAL_LIFESTYLE_FEATURES,
    ALL_10_TEMPORAL_VARIABLES,
)

NUMERICAL_VARIABLES = CRITICAL_NUMERICAL_FEATURES
CATEGORICAL_VARIABLES = CRITICAL_LIFESTYLE_FEATURES

NUMERICAL_DELTA_FIELDS = [f"delta_{var}" for var in NUMERICAL_VARIABLES] + [f"{var}_change" for var in NUMERICAL_VARIABLES]
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
    "smoking": {"label": "Smoking Status", "unit": "", "type": "categorical"},
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
        return "True"
    if low in {"0", "false", "no", "non-smoker"}:
        return "False"
    return s.capitalize()


class TemporalFeatureService:
    """
    Authoritative service for computing, upserting, and retrieving temporal patient data.
    """

    @staticmethod
    def calculate_snapshot(
        current_data: Dict[str, Any],
        previous_data: Optional[Dict[str, Any]],
        patient_id: str,
        current_visit_id: Optional[int] = None,
        previous_visit_id: Optional[int] = None,
        current_visit_date: Optional[datetime] = None,
        previous_visit_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate temporal changes and rates between current visit and immediately preceding visit.
        """
        current_dt = current_visit_date or current_data.get("visit_date")
        if isinstance(current_dt, str):
            try:
                current_dt = datetime.fromisoformat(current_dt.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                current_dt = None

        prev_dt = previous_visit_date or (previous_data.get("visit_date") if previous_data else None)
        if isinstance(prev_dt, str):
            try:
                prev_dt = datetime.fromisoformat(prev_dt.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                prev_dt = None

        is_first_visit = (previous_data is None or len(previous_data) == 0)

        # Days between visits
        days_between: Optional[float] = None
        if not is_first_visit and current_dt and prev_dt:
            diff_days = (current_dt - prev_dt).total_seconds() / 86400.0
            days_between = max(round(diff_days, 1), 1.0)

        snapshot: Dict[str, Any] = {
            "patient_id": str(patient_id),
            "current_visit_id": current_visit_id or current_data.get("id") or current_data.get("visit_id"),
            "previous_visit_id": previous_visit_id or (previous_data.get("id") or previous_data.get("visit_id") if previous_data else None),
            "current_visit_date": current_dt,
            "previous_visit_date": prev_dt,
            "days_between_visits": days_between,
            "is_first_visit": is_first_visit,
            "has_previous_visit": not is_first_visit,
        }

        # ── 1. Calculate 8 Critical Numerical Changes and Rates ──────────────
        numerical_details: Dict[str, Dict[str, Optional[float]]] = {}
        for var in NUMERICAL_VARIABLES:
            curr_raw = current_data.get(var)
            curr_val = _safe_float(curr_raw)
            snapshot[f"current_{var}"] = curr_val

            if is_first_visit or previous_data is None:
                prev_val = None
                change_val = None
                rate_val = None
            else:
                prev_raw = previous_data.get(var)
                prev_val = _safe_float(prev_raw)
                if curr_val is not None and prev_val is not None:
                    change_val = round(curr_val - prev_val, 2)
                    if days_between is not None and days_between > 0:
                        rate_val = round(change_val / days_between, 4)
                    else:
                        rate_val = 0.0
                else:
                    change_val = None
                    rate_val = None

            snapshot[f"previous_{var}"] = prev_val
            snapshot[f"prev_{var}"] = prev_val
            snapshot[f"{var}_change"] = change_val
            snapshot[f"delta_{var}"] = change_val
            snapshot[f"{var}_rate"] = rate_val

            numerical_details[var] = {
                "previous": prev_val,
                "current": curr_val,
                "change": change_val,
                "delta": change_val,
                "rate": rate_val,
                "unit": VARIABLE_METADATA.get(var, {}).get("unit", ""),
                "label": VARIABLE_METADATA.get(var, {}).get("label", var),
            }

        # ── 2. Calculate 2 Critical Lifestyle Changes ────────────────────────
        categorical_details: Dict[str, Dict[str, Any]] = {}

        # Smoking
        curr_smoke_raw = current_data.get("smoking") or current_data.get("smoking_status")
        curr_smoke = normalize_smoking_value(curr_smoke_raw)
        snapshot["current_smoking"] = curr_smoke
        snapshot["current_smoking_status"] = curr_smoke

        if is_first_visit or previous_data is None:
            prev_smoke = None
            smoke_changed = None
        else:
            prev_smoke_raw = previous_data.get("smoking") or previous_data.get("smoking_status")
            prev_smoke = normalize_smoking_value(prev_smoke_raw)
            if curr_smoke is not None and prev_smoke is not None:
                smoke_changed = bool(curr_smoke.lower() != prev_smoke.lower())
            else:
                smoke_changed = None

        snapshot["previous_smoking"] = prev_smoke
        snapshot["prev_smoking"] = prev_smoke
        snapshot["previous_smoking_status"] = prev_smoke
        snapshot["smoking_changed"] = smoke_changed
        snapshot["smoking_status_changed"] = smoke_changed
        categorical_details["smoking"] = {
            "previous": prev_smoke,
            "current": curr_smoke,
            "changed": smoke_changed,
            "label": "Smoking Status",
        }

        # Physical Activity
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
        snapshot["prev_physical_activity"] = prev_act
        snapshot["physical_activity_changed"] = act_changed
        categorical_details["physical_activity"] = {
            "previous": prev_act,
            "current": curr_act,
            "changed": act_changed,
            "label": "Physical Activity Level",
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
        """
        query = """
            SELECT id, patient_id, visit_number, visit_date, visit_timestamp,
                   age, gender, bmi, chest_pain_type, systolic_bp, diastolic_bp,
                   resting_heart_rate, max_heart_rate, cholesterol, hdl, ldl,
                   fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                   oldpeak, st_slope, num_major_vessels, thalassemia, smoking,
                   smoking_status, family_history, physical_activity, stress_level, prediction
            FROM patient_visits
            WHERE patient_id = :pid
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
    def upsert_temporal_patient_data(
        session: Session,
        patient_id: str,
        current_visit: Dict[str, Any] | PatientVisit,
        previous_visit: Optional[Dict[str, Any] | PatientVisit] = None,
    ) -> TemporalPatientData:
        """
        Upsert the single temporal record for this patient in temporal_patient_data table.
        Contains ONLY the latest two visits (previous and current).
        """
        cur_dict = current_visit if isinstance(current_visit, dict) else current_visit.to_feature_dict()
        prev_dict = (
            previous_visit if isinstance(previous_visit, dict)
            else (previous_visit.to_feature_dict() if previous_visit else None)
        )

        cur_id = cur_dict.get("id") or cur_dict.get("visit_id")
        prev_id = prev_dict.get("id") or prev_dict.get("visit_id") if prev_dict else None

        cur_date = cur_dict.get("visit_date")
        prev_date = prev_dict.get("visit_date") if prev_dict else None

        snapshot = TemporalFeatureService.calculate_snapshot(
            current_data=cur_dict,
            previous_data=prev_dict,
            patient_id=str(patient_id),
            current_visit_id=cur_id,
            previous_visit_id=prev_id,
            current_visit_date=cur_date,
            previous_visit_date=prev_date,
        )

        temp_record = session.query(TemporalPatientData).filter_by(patient_id=str(patient_id)).first()
        if not temp_record:
            temp_record = TemporalPatientData(patient_id=str(patient_id))
            session.add(temp_record)

        temp_record.current_visit_id = snapshot["current_visit_id"]
        temp_record.previous_visit_id = snapshot["previous_visit_id"]
        temp_record.current_visit_date = snapshot["current_visit_date"]
        temp_record.previous_visit_date = snapshot["previous_visit_date"]
        temp_record.days_between_visits = snapshot["days_between_visits"]

        # Critical numerical features
        for var in NUMERICAL_VARIABLES:
            setattr(temp_record, f"current_{var}", snapshot.get(f"current_{var}"))
            setattr(temp_record, f"previous_{var}", snapshot.get(f"previous_{var}"))
            setattr(temp_record, f"{var}_change", snapshot.get(f"{var}_change"))
            setattr(temp_record, f"{var}_rate", snapshot.get(f"{var}_rate"))

        # Lifestyle features
        temp_record.current_smoking = snapshot.get("current_smoking")
        temp_record.previous_smoking = snapshot.get("previous_smoking")
        temp_record.smoking_changed = snapshot.get("smoking_changed")

        temp_record.current_physical_activity = snapshot.get("current_physical_activity")
        temp_record.previous_physical_activity = snapshot.get("previous_physical_activity")
        temp_record.physical_activity_changed = snapshot.get("physical_activity_changed")

        temp_record.updated_at = datetime.utcnow()
        session.flush()
        return temp_record

    @staticmethod
    def get_patient_temporal_timeline(
        session: Session,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve complete chronological visit timeline for patient_id,
        calculating the temporal snapshot for each visit relative to its immediate predecessor.
        """
        visits_query = text("""
            SELECT id, patient_id, visit_number, visit_date, visit_timestamp,
                   age, gender, bmi, chest_pain_type, systolic_bp, diastolic_bp,
                   resting_heart_rate, max_heart_rate, cholesterol, hdl, ldl,
                   fasting_blood_sugar, hba1c, diabetes, resting_ecg, exercise_angina,
                   oldpeak, st_slope, num_major_vessels, thalassemia, smoking,
                   smoking_status, family_history, physical_activity, stress_level, prediction
            FROM patient_visits
            WHERE patient_id = :pid
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
                current_visit_id=curr_v.get("id"),
                previous_visit_id=prev_v.get("id") if prev_v else None,
                current_visit_date=curr_v.get("visit_date"),
                previous_visit_date=prev_v.get("visit_date") if prev_v else None,
            )
            timeline.append({
                "visit_number": curr_v.get("visit_number") or (i + 1),
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
                SELECT patient_id FROM patient_visits
                GROUP BY patient_id HAVING COUNT(*) > 1
            ) s
        """)).scalar() or 0

        single_res = session.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT patient_id FROM patient_visits
                GROUP BY patient_id HAVING COUNT(*) = 1
            ) s
        """)).scalar() or 0

        max_visits = session.execute(text("""
            SELECT COALESCE(MAX(cnt), 1) FROM (
                SELECT COUNT(*) as cnt FROM patient_visits GROUP BY patient_id
            ) s
        """)).scalar() or 1

        avg_visits = round(float(total_visits) / float(max(total_patients, 1)), 2)

        temporal_records_count = session.execute(text(
            "SELECT COUNT(*) FROM temporal_patient_data"
        )).scalar() or 0

        numeric_deltas_available = session.execute(text("""
            SELECT COUNT(*) FROM temporal_patient_data
            WHERE systolic_bp_change IS NOT NULL
        """)).scalar() or 0

        return {
            "total_patients": int(total_patients),
            "total_visits": int(total_visits),
            "patients_with_multiple_visits": int(multi_res),
            "patients_with_one_visit": int(single_res),
            "average_visits_per_patient": float(avg_visits),
            "maximum_visits_per_patient": int(max_visits),
            "temporal_snapshots": int(temporal_records_count),
            "temporal_records_count": int(temporal_records_count),
            "numeric_delta_availability": int(numeric_deltas_available),
            "temporal_variables_count": 10,
            "numerical_delta_features_count": 8,
            "categorical_change_features_count": 2,
        }
