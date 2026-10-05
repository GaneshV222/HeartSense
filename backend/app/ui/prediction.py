"""
HeartSense – Patient Prediction Page

Streamlit form for entering clinical data, generating a prediction,
and storing the visit in the database.
"""

import streamlit as st
from datetime import datetime, date

from app.config import FEATURE_NAMES, FEATURE_LABELS
from app.database import get_session, init_db
from app.services.risk_service import assess_patient_risk


def render_prediction():
    """Render the patient prediction form and results."""
    st.title("🩺 Patient Prediction")
    st.markdown(
        "Enter patient clinical data to generate a cardiovascular disease "
        "risk prediction."
    )

    st.info(
        "⚕️ **Disclaimer:** This system provides a machine-learning risk "
        "assessment and is **not a medical diagnosis**. Always consult a "
        "qualified healthcare professional.",
        icon="⚠️",
    )

    st.markdown("---")

    # ── Patient Identification ─────────────────────────────────────────────
    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("Patient Information")
        p_col1, p_col2, p_col3 = st.columns(3)
        patient_code = p_col1.text_input("Patient Code *", placeholder="e.g. P001")
        patient_name = p_col2.text_input("Patient Name", placeholder="e.g. John Doe")
        visit_date = p_col3.date_input("Visit Date", value=date.today())

        st.markdown("---")
        st.subheader("Clinical Data")

        # ── Clinical feature inputs ────────────────────────────────────────
        clinical = {}

        row1 = st.columns(3)
        clinical["age"] = row1[0].number_input(
            FEATURE_LABELS["age"], min_value=1, max_value=120, value=50, step=1
        )
        clinical["sex"] = row1[1].selectbox(
            FEATURE_LABELS["sex"], options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female"
        )
        clinical["cp"] = row1[2].selectbox(
            FEATURE_LABELS["cp"],
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Typical Angina",
                1: "1 – Atypical Angina",
                2: "2 – Non-anginal Pain",
                3: "3 – Asymptomatic",
            }[x],
        )

        row2 = st.columns(3)
        clinical["trestbps"] = row2[0].number_input(
            FEATURE_LABELS["trestbps"], min_value=50, max_value=250, value=120, step=1
        )
        clinical["chol"] = row2[1].number_input(
            FEATURE_LABELS["chol"], min_value=50, max_value=600, value=200, step=1
        )
        clinical["fbs"] = row2[2].selectbox(
            FEATURE_LABELS["fbs"], options=[0, 1],
            format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)",
        )

        row3 = st.columns(3)
        clinical["restecg"] = row3[0].selectbox(
            FEATURE_LABELS["restecg"],
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Normal",
                1: "1 – ST-T Wave Abnormality",
                2: "2 – LV Hypertrophy",
            }[x],
        )
        clinical["thalach"] = row3[1].number_input(
            FEATURE_LABELS["thalach"], min_value=50, max_value=250, value=150, step=1
        )
        clinical["exang"] = row3[2].selectbox(
            FEATURE_LABELS["exang"], options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        row4 = st.columns(3)
        clinical["oldpeak"] = row4[0].number_input(
            FEATURE_LABELS["oldpeak"], min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f"
        )
        clinical["slope"] = row4[1].selectbox(
            FEATURE_LABELS["slope"],
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Upsloping",
                1: "1 – Flat",
                2: "2 – Downsloping",
            }[x],
        )
        clinical["ca"] = row4[2].selectbox(
            FEATURE_LABELS["ca"], options=[0, 1, 2, 3]
        )

        row5 = st.columns(3)
        clinical["thal"] = row5[0].selectbox(
            FEATURE_LABELS["thal"],
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Normal",
                1: "1 – Fixed Defect",
                2: "2 – Reversible Defect",
            }[x],
        )

        submitted = st.form_submit_button("🔍 Generate Prediction", use_container_width=True)

    # ── Handle submission ──────────────────────────────────────────────────
    if submitted:
        if not patient_code.strip():
            st.error("Please enter a Patient Code.")
            return

        with st.spinner("Generating prediction…"):
            try:
                init_db()
                session = get_session()
                visit_dt = datetime.combine(visit_date, datetime.min.time())

                result = assess_patient_risk(
                    session=session,
                    patient_code=patient_code.strip(),
                    clinical_values=clinical,
                    visit_date=visit_dt,
                    patient_name=patient_name.strip() or None,
                )
                session.close()

                # ── Display results ────────────────────────────────────────
                st.markdown("---")
                st.subheader("📋 Prediction Results")

                pred = result["prediction"]
                if pred["prediction"] == 1:
                    st.error(f"🔴 **{pred['label']}**")
                else:
                    st.success(f"🟢 **{pred['label']}**")

                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.metric("Prediction", "Disease" if pred["prediction"] == 1 else "No Disease")
                rcol2.metric("Probability", f"{pred['probability']:.2%}")
                rcol3.metric("Model", pred["model_name"])

                st.caption(f"Visit recorded: {visit_dt.strftime('%Y-%m-%d')}")

                # ── Temporal changes (if returning patient) ────────────────
                if not result["is_first_visit"] and result["temporal_changes"]:
                    st.markdown("---")
                    st.subheader("📊 Changes from Previous Visit")
                    import pandas as pd
                    changes_df = pd.DataFrame(result["temporal_changes"])
                    changes_df = changes_df[["label", "previous", "current", "change", "direction"]]
                    changes_df.columns = ["Parameter", "Previous", "Current", "Change", "Direction"]
                    st.dataframe(changes_df, use_container_width=True, hide_index=True)

                    risk = result["risk_summary"]
                    if risk and risk["concerning"]:
                        st.warning(
                            f"⚠️ {len(risk['concerning'])} concerning change(s) detected: "
                            + ", ".join([c["label"] for c in risk["concerning"]])
                        )
                    if risk and risk["improving"]:
                        st.success(
                            f"✅ {len(risk['improving'])} improving change(s): "
                            + ", ".join([c["label"] for c in risk["improving"]])
                        )
                else:
                    if result["is_first_visit"]:
                        st.info("ℹ️ This is the patient's first visit – no previous data for comparison.")

            except FileNotFoundError as e:
                st.error(f"Model not found: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
                raise
