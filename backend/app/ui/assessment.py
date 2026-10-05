"""
HeartSense – Assessment Page

Combines Manual Patient Details and Hospital Report Upload.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date

from app.config import FEATURE_LABELS
from app.database import get_session, init_db
from app.services.risk_service import assess_patient_risk

def render_assessment():
    st.title("Cardiovascular Health Assessment")
    st.markdown("Enter your health information or upload your medical report to assess your current cardiovascular risk.")
    st.markdown("---")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("Enter Your Details")
        st.markdown("Fill in your health information below.")
        _render_manual_form()

    with col2:
        st.subheader("Upload Hospital Report")
        st.markdown("Upload your medical report and enter/use the required information.")
        st.info("Supported formats: PDF, JPG, PNG (Manual Verification Required)")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "png"])
        
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            st.markdown("### Please verify your details:")
            st.markdown("We could not automatically extract all values from this document format. Please enter the values from your report in the form on the left to continue.")
            
def _render_manual_form(key_prefix="manual_"):
    with st.form(f"{key_prefix}prediction_form", clear_on_submit=False):
        st.markdown("#### Patient Information")
        p_col1, p_col2, p_col3 = st.columns(3)
        patient_code = p_col1.text_input("Patient Code *", placeholder="e.g. P001", key=f"{key_prefix}code")
        patient_name = p_col2.text_input("Patient Name", placeholder="e.g. John Doe", key=f"{key_prefix}name")
        visit_date = p_col3.date_input("Visit Date", value=date.today(), key=f"{key_prefix}date")

        st.markdown("---")
        st.markdown("#### Clinical Data")

        clinical = {}
        row1 = st.columns(3)
        clinical["age"] = row1[0].number_input(FEATURE_LABELS["age"], min_value=1, max_value=120, value=50, step=1, key=f"{key_prefix}age")
        clinical["sex"] = row1[1].selectbox(FEATURE_LABELS["sex"], options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key=f"{key_prefix}sex")
        clinical["cp"] = row1[2].selectbox(FEATURE_LABELS["cp"], options=[0, 1, 2, 3], format_func=lambda x: ["0 – Typical Angina", "1 – Atypical Angina", "2 – Non-anginal Pain", "3 – Asymptomatic"][x], key=f"{key_prefix}cp")

        row2 = st.columns(3)
        clinical["trestbps"] = row2[0].number_input(FEATURE_LABELS["trestbps"], min_value=50, max_value=250, value=120, step=1, key=f"{key_prefix}trestbps")
        clinical["chol"] = row2[1].number_input(FEATURE_LABELS["chol"], min_value=50, max_value=600, value=200, step=1, key=f"{key_prefix}chol")
        clinical["fbs"] = row2[2].selectbox(FEATURE_LABELS["fbs"], options=[0, 1], format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)", key=f"{key_prefix}fbs")

        row3 = st.columns(3)
        clinical["restecg"] = row3[0].selectbox(FEATURE_LABELS["restecg"], options=[0, 1, 2], format_func=lambda x: ["0 – Normal", "1 – ST-T Wave Abnormality", "2 – LV Hypertrophy"][x], key=f"{key_prefix}restecg")
        clinical["thalach"] = row3[1].number_input(FEATURE_LABELS["thalach"], min_value=50, max_value=250, value=150, step=1, key=f"{key_prefix}thalach")
        clinical["exang"] = row3[2].selectbox(FEATURE_LABELS["exang"], options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"{key_prefix}exang")

        row4 = st.columns(3)
        clinical["oldpeak"] = row4[0].number_input(FEATURE_LABELS["oldpeak"], min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f", key=f"{key_prefix}oldpeak")
        clinical["slope"] = row4[1].selectbox(FEATURE_LABELS["slope"], options=[0, 1, 2], format_func=lambda x: ["0 – Upsloping", "1 – Flat", "2 – Downsloping"][x], key=f"{key_prefix}slope")
        clinical["ca"] = row4[2].selectbox(FEATURE_LABELS["ca"], options=[0, 1, 2, 3], key=f"{key_prefix}ca")

        row5 = st.columns(3)
        clinical["thal"] = row5[0].selectbox(FEATURE_LABELS["thal"], options=[0, 1, 2], format_func=lambda x: ["0 – Normal", "1 – Fixed Defect", "2 – Reversible Defect"][x], key=f"{key_prefix}thal")

        st.markdown("---")
        submitted = st.form_submit_button("Assess My Risk", use_container_width=True)

    if submitted:
        if not patient_code.strip():
            st.error("Please enter a Patient Code.")
            return

        with st.spinner("Assessing Risk..."):
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

                _render_results(result, visit_dt, clinical)

            except FileNotFoundError as e:
                st.error(f"Model not found: {e}. Please ensure you have run the training pipeline.")
            except Exception as e:
                st.error(f"Error: {e}")

def _render_results(result, visit_dt, clinical):
    st.markdown("---")
    st.markdown("## Cardiovascular Risk Assessment")
    
    patient_info_str = f"{result['patient_code']}"
    if result.get("patient_name"):
        patient_info_str += f" ({result['patient_name']})"
        
    st.markdown(f"**Patient:** {patient_info_str}")
    st.markdown(f"**Assessment Date:** {visit_dt.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    st.markdown("### Your Assessment Result")
    
    pred = result["prediction"]
    
    if pred["prediction"] == 1:
        st.error("### Disease Predicted", icon="🔴")
    else:
        st.success("### No Disease Predicted", icon="🟢")
        
    pcol1, pcol2 = st.columns(2)
    pcol1.metric("Probability", f"{pred['probability']:.2%}")
    pcol2.metric("Model Used", pred["model_name"])
    
    st.info("This system provides a machine-learning risk assessment and is not a medical diagnosis.")
    
    st.markdown("---")
    st.markdown("### Current Health Values")
    
    val_cols = st.columns(4)
    items = list(clinical.items())
    for i in range(len(items)):
        k, v = items[i]
        label = FEATURE_LABELS.get(k, k)
        val_cols[i % 4].metric(label.split(" (")[0], str(v))

    if not result["is_first_visit"] and result["temporal_changes"]:
        st.markdown("---")
        st.markdown("### Changes Since Your Last Visit")
        
        changes_df = pd.DataFrame(result["temporal_changes"])
        changes_df = changes_df[["label", "previous", "current", "change", "direction"]]
        changes_df.columns = ["Parameter", "Previous", "Current", "Change", "Direction"]
        
        def _color_direction(val):
            if val == "Increased":
                return "color: #e74c3c"
            elif val == "Decreased":
                return "color: #2ecc71"
            return "color: #95a5a6"
            
        st.dataframe(
            changes_df.style.applymap(_color_direction, subset=["Direction"]),
            use_container_width=True,
            hide_index=True
        )
