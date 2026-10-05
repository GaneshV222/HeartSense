"""
HeartSense – Patient History Page

Displays chronological visit history with clinical values and risk assessments.
"""

import streamlit as st
import pandas as pd
from app.database import get_session, init_db
from app.services.patient_service import get_all_patients, get_patient_by_code
from app.services.risk_service import get_patient_timeline
from app.config import FEATURE_LABELS

def render_history():
    st.title("My History")
    st.markdown("View your previous health assessments.")
    st.markdown("---")

    try:
        init_db()
        session = get_session()

        patients = get_all_patients(session)
        if not patients:
            st.info("No records found. Please take an assessment first.")
            session.close()
            return

        patient_options = {
            f"{p.patient_code} – {p.name or 'Unknown'}": p.patient_code
            for p in patients
        }
        
        st.markdown("### Select Patient Record")
        col1, col2 = st.columns(2)
        selected_label = col1.selectbox("Choose a record from the list:", options=list(patient_options.keys()))
        patient_code = patient_options[selected_label]
        
        manual_code = col2.text_input("Or enter Patient Code directly:", placeholder="e.g. P001")
        if manual_code.strip():
            patient_code = manual_code.strip()

        st.markdown("---")

        timeline = get_patient_timeline(session, patient_code)
        session.close()

        if not timeline:
            st.warning(f"No previous assessments found for **{patient_code}**.")
            return

        patient = get_patient_by_code(get_session(), patient_code)
        st.markdown(f"### Health Record: {patient_code} {('- ' + patient.name) if patient and patient.name else ''}")
        st.markdown(f"**Total Assessments:** {len(timeline)}")

        st.markdown("---")
        st.markdown("### Assessment Timeline")

        for i, entry in enumerate(timeline):
            ts = entry["timestamp"]
            ts_str = ts.strftime("%Y-%m-%d") if ts else "N/A"
            pred = entry.get("prediction")
            prob = entry.get("probability")

            result_str = "Needs Attention" if pred == 1 else "Normal" if pred == 0 else "N/A"
            icon = "🔴" if pred == 1 else "🟢" if pred == 0 else "⚪"
            
            with st.expander(
                f"Visit {i + 1} — {ts_str} | {icon} {result_str}",
                expanded=(i == len(timeline) - 1),
            ):
                st.markdown(f"**Assessment Result:** {result_str}")
                if prob is not None:
                    st.markdown(f"**Risk Probability:** {prob:.0%}")

                st.markdown("**Health Measurements:**")
                feats = entry.get("features", {})
                
                # Show measurements in a clean 3-column layout instead of a dense table
                m_cols = st.columns(3)
                idx = 0
                for k, v in feats.items():
                    label = FEATURE_LABELS.get(k, k).split(" (")[0]
                    m_cols[idx % 3].markdown(f"**{label}:** {v}")
                    idx += 1

        probs = [
            {"Date": e["timestamp"], "Risk Probability (%)": e["probability"] * 100}
            for e in timeline if e.get("probability") is not None
        ]
        
        if len(probs) > 1:
            st.markdown("---")
            st.markdown("### Risk Trend Over Time")
            prob_df = pd.DataFrame(probs).set_index("Date")
            st.line_chart(prob_df)

    except Exception as e:
        st.error(f"Error accessing records: {e}")

