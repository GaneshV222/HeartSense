"""
HeartSense – Dynamic / Temporal Risk Assessment Page

Displays:
  • Current vs. previous clinical values
  • Change and direction for each feature
  • Concerning / improving / stable changes
  • Current prediction
  • Full patient timeline
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.database import get_session, init_db
from app.services.patient_service import (
    get_all_patients,
    get_patient_by_code,
    get_patient_visits,
)
from app.services.risk_service import get_patient_timeline
from app.ml.temporal import compute_temporal_changes, summarize_risk_changes
from app.config import FEATURE_LABELS


def render_temporal_risk():
    """Render the dynamic risk assessment page."""
    st.title("📊 Dynamic Risk Assessment")
    st.markdown(
        "Compare a patient's most recent visit with their previous visit to "
        "identify concerning trends and track cardiovascular risk over time."
    )
    st.markdown("---")

    try:
        init_db()
        session = get_session()

        # ── Patient selection ──────────────────────────────────────────────
        patients = get_all_patients(session)
        if not patients:
            st.info("No patients in the database yet.")
            session.close()
            return

        patient_options = {
            f"{p.patient_code} – {p.name or 'N/A'}": p.patient_code
            for p in patients
        }
        selected_label = st.selectbox("Select Patient", list(patient_options.keys()))
        patient_code = patient_options[selected_label]

        st.markdown("---")

        # ── Load visits ────────────────────────────────────────────────────
        patient = get_patient_by_code(session, patient_code)
        if not patient:
            st.warning("Patient not found.")
            session.close()
            return

        visits = get_patient_visits(session, patient.id)

        if len(visits) < 2:
            st.info(
                f"Patient **{patient_code}** has only **{len(visits)}** visit(s). "
                "At least 2 visits are needed for temporal comparison."
            )
            if visits:
                st.subheader("Current Visit Values")
                latest = visits[-1]
                feats = latest.to_feature_dict()
                feat_df = pd.DataFrame([
                    {"Parameter": FEATURE_LABELS.get(k, k), "Value": v}
                    for k, v in feats.items()
                ])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

                if latest.prediction:
                    pred = latest.prediction.prediction
                    prob = latest.prediction.probability
                    if pred == 1:
                        st.error(f"🔴 **Disease Predicted** (Probability: {prob:.2%})")
                    else:
                        st.success(f"🟢 **No Disease Predicted** (Probability: {prob:.2%})")
            session.close()
            return

        # ── Temporal comparison ────────────────────────────────────────────
        current_visit = visits[-1]
        previous_visit = visits[-2]

        st.subheader(f"Patient: {patient_code}" + (f" – {patient.name}" if patient.name else ""))

        vcol1, vcol2 = st.columns(2)
        vcol1.metric("Current Visit", current_visit.visit_timestamp.strftime("%Y-%m-%d"))
        vcol2.metric("Previous Visit", previous_visit.visit_timestamp.strftime("%Y-%m-%d"))

        current_feats = current_visit.to_feature_dict()
        previous_feats = previous_visit.to_feature_dict()

        changes = compute_temporal_changes(current_feats, previous_feats)
        risk_summary = summarize_risk_changes(changes)

        # ── Change table ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Current Values + Changes")

        changes_df = pd.DataFrame(changes)
        changes_df = changes_df[["label", "previous", "current", "change", "direction"]]
        changes_df.columns = ["Parameter", "Previous", "Current", "Change", "Direction"]

        # Style the direction column
        def _color_direction(val):
            if val == "Increased":
                return "color: #e74c3c"
            elif val == "Decreased":
                return "color: #2ecc71"
            return "color: #95a5a6"

        st.dataframe(
            changes_df.style.applymap(_color_direction, subset=["Direction"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Risk summary ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚠️ Risk Change Summary")

        if risk_summary["concerning"]:
            st.error(
                f"**{len(risk_summary['concerning'])} Concerning Change(s):**\n\n"
                + "\n".join([
                    f"- {c['label']}: {c['previous']} → {c['current']} ({c['direction']} by {abs(c['change'])})"
                    for c in risk_summary["concerning"]
                ])
            )

        if risk_summary["improving"]:
            st.success(
                f"**{len(risk_summary['improving'])} Improving Change(s):**\n\n"
                + "\n".join([
                    f"- {c['label']}: {c['previous']} → {c['current']} ({c['direction']} by {abs(c['change'])})"
                    for c in risk_summary["improving"]
                ])
            )

        if risk_summary["stable"]:
            st.info(
                f"**{len(risk_summary['stable'])} Stable / Other:**\n\n"
                + ", ".join([c["label"] for c in risk_summary["stable"]])
            )

        # ── Current prediction ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔮 Current Prediction")

        if current_visit.prediction:
            pred = current_visit.prediction.prediction
            prob = current_visit.prediction.probability
            model = current_visit.prediction.model_name

            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Prediction", "Disease" if pred == 1 else "No Disease")
            pcol2.metric("Probability", f"{prob:.2%}" if prob else "N/A")
            pcol3.metric("Model", model or "N/A")

            if pred == 1:
                st.error("🔴 The model predicts **cardiovascular disease risk**.")
            else:
                st.success("🟢 The model predicts **low cardiovascular disease risk**.")
        else:
            st.warning("No prediction recorded for the current visit.")

        # ── Timeline chart ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Risk Timeline")

        timeline = get_patient_timeline(session, patient_code)
        probs = [
            {"Visit": f"Visit {i+1}\n{e['timestamp'].strftime('%Y-%m-%d')}",
             "Date": e["timestamp"],
             "Probability": e["probability"]}
            for i, e in enumerate(timeline) if e.get("probability") is not None
        ]

        if len(probs) > 1:
            prob_df = pd.DataFrame(probs)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                prob_df["Date"], prob_df["Probability"],
                marker="o", linewidth=2, color="#e74c3c",
            )
            ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5, label="Threshold")
            ax.set_ylabel("Disease Probability")
            ax.set_xlabel("Visit Date")
            ax.set_title(f"CVD Risk Trend – {patient_code}")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Need at least 2 visits with predictions for a trend chart.")

        session.close()

    except Exception as e:
        st.error(f"Error: {e}")
        raise
