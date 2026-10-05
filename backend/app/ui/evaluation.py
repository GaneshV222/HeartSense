"""
HeartSense – Model Evaluation Page

Displays:
  • Model comparison table
  • Best model metrics
  • Confusion matrix
  • ROC curve
  • Best hyperparameters
  • Selected features
  • Model comparison bar chart
"""

import os
import json
import streamlit as st
import pandas as pd
import joblib

from app.config import MODEL_ARTIFACT_DIR
from app.ml.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
)


def render_evaluation():
    """Render the model evaluation page."""
    st.title("📊 Model Evaluation")
    st.markdown("Detailed evaluation of all trained models and the selected best model.")
    st.markdown("---")

    artifact_dir = MODEL_ARTIFACT_DIR

    # ── Check artifacts exist ──────────────────────────────────────────────
    required = ["model_metrics.json", "model_comparison.csv", "best_model.joblib", "test_data.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(artifact_dir, f))]
    if missing:
        st.warning(f"⚠️ Missing artifacts: {missing}. Run `python train.py` first.")
        return

    # ── Load artifacts ─────────────────────────────────────────────────────
    with open(os.path.join(artifact_dir, "model_metrics.json"), "r") as f:
        best_metrics = json.load(f)

    comparison_df = pd.read_csv(os.path.join(artifact_dir, "model_comparison.csv"))
    best_model = joblib.load(os.path.join(artifact_dir, "best_model.joblib"))
    test_data = pd.read_csv(os.path.join(artifact_dir, "test_data.csv"))
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # ── Best model metrics ─────────────────────────────────────────────────
    st.subheader(f"🏆 Best Model: {best_metrics.get('model_name', 'N/A')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.2%}")
    col2.metric("Precision", f"{best_metrics.get('precision', 0):.2%}")
    col3.metric("Recall / Sensitivity", f"{best_metrics.get('recall', 0):.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Specificity", f"{best_metrics.get('specificity', 0):.2%}")
    col5.metric("F1-Score", f"{best_metrics.get('f1', 0):.2%}")
    col6.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.4f}")

    st.markdown(
        "> **Selection Rule:** Best model chosen by ROC-AUC (primary), "
        "Recall (secondary), F1-Score (tertiary). Prioritising recall ensures "
        "the model detects as many CVD-positive cases as possible."
    )

    st.markdown("---")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    st.subheader("🔢 Confusion Matrix")
    cm_fig = plot_confusion_matrix(best_model, X_test, y_test, best_metrics.get("model_name", ""))
    st.pyplot(cm_fig)

    st.markdown("---")

    # ── ROC Curve ──────────────────────────────────────────────────────────
    st.subheader("📈 ROC Curve")
    roc_fig = plot_roc_curve(best_model, X_test, y_test, best_metrics.get("model_name", ""))
    st.pyplot(roc_fig)

    st.markdown("---")

    # ── All Models Comparison ──────────────────────────────────────────────
    st.subheader("📊 All Models Comparison")
    st.dataframe(
        comparison_df.style.highlight_max(
            axis=0,
            subset=["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"],
            color="#2ecc71",
        ),
        use_container_width=True,
        hide_index=True,
    )

    comp_fig = plot_model_comparison(comparison_df)
    st.pyplot(comp_fig)

    st.markdown("---")

    # ── Best Hyperparameters ───────────────────────────────────────────────
    st.subheader("⚙️ Best Hyperparameters")
    tuning_path = os.path.join(artifact_dir, "tuning_results.json")
    if os.path.exists(tuning_path):
        with open(tuning_path, "r") as f:
            tuning_results = json.load(f)

        for model_name, info in tuning_results.items():
            with st.expander(f"{model_name}" + (" ⭐" if model_name == best_metrics.get("model_name") else "")):
                st.json(info.get("best_params", {}))
                cv = info.get("best_cv_score")
                if cv is not None:
                    st.metric("Cross-Validation ROC-AUC", f"{cv:.4f}")
    else:
        st.info("Tuning results not available.")

    st.markdown("---")

    # ── Selected Features ──────────────────────────────────────────────────
    st.subheader("🎯 Selected Features")
    features_path = os.path.join(artifact_dir, "selected_features.json")
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            feature_info = json.load(f)

        sel = feature_info.get("selected_features", [])
        rem = feature_info.get("removed_features", [])
        scores = feature_info.get("feature_scores", {})

        st.info(f"**{len(sel)}** features selected, **{len(rem)}** removed")

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.markdown("**✅ Selected**")
            for f in sel:
                st.markdown(f"- `{f}`")
        with fcol2:
            st.markdown("**❌ Removed**")
            for f in rem:
                st.markdown(f"- `{f}`")

        if scores:
            scores_df = pd.DataFrame(
                {"Feature": list(scores.keys()), "MI Score": list(scores.values())}
            ).sort_values("MI Score", ascending=True)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#2ecc71" if f in sel else "#e74c3c" for f in scores_df["Feature"]]
            ax.barh(scores_df["Feature"], scores_df["MI Score"], color=colors)
            ax.set_xlabel("Mutual Information Score")
            ax.set_title("Feature Importance (Green = Selected, Red = Removed)")
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Feature selection info not available.")
