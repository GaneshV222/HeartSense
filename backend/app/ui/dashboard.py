"""
HeartSense – Dashboard Page

Displays:
  • Project title & description
  • Best model & key metrics (metric cards)
  • Dataset information
  • Class distribution (before/after SMOTE)
  • Selected features
  • Model comparison table & chart
"""

import os
import json
import streamlit as st
import pandas as pd
from app.config import MODEL_ARTIFACT_DIR


def render_dashboard():
    """Render the main dashboard page."""
    st.title("🫀 HeartSense Dashboard")
    st.markdown(
        "**Cardiovascular Disease Prediction & Dynamic Risk Assessment System**"
    )
    st.markdown("---")

    artifact_dir = MODEL_ARTIFACT_DIR

    # ── Check if training has been completed ───────────────────────────────
    metrics_path = os.path.join(artifact_dir, "model_metrics.json")
    pipeline_path = os.path.join(artifact_dir, "pipeline_info.json")
    comparison_path = os.path.join(artifact_dir, "model_comparison.csv")
    features_path = os.path.join(artifact_dir, "selected_features.json")

    if not os.path.exists(metrics_path):
        st.warning(
            "⚠️ No trained model found. Please run `python train.py` first."
        )
        return

    # ── Load artefacts ─────────────────────────────────────────────────────
    with open(metrics_path, "r") as f:
        best_metrics = json.load(f)
    with open(pipeline_path, "r") as f:
        pipeline_info = json.load(f)
    with open(features_path, "r") as f:
        feature_info = json.load(f)

    comparison_df = pd.read_csv(comparison_path)

    # ── Best Model Highlight ───────────────────────────────────────────────
    st.subheader("🏆 Best Model")
    st.success(f"**{best_metrics.get('model_name', 'N/A')}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.2%}")
    col2.metric("Precision", f"{best_metrics.get('precision', 0):.2%}")
    col3.metric("Recall / Sensitivity", f"{best_metrics.get('recall', 0):.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Specificity", f"{best_metrics.get('specificity', 0):.2%}")
    col5.metric("F1-Score", f"{best_metrics.get('f1', 0):.2%}")
    col6.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.4f}")

    st.markdown("---")

    # ── Dataset Information ────────────────────────────────────────────────
    st.subheader("📊 Dataset Information")
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Raw Records", pipeline_info.get("dataset_rows_raw", "N/A"))
    d_col2.metric("Clean Records", pipeline_info.get("dataset_rows_clean", "N/A"))
    d_col3.metric("Features (raw)", pipeline_info.get("dataset_cols_raw", "N/A"))

    st.markdown("---")

    # ── Class Distribution ─────────────────────────────────────────────────
    st.subheader("⚖️ Class Distribution")

    smote = pipeline_info.get("smote", {})
    before = smote.get("before", {})
    after = smote.get("after", {})

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**Before SMOTE**")
        before_df = pd.DataFrame(
            {"Class": [str(k) for k in before.keys()],
             "Count": list(before.values())}
        )
        st.dataframe(before_df, use_container_width=True, hide_index=True)
        st.bar_chart(before_df.set_index("Class"))

    with bcol2:
        st.markdown("**After SMOTE**")
        after_df = pd.DataFrame(
            {"Class": [str(k) for k in after.keys()],
             "Count": list(after.values())}
        )
        st.dataframe(after_df, use_container_width=True, hide_index=True)
        st.bar_chart(after_df.set_index("Class"))

    st.markdown("---")

    # ── Selected Features ──────────────────────────────────────────────────
    st.subheader("🎯 Selected Features")
    fs = pipeline_info.get("feature_selection", {})
    sel = fs.get("selected", [])
    rem = fs.get("removed", [])
    st.info(f"**{len(sel)}** features selected out of {len(sel) + len(rem)}")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("**✅ Selected**")
        for feat in sel:
            st.markdown(f"- `{feat}`")
    with fcol2:
        st.markdown("**❌ Removed**")
        if rem:
            for feat in rem:
                st.markdown(f"- `{feat}`")
        else:
            st.markdown("_None removed_")

    # Feature importance scores
    scores = feature_info.get("feature_scores", {})
    if scores:
        scores_df = pd.DataFrame(
            {"Feature": list(scores.keys()), "MI Score": list(scores.values())}
        ).sort_values("MI Score", ascending=False)
        st.bar_chart(scores_df.set_index("Feature"))

    st.markdown("---")

    # ── Train / Test Split ─────────────────────────────────────────────────
    st.subheader("📂 Train / Test Split")
    split = pipeline_info.get("split", {})
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Training Samples", split.get("train_size", "N/A"))
    scol2.metric("Test Samples", split.get("test_size", "N/A"))
    scol3.metric("Test Ratio", f"{split.get('test_ratio', 0.2):.0%}")

    st.markdown("---")

    # ── Model Comparison ───────────────────────────────────────────────────
    st.subheader("📈 Model Comparison")
    st.dataframe(
        comparison_df.style.highlight_max(
            axis=0,
            subset=["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"],
            color="#2ecc71",
        ),
        use_container_width=True,
        hide_index=True,
    )
