"""
HeartSense – Analytics Page

Displays technical project information, evaluation metrics, and model comparison.
"""

import os
import json
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from app.config import MODEL_ARTIFACT_DIR
from app.ml.evaluation import plot_confusion_matrix, plot_roc_curve, plot_model_comparison

def render_analytics():
    st.title("Analytics & Model Evaluation")
    st.markdown("Technical details on dataset, features, and machine learning models.")
    st.markdown("---")

    artifact_dir = MODEL_ARTIFACT_DIR

    metrics_path = os.path.join(artifact_dir, "model_metrics.json")
    pipeline_path = os.path.join(artifact_dir, "pipeline_info.json")
    comparison_path = os.path.join(artifact_dir, "model_comparison.csv")
    features_path = os.path.join(artifact_dir, "selected_features.json")
    best_model_path = os.path.join(artifact_dir, "best_model.joblib")
    test_data_path = os.path.join(artifact_dir, "test_data.csv")

    required = [metrics_path, pipeline_path, comparison_path, features_path, best_model_path, test_data_path]
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        st.warning("⚠️ Training artifacts not found. Please run the training pipeline first.")
        return

    with open(metrics_path, "r") as f:
        best_metrics = json.load(f)
    with open(pipeline_path, "r") as f:
        pipeline_info = json.load(f)
    with open(features_path, "r") as f:
        feature_info = json.load(f)

    comparison_df = pd.read_csv(comparison_path)
    best_model = joblib.load(best_model_path)
    test_data = pd.read_csv(test_data_path)
    
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    st.subheader("📊 Dataset Information")
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Raw Records", pipeline_info.get("dataset_rows_raw", "N/A"))
    d_col2.metric("Clean Records", pipeline_info.get("dataset_rows_clean", "N/A"))
    d_col3.metric("Features (raw)", pipeline_info.get("dataset_cols_raw", "N/A"))

    st.markdown("---")

    st.subheader(f"🏆 Best Performing Model: {best_metrics.get('model_name', 'N/A')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.2%}")
    col2.metric("Precision", f"{best_metrics.get('precision', 0):.2%}")
    col3.metric("Recall / Sensitivity", f"{best_metrics.get('recall', 0):.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Specificity", f"{best_metrics.get('specificity', 0):.2%}")
    col5.metric("F1-Score", f"{best_metrics.get('f1', 0):.2%}")
    col6.metric("ROC-AUC", f"{best_metrics.get('roc_auc', 0):.4f}")

    st.markdown("---")
    
    st.subheader("🎯 Selected Features")
    sel = feature_info.get("selected_features", feature_info.get("selected", []))
    rem = feature_info.get("removed_features", feature_info.get("removed", []))
    
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("**Selected**")
        st.markdown(", ".join([f"`{feat}`" for feat in sel]))
    with fcol2:
        st.markdown("**Removed**")
        st.markdown(", ".join([f"`{feat}`" for feat in rem]) if rem else "_None removed_")

    st.markdown("---")
    
    st.subheader("🔢 Confusion Matrix")
    cm_fig = plot_confusion_matrix(best_model, X_test, y_test, best_metrics.get("model_name", ""))
    st.pyplot(cm_fig)

    st.markdown("---")
    
    st.subheader("📈 ROC Curve")
    roc_fig = plot_roc_curve(best_model, X_test, y_test, best_metrics.get("model_name", ""))
    st.pyplot(roc_fig)

    st.markdown("---")
    
    st.subheader("📊 Model Comparison")
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
