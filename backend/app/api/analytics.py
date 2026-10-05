import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from app.config import MODEL_ARTIFACT_DIR

router = APIRouter()

@router.get("/analytics")
def get_analytics():
    artifact_dir = MODEL_ARTIFACT_DIR
    metrics_path = os.path.join(artifact_dir, "model_metrics.json")
    pipeline_path = os.path.join(artifact_dir, "pipeline_info.json")
    comparison_path = os.path.join(artifact_dir, "model_comparison.csv")
    features_path = os.path.join(artifact_dir, "selected_features.json")

    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Analytics data not found")

    with open(metrics_path, "r") as f:
        best_metrics = json.load(f)
    with open(pipeline_path, "r") as f:
        pipeline_info = json.load(f)
    with open(features_path, "r") as f:
        feature_info = json.load(f)
        
    tuning_path = os.path.join(artifact_dir, "tuning_results.json")
    with open(tuning_path, "r") as f:
        tuning_results = json.load(f)
        
    roc_path = os.path.join(artifact_dir, "roc_data.json")
    with open(roc_path, "r") as f:
        roc_data = json.load(f)
        
    cm_path = os.path.join(artifact_dir, "confusion_matrix.json")
    with open(cm_path, "r") as f:
        cm_data = json.load(f)
        
    comparison_df = pd.read_csv(comparison_path)
    
    return {
        "dataset": {
            "raw_records": pipeline_info.get("dataset_rows_raw"),
            "clean_records": pipeline_info.get("dataset_rows_clean"),
            "features_raw": pipeline_info.get("dataset_cols_raw"),
        },
        "smote": pipeline_info.get("smote"),
        "split": pipeline_info.get("split"),
        "features": {
            "selected": feature_info.get("selected_features", feature_info.get("selected", [])),
            "removed": feature_info.get("removed_features", feature_info.get("removed", [])),
            "k": pipeline_info.get("feature_selection", {}).get("k")
        },
        "models": tuning_results,
        "comparison": comparison_df.to_dict(orient="records"),
        "best_model": best_metrics,
        "roc": roc_data,
        "confusion_matrix": cm_data
    }
