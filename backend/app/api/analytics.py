"""
HeartSense – Analytics API
"""

import json
import logging
import os
from fastapi import APIRouter
from sqlalchemy.exc import SQLAlchemyError
from app.config import MODEL_ARTIFACT_DIR
from app.database import get_session
from app.services.temporal_service import (
    TemporalFeatureService,
    ALL_10_TEMPORAL_VARIABLES,
    NUMERICAL_DELTA_FIELDS,
    CATEGORICAL_CHANGED_FIELDS,
    VARIABLE_METADATA,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Unable to read analytics artifact %s", path)
        return default


@router.get("/analytics")
def get_analytics():
    artifact_dir = MODEL_ARTIFACT_DIR
    fallback_artifact_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
    )
    summary = _load_json(os.path.join(artifact_dir, "temporal_pipeline_summary.json"), {})
    if not summary and fallback_artifact_dir != os.path.abspath(artifact_dir):
        summary = _load_json(os.path.join(fallback_artifact_dir, "temporal_pipeline_summary.json"), {})
    optimized = summary.get("optimized", {})
    baseline = summary.get("baseline", {})
    optimized_metrics = optimized.get("test_metrics", {})
    baseline_metrics = baseline.get("test_metrics", {})
    confusion_matrix_data = optimized_metrics.get("confusion_matrix", {})
    smote_data = summary.get("smote", {})

    has_model_metrics = bool(optimized_metrics)

    smote_mode_label = (
        smote_data.get("applied_to")
        or "full dataset before train/validation/test split"
    )
    official_mode = str(smote_mode_label)

    def model_row(name, values, mode_label=None):
        mode_label = mode_label or official_mode
        return {
            "model_name": name,
            "accuracy": values.get("accuracy"),
            "precision": values.get("precision"),
            "recall": values.get("recall_sensitivity", values.get("recall")),
            "specificity": values.get("specificity"),
            "f1": values.get("f1"),
            "roc_auc": values.get("roc_auc"),
            "pr_auc": values.get("pr_auc"),
            "confusion_matrix": values.get("confusion_matrix"),
            "threshold": values.get("threshold"),
            "smote_mode": mode_label,
            "mode": mode_label,
        }

    validation_candidates = summary.get("validation_candidates", {})
    comparison = []
    for model_name, values in validation_candidates.items():
        comparison.append(
            model_row(
                model_name,
                {
                    "accuracy": values.get("accuracy"),
                    "precision": values.get("precision"),
                    "recall_sensitivity": values.get("recall_sensitivity"),
                    "specificity": values.get("specificity"),
                    "f1": values.get("f1"),
                    "roc_auc": values.get("roc_auc"),
                    "pr_auc": values.get("pr_auc"),
                    "confusion_matrix": values.get("confusion_matrix"),
                    "threshold": values.get("threshold"),
                },
                official_mode,
            )
        )
    if not comparison:
        comparison = [
            model_row(baseline.get("model", "Baseline"), baseline_metrics),
            model_row(optimized.get("model", "Optimized"), optimized_metrics),
        ]
    comparison = [row for row in comparison if row["accuracy"] is not None]
    selected_model_test_metrics = model_row(
        optimized.get("model", "Selected model"), optimized_metrics
    )
    experimental_summary = summary.get("experimental_smote_before_split")
    experimental_comparison = []
    if experimental_summary:
        for model_name, values in (experimental_summary.get("metrics") or {}).items():
            experimental_comparison.append({
                "model_name": model_name,
                "accuracy": values.get("accuracy"),
                "precision": values.get("precision"),
                "recall": values.get("recall_sensitivity"),
                "specificity": values.get("specificity"),
                "f1": values.get("f1"),
                "roc_auc": values.get("roc_auc"),
                "pr_auc": values.get("pr_auc"),
                "confusion_matrix": values.get("confusion_matrix"),
                "threshold": values.get("threshold"),
                "smote_mode": experimental_summary.get("label", "Experimental / leakage-contaminated / SMOTE before split"),
                "mode": experimental_summary.get("label", "Experimental / leakage-contaminated / SMOTE before split"),
            })
    comparative_analysis = (
        [
            {
                "model_name": optimized.get("model"),
                "before_accuracy": baseline_metrics.get("accuracy"),
                "after_accuracy": optimized_metrics.get("accuracy"),
                "before_precision": baseline_metrics.get("precision"),
                "after_precision": optimized_metrics.get("precision"),
                "before_recall": baseline_metrics.get("recall_sensitivity"),
                "after_recall": optimized_metrics.get("recall_sensitivity"),
                "before_f1": baseline_metrics.get("f1"),
                "after_f1": optimized_metrics.get("f1"),
                "before_roc_auc": baseline_metrics.get("roc_auc"),
                "after_roc_auc": optimized_metrics.get("roc_auc"),
            }
        ]
        if baseline_metrics and optimized_metrics
        else []
    )
    cross_validation = summary.get("cross_validation", {})
    cross_validation_summary = []
    for model_name, values in (summary.get("validation_candidates", {}) or {}).items():
        if isinstance(values, dict) and "fold_metrics" in values:
            cross_validation_summary.append({"model_name": model_name, **values})
    if not cross_validation_summary:
        cross_validation_summary = [
            {"model_name": baseline.get("model"), **cross_validation["baseline"]}
            for _ in [0]
            if "baseline" in cross_validation
        ] + [
            {"model_name": optimized.get("model"), **cross_validation["optimized"]}
            for _ in [0]
            if "optimized" in cross_validation
        ]
    cross_validation_folds = [
        {
            "model_name": row.get("model_name"),
            **fold,
        }
        for row in cross_validation_summary
        for fold in row.get("fold_metrics", [])
    ]
    cross_validation_comparison = []
    smote_model_comparison = comparison

    # Fetch live database temporal statistics
    live_coverage_error = None
    session = None
    try:
        session = get_session()
        live_coverage = TemporalFeatureService.get_temporal_coverage_analytics(session)
    except SQLAlchemyError:
        live_coverage = {}
        live_coverage_error = "Unable to load live PostgreSQL temporal statistics."
        logger.exception("Unable to load live PostgreSQL temporal statistics")
    finally:
        if session is not None:
            session.close()

    preprocessing = summary.get("dataset", {})
    smote_data = summary.get("smote", {})
    split_data = summary.get("split", {})
    before_temporal = {"models": comparison[:1]}
    after_temporal = {"models": comparison[-1:]}
    original_smote_count = smote_data.get("samples_before")
    resampled_smote_count = smote_data.get("samples_after")
    synthetic_count = smote_data.get("synthetic_samples_created")

    return {
        "has_model_metrics": has_model_metrics,
        "smote_mode": official_mode,
        "smote_mode_label": official_mode,
        "database_error": live_coverage_error,
        "dataset": {
            "raw_records": preprocessing.get("rows"),
            "clean_records": preprocessing.get("rows"),
            "unique_patients": preprocessing.get("patient_count"),
            "visits_per_patient": (
                preprocessing.get("rows") / preprocessing.get("patient_count")
                if preprocessing.get("rows") and preprocessing.get("patient_count")
                else None
            ),
            "temporal_status": (
                f"{preprocessing.get('multi_visit_patients', 0)} patients have multiple visits."
                if preprocessing
                else "Dataset validation report is not available."
            ),
        },
        "preprocessing": preprocessing,
        "database": summary.get("database", {}),
        "temporal_statistics": live_coverage,
        "temporal_stats": live_coverage,
        "feature_architecture": {
            "source_variables_count": 10,
            "source_variables": [
                {"name": var, "label": VARIABLE_METADATA[var]["label"], "unit": VARIABLE_METADATA[var]["unit"], "type": VARIABLE_METADATA[var]["type"]}
                for var in ALL_10_TEMPORAL_VARIABLES
            ],
            "numerical_delta_features_count": 8,
            "numerical_delta_features": NUMERICAL_DELTA_FIELDS,
            "categorical_change_features_count": 2,
            "categorical_change_features": CATEGORICAL_CHANGED_FIELDS,
        },
        "smote": smote_data,
        "smote_model_comparison": smote_model_comparison,
        "smote_analysis": {
            "records_before": original_smote_count,
            "records_after": resampled_smote_count,
            "synthetic_created": synthetic_count,
            "distribution_before": smote_data.get("before", {}),
            "distribution_after": smote_data.get("after", {}),
            "smote_stage": smote_data.get("applied_to"),
        },
        "split": split_data,
        "train_test": split_data,
        "before_temporal": before_temporal,
        "after_temporal": after_temporal,
        "comparative_analysis": comparative_analysis,
        "cross_validation_summary": cross_validation_summary,
        "cross_validation_folds": cross_validation_folds,
        "cross_validation_comparison": cross_validation_comparison,
        "experimental_comparison": experimental_comparison,
        "comparison": comparison,
        "model_comparison": comparison,
        "selected_model_test_metrics": selected_model_test_metrics,
        "best_model": {
            "name": optimized.get("model"),
            "model_name": optimized.get("model"),
            "model_type": optimized.get("model"),
            "accuracy": optimized_metrics.get("accuracy"),
            "precision": optimized_metrics.get("precision"),
            "recall": optimized_metrics.get("recall_sensitivity"),
            "specificity": optimized_metrics.get("specificity"),
            "f1": optimized_metrics.get("f1"),
            "roc_auc": optimized_metrics.get("roc_auc"),
            "pr_auc": optimized_metrics.get("pr_auc"),
            "confusion_matrix": confusion_matrix_data,
        },
        "roc": {},
        "confusion_matrix": confusion_matrix_data,
        "feature_importances": summary.get("feature_importances", {}),
    }
