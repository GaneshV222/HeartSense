"""Build an auditable snapshot of the current Champion and historical Analytics result."""

from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import DATASET_PATH, MODEL_ARTIFACT_DIR, RANDOM_STATE


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def build_baseline_manifest() -> dict[str, Any]:
    artifact_dir = Path(MODEL_ARTIFACT_DIR)
    root_artifacts = artifact_dir.parent.parent / "artifacts"
    dataset = Path(DATASET_PATH).resolve()
    source = pd.read_csv(dataset, usecols=["prediction"])
    counts = source["prediction"].value_counts().sort_index()
    summary = _read_json(artifact_dir / "temporal_pipeline_summary.json")
    features = _read_json(artifact_dir / "temporal_feature_manifest.json")
    active_path = artifact_dir / "temporal_model_artifact.joblib"
    active = joblib.load(active_path) if active_path.is_file() else {}
    analytics_metrics = _read_json(root_artifacts / "model_metrics.json")
    analytics_pipeline = _read_json(root_artifacts / "pipeline_info.json")
    confusion = analytics_metrics.get("confusion_matrix", {})
    confusion_total = sum(
        int(confusion.get(key, 0)) for key in ("tn", "fp", "fn", "tp")
    )
    saved_split_total = int(analytics_pipeline.get("split", {}).get("test_size", 0))

    package_names = ("scikit-learn", "imbalanced-learn", "xgboost", "pandas", "numpy")
    packages = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    sampler = summary.get("smote", {})
    numerical_features = features.get("numeric_features", [])
    categorical_features = features.get("categorical_features", [])
    sampler_configuration = {
        "algorithm": sampler.get("algorithm", "SMOTENC"),
        "sampling_strategy": "auto",
        "random_state": RANDOM_STATE,
        "k_neighbors": sampler.get("k_neighbors", 5),
        "categorical_features": categorical_features,
        "categorical_feature_positions": list(
            range(len(numerical_features), len(numerical_features) + len(categorical_features))
        ),
        "ordering": "preprocess and fit encoders on complete cumulative data, SMOTENC, then stratified split",
    }
    active_metrics = summary.get("optimized", {}).get("test_metrics", {})
    return {
        "manifest_version": 1,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "production_champion": {
            "active_pointer": str(active_path.resolve()),
            "model_name": active.get("model_name"),
            "model_family": type(active.get("model")).__name__ if active.get("model") is not None else None,
            "feature_schema": active.get("feature_columns", []),
            "feature_count": len(active.get("feature_columns", [])),
            "target_name": active.get("target_name"),
            "target_mapping": active.get("target_mapping"),
            "threshold": active.get("threshold"),
            "threshold_selection": "validation F1, then recall and accuracy tie-breakers; persisted threshold 0.48",
            "parameters": active.get("model_parameters"),
            "artifact_sha256": sha256(active_path.read_bytes()).hexdigest() if active_path.is_file() else None,
            "persisted_diagnostic_metrics": active_metrics,
            "persisted_evaluation_protocol": summary.get("split", {}).get("method"),
        },
        "current_original_dataset_observation": {
            "path": str(dataset),
            "sha256": sha256(dataset.read_bytes()).hexdigest(),
            "real_rows": int(len(source)),
            "class_counts": {str(label): int(count) for label, count in counts.items()},
            "target_mapping": active.get("target_mapping"),
            "warning": "This fingerprint is measured now; the persisted historical Champion manifest did not record its training input hash, so it does not prove historical byte-for-byte source identity.",
        },
        "reconstructed_temporal_pipeline_baseline": {
            "status": "persisted_metrics_audited_but_not_reproduced",
            "dataset_sha256_recorded_by_original_run": None,
            "real_rows": summary.get("split", {}).get("source_rows_before_smote"),
            "feature_order": features.get("features", []),
            "feature_count": features.get("feature_count"),
            "preprocessing_version": "verified 34-feature temporal preprocessing; median numeric imputation and scaling; missing-category imputation and ordinal encoding; one-hot postprocessing",
            "sampler": sampler_configuration,
            "class_counts_before": sampler.get("before"),
            "class_counts_after": sampler.get("after"),
            "synthetic_rows": sampler.get("synthetic_samples_created"),
            "split_seed": RANDOM_STATE,
            "split_sizes": summary.get("split"),
            "model": summary.get("optimized", {}).get("model"),
            "model_parameters": summary.get("optimized", {}).get("parameters"),
            "threshold_selection_method": "validation F1, with recall and accuracy tie-breakers",
            "metric_calculation": "sklearn accuracy, precision, recall, specificity from TN/(TN+FP), F1, ROC-AUC, average precision PR-AUC, and confusion matrix",
            "package_versions_at_snapshot": packages,
            "limitations": [
                "The original persisted run omitted the dataset SHA-256 and training duration.",
                "Current environment has scikit-learn 1.9.1 while the artifact records scikit-learn 1.7.2; joblib load emits an InconsistentVersionWarning.",
                "Current requirements.txt constrains scikit-learn to <1.5, which does not match the installed 1.9.1 runtime.",
                "Reported metrics are SMOTE-before-split diagnostic metrics, not estimates on unseen real patients.",
            ],
        },
        "historical_analytics_84_42_percent": {
            "status": "not_reproducible_from_persisted_artifacts",
            "persisted_metrics": analytics_metrics,
            "pipeline_info": analytics_pipeline,
            "confusion_matrix_total": confusion_total,
            "pipeline_info_test_rows": saved_split_total,
            "discrepancy": (
                f"The saved confusion matrix contains {confusion_total} predictions, but the saved pipeline_info.json reports {saved_split_total} test rows; "
                f"pipeline_info.json also identifies {analytics_pipeline.get('best_model')} as its best model while model_metrics.json identifies {analytics_metrics.get('model_name')}. "
                "The metric and split artifacts therefore cannot be tied to the same measured run."
            ),
            "claimed_accuracy": analytics_metrics.get("accuracy"),
        },
    }


def write_baseline_manifest(path: str | Path | None = None) -> Path:
    output = Path(path) if path else Path(MODEL_ARTIFACT_DIR) / "incremental_learning" / "baseline_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_baseline_manifest(), indent=2, default=str), encoding="utf-8")
    return output
