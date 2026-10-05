"""
HeartSense – Model Training

Orchestrates the full ML pipeline:

    1. Load dataset
    2. Preprocess
    3. SMOTE  (before split!)
    4. Feature selection  (after SMOTE!)
    5. Train-test split
    6. Train 6 models with hyperparameter tuning
    7. Evaluate & compare
    8. Select best model
    9. Save all artifacts

Models: XGBoost, Random Forest, Decision Tree, SVM, KNN, Naive Bayes
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from app.config import (
    MODEL_ARTIFACT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    FEATURE_SELECTION_K,
)
from app.ml.data_loader import load_dataset
from app.ml.preprocessing import clean_data, separate_features_target, inspect_dataset
from app.ml.smote import apply_smote
from app.ml.feature_selection import select_features, save_feature_artifacts
from app.ml.tuning import tune_model
from app.ml.evaluation import (
    evaluate_all_models,
    select_best_model,
    save_metrics,
)


# ── Model factory ──────────────────────────────────────────────────────────────

def get_base_models() -> dict:
    """Return a dict of model_name → untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_training_pipeline(dataset_path: str | None = None) -> dict:
    """
    Execute the complete training pipeline and return a summary dict.

    The pipeline follows the CRITICAL order:
        Preprocessing → SMOTE → Feature Selection → Train-Test Split
    """
    artifact_dir = MODEL_ARTIFACT_DIR
    os.makedirs(artifact_dir, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1 – DATA PREPARATION")
    print("=" * 70)
    df_raw = load_dataset(dataset_path)
    dataset_info = inspect_dataset(df_raw)

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    df_clean = clean_data(df_raw)
    X, y = separate_features_target(df_clean)
    print(f"[Pipeline] Features shape: {X.shape}, Target shape: {y.shape}")

    # ── 3. SMOTE (BEFORE split) ────────────────────────────────────────────────
    X_smote, y_smote, smote_info = apply_smote(X, y)

    # ── 4. Feature Selection (AFTER SMOTE, BEFORE split) ───────────────────────
    X_selected, selector, feature_info = select_features(X_smote, y_smote, k=FEATURE_SELECTION_K)
    save_feature_artifacts(selector, feature_info, artifact_dir)

    # ── 5. Train-Test Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_smote, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_smote
    )
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total": len(X_selected),
        "test_ratio": TEST_SIZE,
    }
    print(f"[Pipeline] Train: {split_info['train_size']}, Test: {split_info['test_size']}")

    # ── 6. Train & Tune all models ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 – MODEL DEVELOPMENT & SELECTION")
    print("=" * 70)
    base_models = get_base_models()
    trained_models = {}
    tuning_results = {}

    for name, model in base_models.items():
        print(f"\n--- Training: {name} ---")
        best_estimator, tune_result = tune_model(model, name, X_train, y_train)
        trained_models[name] = best_estimator
        tuning_results[name] = tune_result

        # Save individual model
        model_file = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best_estimator, os.path.join(artifact_dir, model_file))

    # ── 7. Evaluate & Compare ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3 – MODEL EVALUATION")
    print("=" * 70)
    comparison_df = evaluate_all_models(trained_models, X_test, y_test)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    # ── 8. Select best model ───────────────────────────────────────────────────
    best_name, best_estimator, best_metrics = select_best_model(comparison_df, trained_models)

    # ── 9. Save artifacts ──────────────────────────────────────────────────────
    joblib.dump(best_estimator, os.path.join(artifact_dir, "best_model.joblib"))
    save_metrics(best_metrics, comparison_df, artifact_dir)

    # Calculate and save ROC and CM data for the best model
    try:
        y_proba_best = best_estimator.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            y_proba_best = best_estimator.decision_function(X_test)
        except AttributeError:
            y_proba_best = best_estimator.predict(X_test)
            
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    with open(os.path.join(artifact_dir, "roc_data.json"), "w") as f:
        json.dump(roc_data, f)
        
    y_pred_best = best_estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_data = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
    with open(os.path.join(artifact_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm_data, f)

    # Save tuning results
    # Convert numpy types to native Python for JSON serialization
    tuning_serializable = {}
    for name, result in tuning_results.items():
        tuning_serializable[name] = {
            "model_name": result["model_name"],
            "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in result["best_params"].items()},
            "best_cv_score": result["best_cv_score"],
        }

    with open(os.path.join(artifact_dir, "tuning_results.json"), "w") as f:
        json.dump(tuning_serializable, f, indent=2, default=str)

    # Save pipeline info
    pipeline_info = {
        "dataset_rows_raw": int(df_raw.shape[0]),
        "dataset_cols_raw": int(df_raw.shape[1]),
        "dataset_rows_clean": int(df_clean.shape[0]),
        "smote": smote_info,
        "feature_selection": {
            "selected": feature_info["selected_features"],
            "removed": feature_info["removed_features"],
            "k": feature_info["k"],
        },
        "split": split_info,
        "best_model": best_name,
    }
    with open(os.path.join(artifact_dir, "pipeline_info.json"), "w") as f:
        json.dump(pipeline_info, f, indent=2, default=str)

    # Save test data for evaluation page
    test_data = pd.DataFrame(X_test, columns=X_selected.columns)
    test_data["target"] = y_test.values
    test_data.to_csv(os.path.join(artifact_dir, "test_data.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE – Best model: {best_name}")
    print(f"Artifacts saved to: {artifact_dir}")
    print("=" * 70)

    return {
        "dataset_info": dataset_info,
        "smote_info": smote_info,
        "feature_info": feature_info,
        "split_info": split_info,
        "tuning_results": tuning_results,
        "comparison_df": comparison_df,
        "best_name": best_name,
        "best_metrics": best_metrics,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training_pipeline()
