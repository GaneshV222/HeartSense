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
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    evaluate_model,
    select_best_model,
    save_metrics,
)


# ── Model factory ──────────────────────────────────────────────────────────────

def get_base_models() -> dict:
    """Return a dict of model_name → untrained estimator."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=RANDOM_STATE, probability=True, class_weight="balanced")),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]),
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

    # Keep the unsampled feature space for a clean before/after SMOTE accuracy comparison.
    X_before_smt = X.copy()
    y_before_smt = y.copy()

    # ── 3. SMOTE (BEFORE split) ────────────────────────────────────────────────
    X_smote, y_smote, smote_info = apply_smote(X, y)

    # ── 4. Feature Selection (AFTER SMOTE, BEFORE split) ───────────────────────
    X_selected, selector, feature_info = select_features(X_smote, y_smote, k=FEATURE_SELECTION_K)
    save_feature_artifacts(selector, feature_info, artifact_dir)

    # Apply the same selector to the original encoded data to compare the same feature set before/after SMOTE.
    X_before_prepared = X_before_smt.copy()
    if isinstance(X_before_prepared, pd.DataFrame):
        non_numeric = list(X_before_prepared.select_dtypes(exclude=[np.number]).columns)
        if non_numeric:
            X_before_prepared = pd.get_dummies(X_before_prepared, columns=non_numeric, dtype=float)
        X_before_prepared = X_before_prepared.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_before_selected = selector.transform(X_before_prepared)
    X_before_selected = pd.DataFrame(X_before_selected, columns=X_selected.columns)

    # ── 5. Train-Test Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_smote, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_smote
    )
    X_before_train, X_before_test, y_before_train, y_before_test = train_test_split(
        X_before_selected, y_before_smt, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_before_smt
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

    # Compare model performance before vs after SMOTE without altering the normal pipeline.
    smote_model_comparison = []
    for model_name, model in trained_models.items():
        before_model = clone(model)
        before_model.fit(X_before_train, y_before_train)
        before_metrics = evaluate_model(before_model, X_before_test, y_before_test, model_name=model_name)
        after_metrics = evaluate_model(model, X_test, y_test, model_name=model_name)

        accuracy_lift = float(after_metrics["accuracy"]) - float(before_metrics["accuracy"])
        f1_lift = float(after_metrics["f1"]) - float(before_metrics["f1"])
        roc_auc_lift = float(after_metrics["roc_auc"]) - float(before_metrics["roc_auc"])
        smote_model_comparison.append({
            "model_name": model_name,
            "before_accuracy": round(before_metrics["accuracy"], 4),
            "after_accuracy": round(after_metrics["accuracy"], 4),
            "accuracy_lift": round(accuracy_lift, 4),
            "accuracy_lift_pct": f"{accuracy_lift * 100:+.2f}%",
            "before_f1": round(before_metrics["f1"], 4),
            "after_f1": round(after_metrics["f1"], 4),
            "f1_lift": round(f1_lift, 4),
            "f1_lift_pct": f"{f1_lift * 100:+.2f}%",
            "before_roc_auc": round(before_metrics["roc_auc"], 4),
            "after_roc_auc": round(after_metrics["roc_auc"], 4),
            "roc_auc_lift": round(roc_auc_lift, 4),
            "roc_auc_lift_val": f"{roc_auc_lift:+.4f}",
            "before_recall": round(before_metrics["recall"], 4),
            "after_recall": round(after_metrics["recall"], 4),
            "recall_lift": round(float(after_metrics["recall"]) - float(before_metrics["recall"]), 4),
            "clinical_impact": "Positive SMOTE gain" if accuracy_lift >= 0 else "Requires review",
        })

    with open(os.path.join(artifact_dir, "smote_model_accuracy.json"), "w") as f:
        json.dump(smote_model_comparison, f, indent=2)

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
        "smote_model_comparison": smote_model_comparison,
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
