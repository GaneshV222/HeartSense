"""
HeartSense – Model Evaluation

Computes all required metrics on the held-out Test Data and generates
Matplotlib visualisations (confusion matrix, ROC curve).
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from app.config import MODEL_ARTIFACT_DIR


def _specificity(y_true, y_pred) -> float:
    """Calculate specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """
    Evaluate a trained model against test data.

    Returns a dict with accuracy, precision, recall, specificity, f1, roc_auc.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC (not all models support predict_proba)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except (AttributeError, IndexError):
        try:
            y_decision = model.decision_function(X_test)
            roc_auc = float(roc_auc_score(y_test, y_decision))
            y_proba = y_decision  # store for ROC curve
        except AttributeError:
            roc_auc = float(roc_auc_score(y_test, y_pred))
            y_proba = y_pred

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "specificity": round(float(_specificity(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
    }
    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every model in *models* (name → estimator) and return a
    comparison DataFrame sorted by the best-model criteria.
    """
    rows = []
    for name, model in models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append(m)
    df = pd.DataFrame(rows)
    df.sort_values(by=["roc_auc", "recall", "f1"], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def select_best_model(comparison_df: pd.DataFrame, models: dict):
    """
    Select the best model according to the priority rule:
      ROC-AUC (primary) → Recall (secondary) → F1 (tertiary)

    Returns (best_name, best_estimator, best_metrics_dict).
    """
    # comparison_df is already sorted by the criteria
    best_row = comparison_df.iloc[0]
    best_name = best_row["model_name"]
    best_estimator = models[best_name]
    best_metrics = best_row.to_dict()
    print(f"[Evaluation] Best model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.4f})")
    return best_name, best_estimator, best_metrics


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name: str = ""):
    """Return a Matplotlib Figure of the confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["No Disease (0)", "Disease (1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Actual",
        xlabel="Predicted",
        title=f"Confusion Matrix – {model_name}",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=18,
            )
    fig.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, model_name: str = ""):
    """Return a Matplotlib Figure of the ROC curve."""
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            y_proba = model.decision_function(X_test)
        except AttributeError:
            y_proba = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame):
    """Return a grouped bar chart comparing all models across metrics."""
    metrics_cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    df_plot = comparison_df.set_index("model_name")[metrics_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Persistence ────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, comparison_df: pd.DataFrame, artifact_dir: str = MODEL_ARTIFACT_DIR):
    """Save evaluation artefacts to disk."""
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    comparison_df.to_csv(os.path.join(artifact_dir, "model_comparison.csv"), index=False)
    print(f"[Evaluation] Metrics saved to {artifact_dir}")


def load_metrics(artifact_dir: str = MODEL_ARTIFACT_DIR) -> dict:
    """Load the best-model metrics from JSON."""
    path = os.path.join(artifact_dir, "model_metrics.json")
    with open(path, "r") as f:
        return json.load(f)


def load_comparison(artifact_dir: str = MODEL_ARTIFACT_DIR) -> pd.DataFrame:
    """Load the model comparison CSV."""
    path = os.path.join(artifact_dir, "model_comparison.csv")
    return pd.read_csv(path)
