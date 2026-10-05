"""
Tests for model training, tuning, comparison, and best-model selection.
"""

import pytest
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.data_loader import load_dataset
from src.ml.preprocessing import clean_data, separate_features_target
from src.ml.smote import apply_smote
from src.ml.feature_selection import select_features
from src.ml.model_training import get_base_models
from src.ml.tuning import tune_model
from src.ml.evaluation import evaluate_model, evaluate_all_models, select_best_model
from src.config import RANDOM_STATE, MODEL_ARTIFACT_DIR


@pytest.fixture(scope="module")
def prepared_data():
    """Pipeline: Load → Clean → SMOTE → FeatureSelect → Split."""
    df = load_dataset()
    df = clean_data(df)
    X, y = separate_features_target(df)
    X_sm, y_sm, _ = apply_smote(X, y)
    X_sel, _, _ = select_features(X_sm, y_sm, k=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_sm, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


class TestModelTraining:
    """Tests for model training."""

    def test_base_models_created(self):
        models = get_base_models()
        assert len(models) == 6
        expected = {"XGBoost", "Random Forest", "Decision Tree", "SVM", "KNN", "Naive Bayes"}
        assert set(models.keys()) == expected

    def test_model_trains(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        models = get_base_models()
        # Train just one model (Naive Bayes is fastest)
        model = models["Naive Bayes"]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


class TestTuning:
    """Tests for hyperparameter tuning."""

    def test_tuning_returns_best_model(self, prepared_data):
        X_train, _, y_train, _ = prepared_data
        models = get_base_models()
        best_estimator, result = tune_model(models["Naive Bayes"], "Naive Bayes", X_train, y_train, cv=3)

        assert best_estimator is not None
        assert "model_name" in result
        assert "best_params" in result
        assert result["model_name"] == "Naive Bayes"


class TestEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_model_returns_all_metrics(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        models = get_base_models()
        model = models["Naive Bayes"]
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, model_name="Naive Bayes")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "specificity" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

        # All metrics should be between 0 and 1
        for key in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
            assert 0 <= metrics[key] <= 1, f"{key} = {metrics[key]} out of range"

    def test_evaluate_all_models(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        # Train 2 quick models
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier

        models = {
            "NB": GaussianNB().fit(X_train, y_train),
            "DT": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
        }
        comparison = evaluate_all_models(models, X_test, y_test)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_select_best_model(self, prepared_data):
        X_train, X_test, y_train, y_test = prepared_data
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier

        models = {
            "NB": GaussianNB().fit(X_train, y_train),
            "DT": DecisionTreeClassifier(random_state=42).fit(X_train, y_train),
        }
        comparison = evaluate_all_models(models, X_test, y_test)
        best_name, best_est, best_metrics = select_best_model(comparison, models)

        assert best_name in models
        assert best_est is not None
        assert "roc_auc" in best_metrics
