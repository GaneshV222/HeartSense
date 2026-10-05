"""
Tests for the data preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.data_loader import load_dataset
from src.ml.preprocessing import clean_data, separate_features_target, inspect_dataset
from src.config import FEATURE_NAMES, TARGET_NAME


@pytest.fixture
def raw_dataset():
    """Load the raw Cleveland dataset."""
    return load_dataset()


@pytest.fixture
def clean_dataset(raw_dataset):
    """Return the cleaned dataset."""
    return clean_data(raw_dataset)


class TestDataLoader:
    """Tests for dataset loading."""

    def test_dataset_loads(self, raw_dataset):
        assert raw_dataset is not None
        assert isinstance(raw_dataset, pd.DataFrame)
        assert len(raw_dataset) > 0

    def test_dataset_has_expected_columns(self, raw_dataset):
        expected = FEATURE_NAMES + [TARGET_NAME]
        for col in expected:
            assert col in raw_dataset.columns, f"Missing column: {col}"

    def test_dataset_has_correct_shape(self, raw_dataset):
        assert raw_dataset.shape[1] == len(FEATURE_NAMES) + 1  # features + target


class TestPreprocessing:
    """Tests for data preprocessing."""

    def test_clean_data_returns_dataframe(self, clean_dataset):
        assert isinstance(clean_dataset, pd.DataFrame)

    def test_no_missing_values_after_cleaning(self, clean_dataset):
        assert clean_dataset.isnull().sum().sum() == 0

    def test_target_is_binary(self, clean_dataset):
        unique_vals = clean_dataset[TARGET_NAME].unique()
        assert set(unique_vals).issubset({0, 1})

    def test_no_duplicate_rows(self, clean_dataset):
        assert clean_dataset.duplicated().sum() == 0

    def test_numerical_values_clipped(self, clean_dataset):
        if "trestbps" in clean_dataset.columns:
            assert clean_dataset["trestbps"].min() >= 50
            assert clean_dataset["trestbps"].max() <= 250

    def test_separate_features_target(self, clean_dataset):
        X, y = separate_features_target(clean_dataset)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert TARGET_NAME not in X.columns
        assert len(X) == len(y)

    def test_inspect_dataset(self, raw_dataset):
        info = inspect_dataset(raw_dataset)
        assert "shape" in info
        assert "missing" in info
        assert "dtypes" in info


class TestSmote:
    """Tests for SMOTE application."""

    def test_smote_balances_classes(self, clean_dataset):
        from src.ml.smote import apply_smote

        X, y = separate_features_target(clean_dataset)
        X_res, y_res, info = apply_smote(X, y)

        # After SMOTE, classes should be balanced
        from collections import Counter
        counts = Counter(y_res)
        assert counts[0] == counts[1], "Classes should be balanced after SMOTE"

    def test_smote_preserves_columns(self, clean_dataset):
        from src.ml.smote import apply_smote

        X, y = separate_features_target(clean_dataset)
        X_res, y_res, info = apply_smote(X, y)

        assert list(X_res.columns) == list(X.columns)

    def test_smote_returns_info(self, clean_dataset):
        from src.ml.smote import apply_smote

        X, y = separate_features_target(clean_dataset)
        _, _, info = apply_smote(X, y)

        assert "before" in info
        assert "after" in info
        assert "samples_before" in info
        assert "samples_after" in info
        assert info["samples_after"] >= info["samples_before"]


class TestFeatureSelection:
    """Tests for feature selection."""

    def test_feature_selection_reduces_or_keeps_features(self, clean_dataset):
        from src.ml.smote import apply_smote
        from src.ml.feature_selection import select_features

        X, y = separate_features_target(clean_dataset)
        X_sm, y_sm, _ = apply_smote(X, y)
        X_sel, selector, info = select_features(X_sm, y_sm, k=8)

        assert X_sel.shape[1] <= X_sm.shape[1]
        assert X_sel.shape[1] == 8

    def test_feature_selection_returns_info(self, clean_dataset):
        from src.ml.smote import apply_smote
        from src.ml.feature_selection import select_features

        X, y = separate_features_target(clean_dataset)
        X_sm, y_sm, _ = apply_smote(X, y)
        _, _, info = select_features(X_sm, y_sm, k=8)

        assert "selected_features" in info
        assert "removed_features" in info
        assert "feature_scores" in info
        assert len(info["selected_features"]) == 8


class TestTrainTestSplit:
    """Tests for train-test split (verifying SMOTE → FS → Split order)."""

    def test_split_after_smote_and_feature_selection(self, clean_dataset):
        from sklearn.model_selection import train_test_split
        from src.ml.smote import apply_smote
        from src.ml.feature_selection import select_features

        X, y = separate_features_target(clean_dataset)

        # 1. SMOTE
        X_sm, y_sm, _ = apply_smote(X, y)

        # 2. Feature Selection
        X_sel, _, _ = select_features(X_sm, y_sm, k=10)

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y_sm, test_size=0.2, random_state=42
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == len(X_sel)
        assert X_train.shape[1] == X_sel.shape[1]
