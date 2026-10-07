"""
HeartSense – Hyperparameter Tuning

Uses GridSearchCV to find the best hyperparameters for each model.
Tuning is performed exclusively on the Training Data.
"""

from sklearn.model_selection import GridSearchCV
from app.config import CV_FOLDS, RANDOM_STATE


# ── Parameter grids for each model ─────────────────────────────────────────────
PARAM_GRIDS = {
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
}


def tune_model(model, model_name: str, X_train, y_train, cv: int = CV_FOLDS):
    """
    Run GridSearchCV for a single model.

    Returns
    -------
    best_model : fitted estimator with best params
    result     : dict with tuning metadata
    """
    param_grid = PARAM_GRIDS.get(model_name, {})

    if not param_grid:
        # No grid defined – just fit the model directly
        model.fit(X_train, y_train)
        return model, {
            "model_name": model_name,
            "best_params": {},
            "best_cv_score": None,
            "param_grid": {},
        }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    result = {
        "model_name": model_name,
        "best_params": grid.best_params_,
        "best_cv_score": round(float(grid.best_score_), 4),
        "param_grid": param_grid,
    }

    print(
        f"[Tuning] {model_name}: CV ROC-AUC = {result['best_cv_score']:.4f}  "
        f"Params = {grid.best_params_}"
    )
    return grid.best_estimator_, result
