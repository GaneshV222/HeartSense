from backend.app.config import *  # noqa: F401,F403

# Legacy compatibility layer used by the repository's historical tests and import
# contract. The canonical backend stores the newer 2026 dataset schema, but the
# legacy ML tests expect the classic Cleveland feature layout and a target named
# "target".
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_NAME = "target"
