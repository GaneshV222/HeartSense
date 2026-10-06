import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
except Exception:
    pass

from app.ml.temporal_pipeline import run_temporal_training_pipeline

def main():
    print("Starting HeartSense True Longitudinal Temporal Pipeline...")
    result = run_temporal_training_pipeline()
    print("\n[SUCCESS] Temporal training pipeline completed successfully!")
    print(f"   Best model: {result['best_name']}")

if __name__ == "__main__":
    main()
