import sys
import os

# Set up paths so that running from HeartSense root works seamlessly
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.ml.temporal_pipeline import run_temporal_training_pipeline

def main():
    result = run_temporal_training_pipeline()
    print("\n[SUCCESS] Temporal training pipeline completed successfully!")
    print(f"   Best model: {result['best_name']}")

if __name__ == "__main__":
    main()
