import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.ml.temporal_pipeline import run_temporal_training_pipeline

def main():
    print("Starting Temporal Training Pipeline...")
    result = run_temporal_training_pipeline()
    print("\n[SUCCESS] Temporal training pipeline completed successfully!")
    print(f"   Best model: {result['best_name']}")

if __name__ == "__main__":
    main()
