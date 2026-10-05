import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml.model_training import run_training_pipeline

def main():
    print("Starting Training Pipeline...")
    result = run_training_pipeline()
    print("\n[SUCCESS] Training pipeline completed successfully!")
    print(f"   Best model: {result['best_name']}")

if __name__ == "__main__":
    main()
