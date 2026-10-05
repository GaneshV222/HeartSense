"""
HeartSense – Training Entry Point

Runs the complete ML training pipeline:

    1. Load Cleveland Heart Disease Dataset
    2. Preprocess data
    3. Apply SMOTE (BEFORE train-test split)
    4. Feature selection (AFTER SMOTE)
    5. Train-test split
    6. Train 6 models with hyperparameter tuning
    7. Compare models
    8. Select best model (by ROC-AUC → Recall → F1)
    9. Evaluate on test data
   10. Save all artifacts

Usage:
    python train.py
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml.model_training import run_training_pipeline


def main():
    print("""
==============================================================
             HeartSense – Training Pipeline                  
                                                              
  Pipeline: Preprocess -> SMOTE -> Feature Select -> Split      
  Models:   XGBoost, Random Forest, Decision Tree,            
            SVM, KNN, Naive Bayes                             
  Best by:  ROC-AUC -> Recall -> F1                            
==============================================================
    """)

    result = run_training_pipeline()

    print("\n[SUCCESS] Training pipeline completed successfully!")
    print(f"   Best model: {result['best_name']}")
    print(f"   ROC-AUC:    {result['best_metrics']['roc_auc']:.4f}")
    print(f"   Recall:     {result['best_metrics']['recall']:.4f}")
    print(f"   F1-Score:   {result['best_metrics']['f1']:.4f}")
    print(f"\n   Run the app: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
