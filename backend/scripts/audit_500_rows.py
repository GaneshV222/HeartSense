import json
import pandas as pd
import numpy as np
from pathlib import Path

batch_path = Path('E:/HeartSense/backend/artifacts/incremental_learning/imports/il-9e8b62a8cd4cf7302a5b.csv')
base_path = Path('E:/HeartSense/backend/data/heart_disease_prediction_2026.csv')

print("Loading batch CSV:", batch_path)
batch = pd.read_csv(batch_path)
print("Batch shape:", batch.shape)

print("Loading baseline CSV:", base_path)
base = pd.read_csv(base_path)
print("Base shape:", base.shape)

print("\n--- 1. IDENTIFIERS AND OVERLAP ---")
batch_pids = set(batch['patient_id'])
base_pids = set(base['patient_id'])
overlap = batch_pids.intersection(base_pids)
print(f"Unique patients in batch: {len(batch_pids)}")
print(f"Unique patients in baseline: {len(base_pids)}")
print(f"Patient overlap between batch and baseline: {len(overlap)}")

# Check duplicate visits within batch
dup_visits = batch.duplicated(subset=['patient_id', 'visit_number']).sum()
print(f"Duplicate (patient_id, visit_number) in batch: {dup_visits}")
print(f"Visit numbers in batch: {batch['visit_number'].value_counts().to_dict()}")

print("\n--- 2. CLASS DISTRIBUTION ---")
print(f"Baseline label counts: {base['prediction'].value_counts().to_dict()} (Ratio: {base['prediction'].value_counts(normalize=True).to_dict()})")
print(f"Batch label counts: {batch['prediction'].value_counts().to_dict()} (Ratio: {batch['prediction'].value_counts(normalize=True).to_dict()})")

print("\n--- 3. MISSING VALUES ---")
null_counts = batch.isnull().sum()
if (null_counts > 0).any():
    print(null_counts[null_counts > 0])
else:
    print("Zero missing values in batch raw columns.")

print("\n--- 4. FEATURE DISTRIBUTIONS (Mean +/- Std) ---")
numerical_cols = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'resting_heart_rate',
    'max_heart_rate', 'cholesterol', 'hdl', 'ldl', 'hba1c', 'oldpeak'
]
for col in numerical_cols:
    b_mean, b_std = base[col].mean(), base[col].std()
    m_mean, m_std = batch[col].mean(), batch[col].std()
    print(f"  {col:20s}: Base={b_mean:6.2f} +/- {b_std:5.2f} | Batch={m_mean:6.2f} +/- {m_std:5.2f} | Delta={m_mean - b_mean:+6.2f}")

print("\n--- 5. CATEGORICAL DISTRIBUTIONS ---")
cat_cols = ['gender', 'chest_pain_type', 'resting_ecg', 'st_slope', 'thalassemia', 'smoking', 'exercise_angina', 'diabetes', 'fasting_blood_sugar', 'family_history', 'physical_activity']
for col in cat_cols:
    b_dist = base[col].value_counts(normalize=True).to_dict()
    m_dist = batch[col].value_counts(normalize=True).to_dict()
    print(f"\n  {col}:")
    for k in sorted(set(list(b_dist.keys()) + list(m_dist.keys()))):
        print(f"    {str(k):20s}: Base={b_dist.get(k, 0):.3f} | Batch={m_dist.get(k, 0):.3f}")
