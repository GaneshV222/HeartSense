import argparse
import sys
import os

# Set up paths so that running from HeartSense root works seamlessly
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.ml.temporal_pipeline import run_temporal_training_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Train the verified HeartSense SMOTE-before-split model."
    )
    parser.add_argument("--dataset")
    parser.add_argument("--temporal-dataset")
    parser.add_argument(
        "--import-database",
        action="store_true",
        help="Replace only source-scoped PostgreSQL rows after validation.",
    )
    parser.add_argument(
        "--store-smote-database",
        action="store_true",
        help="Store the complete SMOTE-before-split training records in PostgreSQL.",
    )
    parser.add_argument("--backup-path")
    parser.add_argument("--expected-database")
    args = parser.parse_args()
    if args.import_database and not (args.backup_path and args.expected_database):
        parser.error("--import-database requires --backup-path and --expected-database.")
    if args.store_smote_database and not args.expected_database:
        parser.error("--store-smote-database requires --expected-database.")
    if args.backup_path and not args.import_database:
        parser.error("--backup-path is only valid with --import-database.")
    if args.expected_database and not (args.import_database or args.store_smote_database):
        parser.error("--expected-database is only valid with --import-database or --store-smote-database.")
    result = run_temporal_training_pipeline(
        args.dataset,
        args.temporal_dataset,
        import_database=args.import_database,
        backup_path=args.backup_path,
        expected_database=args.expected_database,
        store_smote_database=args.store_smote_database,
    )
    print("\n[SUCCESS] Temporal training pipeline completed successfully!")
    print(f"   Best model: {result['best_name']}")

if __name__ == "__main__":
    main()
