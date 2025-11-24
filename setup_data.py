#!/usr/bin/env python3
"""
Data setup utility for the diabetes readmission prediction project.

This script helps team members set up their local data files.
Run this after cloning the repository to verify your data setup.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import after path setup (noqa to ignore linter warning)
from src import config  # noqa: E402


def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    color = "\033[92m" if exists else "\033[91m"  # Green or Red
    reset = "\033[0m"

    print(f"{color}{status}{reset} {description}")
    print(f"   Path: {filepath}")

    if exists:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")

    return exists


def main():
    """Main setup verification function."""
    print("=" * 70)
    print("DIABETES READMISSION PREDICTION - DATA SETUP VERIFICATION")
    print("=" * 70)

    print("\n Project Structure:")
    print(f"   Project Root: {config.PROJECT_ROOT}")
    print(f"   Data Directory: {config.DATA_DIR}")
    print(f"   Results Directory: {config.RESULTS_DIR}")

    print("\n Data Files:")
    print("-" * 70)

    # Check data files
    csv_exists = check_file_exists(config.DIABETIC_DATA_CSV, "Diabetic Data CSV")
    print()

    ids_exists = check_file_exists(config.IDS_MAPPING_CSV, "IDS Mapping CSV")

    print("\n" + "=" * 70)

    # Overall status
    if csv_exists and ids_exists:
        print("\n All required data files are present.")
        print("\nYou're ready to run the analysis! Try:")
        print(
            '   python -c "from src import data_processing as dp; dp.preprocess_pipeline()"'
        )
        print("   or open notebooks/main_analysis.ipynb")
        return 0
    else:
        print("\n SETUP INCOMPLETE - Missing data files!")
        print("\n Setup Instructions:")
        print("   1. Read data/README.md for detailed setup instructions")
        print("   2. Download data files from the team shared drive or UCI repository")
        print("   3. Place files in the data/ directory:")
        print("      - diabetic_data.csv")
        print("      - IDS_mapping.csv")
        print(
            "\n   Alternative: Set environment variables or create src/local_config.py"
        )
        print("   See data/README.md for all options")
        return 1


if __name__ == "__main__":
    sys.exit(main())
