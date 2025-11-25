"""
Quick test to verify GWO + SMOTE implementation works correctly.
This uses synthetic data to ensure all components are functioning.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_selection import select_features_gwo, GWOFeatureSelector
from src.cross_validation import evaluate_with_smote_group_kfold, SMOTEGroupKFoldCV


def test_gwo_feature_selection():
    """Test GWO feature selection with synthetic data."""
    print("\n" + "=" * 70)
    print("TEST 1: GWO Feature Selection")
    print("=" * 70)
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Class distribution: {y_series.value_counts().to_dict()}")
    
    # Initialize estimator
    estimator = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    
    # Test GWO feature selection (with small parameters for speed)
    print("\nRunning GWO feature selection...")
    selector, selected_features = select_features_gwo(
        X_train=X_df,
        y_train=y_series,
        estimator=estimator,
        n_wolves=5,
        n_iterations=5,
        cv_folds=2,
        threshold=0.5,
        random_state=42,
    )
    
    print(f"\n✓ GWO completed successfully!")
    print(f"  - Selected {len(selected_features)} out of {X_df.shape[1]} features")
    print(f"  - Best F1 score: {selector.best_score_:.4f}")
    print(f"  - Selected features: {selected_features[:5]}...")
    
    return True


def test_smote_group_kfold():
    """Test SMOTE Group K-Fold cross-validation."""
    print("\n" + "=" * 70)
    print("TEST 2: SMOTE Group K-Fold Cross-Validation")
    print("=" * 70)
    
    # Create synthetic data with groups
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=8,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced
        random_state=42,
    )
    
    # Create groups (simulating patients)
    groups = np.repeat(np.arange(100), 5)  # 100 patients, 5 samples each
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    groups_series = pd.Series(groups)
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Class distribution: {y_series.value_counts().to_dict()}")
    print(f"Number of unique groups: {groups_series.nunique()}")
    
    # Initialize estimator
    estimator = RandomForestClassifier(n_estimators=20, random_state=42)
    
    # Test SMOTE Group K-Fold CV
    print("\nRunning SMOTE Group K-Fold CV...")
    mean_metrics, fold_scores = evaluate_with_smote_group_kfold(
        X=X_df,
        y=y_series,
        groups=groups_series,
        estimator=estimator,
        n_splits=3,
        smote_sampling_strategy="auto",
        smote_k_neighbors=3,
        random_state=42,
        verbose=False,  # Suppress detailed output
    )
    
    print(f"\n✓ SMOTE Group K-Fold CV completed successfully!")
    print(f"  - F1 Score: {mean_metrics['f1_mean']:.4f} (+/- {mean_metrics['f1_std']:.4f})")
    print(f"  - Accuracy:  {mean_metrics['accuracy_mean']:.4f} (+/- {mean_metrics['accuracy_std']:.4f})")
    print(f"  - Precision: {mean_metrics['precision_mean']:.4f} (+/- {mean_metrics['precision_std']:.4f})")
    print(f"  - Recall:    {mean_metrics['recall_mean']:.4f} (+/- {mean_metrics['recall_std']:.4f})")
    
    return True


def test_integrated_workflow():
    """Test the complete integrated workflow."""
    print("\n" + "=" * 70)
    print("TEST 3: Integrated Workflow (GWO + SMOTE + Group K-Fold)")
    print("=" * 70)
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=400,
        n_features=25,
        n_informative=12,
        n_classes=2,
        weights=[0.65, 0.35],
        random_state=42,
    )
    
    groups = np.repeat(np.arange(80), 5)  # 80 patients, 5 samples each
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    groups_series = pd.Series(groups)
    
    print(f"Initial dataset: {X_df.shape}")
    
    # Step 1: GWO Feature Selection
    print("\nStep 1: GWO Feature Selection...")
    estimator = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    
    selector, selected_features = select_features_gwo(
        X_train=X_df,
        y_train=y_series,
        estimator=estimator,
        n_wolves=5,
        n_iterations=5,
        cv_folds=2,
        random_state=42,
    )
    
    X_selected = X_df[selected_features]
    print(f"  ✓ Selected {len(selected_features)} features")
    
    # Step 2: SMOTE Group K-Fold CV
    print("\nStep 2: SMOTE Group K-Fold CV...")
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    
    mean_metrics, fold_scores = evaluate_with_smote_group_kfold(
        X=X_selected,
        y=y_series,
        groups=groups_series,
        estimator=model,
        n_splits=3,
        smote_sampling_strategy="auto",
        random_state=42,
        verbose=False,
    )
    
    print(f"  ✓ Cross-validation completed")
    print(f"    F1 Score: {mean_metrics['f1_mean']:.4f}")
    
    print(f"\n✓ Integrated workflow completed successfully!")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING TESTS FOR GWO + SMOTE IMPLEMENTATION")
    print("=" * 70)
    
    tests = [
        ("GWO Feature Selection", test_gwo_feature_selection),
        ("SMOTE Group K-Fold CV", test_smote_group_kfold),
        ("Integrated Workflow", test_integrated_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with error:")
            print(f"  {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("\nYou can now use:")
        print("  - python example_gwo_smote_cv.py")
        print("  - jupyter notebook notebooks/gwo_smote_analysis.ipynb")
    else:
        print("SOME TESTS FAILED! ✗")
        print("\nPlease check the error messages above.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
