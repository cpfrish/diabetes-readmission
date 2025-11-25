"""
Example script demonstrating GWO feature selection with SMOTE Group K-Fold CV.

This script shows how to:
1. Load and preprocess data with patient groups preserved
2. Use Grey Wolf Optimizer (GWO) for feature selection
3. Apply SMOTE within Group K-Fold cross-validation
4. Evaluate model performance with proper handling of class imbalance
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.data_processing import preprocess_pipeline
from src.feature_selection import select_features_gwo
from src.cross_validation import evaluate_with_smote_group_kfold


def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("=" * 80)
    print("DIABETES READMISSION PREDICTION")
    print("GWO Feature Selection + SMOTE Group K-Fold Cross-Validation")
    print("Binary Classification: Readmitted <30 days vs. Not readmitted <30 days")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Load and preprocess data with patient groups
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Data Preprocessing (keeping patient groups)")
    print("=" * 80)
    
    result = preprocess_pipeline(
        filepath=None,  # Uses default from config
        random_state=config.RANDOM_STATE,
        keep_groups=True,
    )
    
    # Unpack results
    X_train, X_val, X_test, Y_train, Y_val, Y_test, groups_train, groups_val, groups_test, raw_df = result
    
    print(f"\nClass distribution in training set:")
    print(Y_train.value_counts())
    print(f"Class imbalance ratio: {Y_train.value_counts()[0] / Y_train.value_counts()[1]:.2f}:1")
    
    # ========================================================================
    # STEP 2: Feature selection using GWO
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Feature Selection using Grey Wolf Optimizer")
    print("=" * 80)
    
    # Initialize base estimator for feature selection
    base_estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    
    print(f"\nOriginal number of features: {X_train.shape[1]}")
    
    # Perform GWO feature selection
    # Note: Using smaller parameters for demonstration. Increase for better results.
    gwo_selector, selected_features = select_features_gwo(
        X_train=X_train,
        y_train=Y_train,
        estimator=base_estimator,
        n_wolves=10,  # Increase to 20-30 for better results
        n_iterations=20,  # Increase to 50-100 for better results
        cv_folds=3,
        threshold=0.5,
        random_state=config.RANDOM_STATE,
    )
    
    print(f"\nSelected {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    # Transform datasets using selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    # ========================================================================
    # STEP 3: Model evaluation with SMOTE Group K-Fold CV
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Model Evaluation with SMOTE Group K-Fold CV")
    print("=" * 80)
    
    # Combine train and validation for cross-validation
    X_train_val = pd.concat([X_train_selected, X_val_selected], axis=0)
    Y_train_val = pd.concat([Y_train, Y_val], axis=0)
    groups_train_val = pd.concat([groups_train, groups_val], axis=0)
    
    print(f"\nCombined train+val size: {X_train_val.shape}")
    print(f"Number of unique patients (groups): {groups_train_val.nunique()}")
    
    # ========================================================================
    # STEP 3a: Evaluate Random Forest with SMOTE Group K-Fold
    # ========================================================================
    print("\n" + "-" * 80)
    print("Random Forest with SMOTE Group K-Fold CV")
    print("-" * 80)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    
    rf_metrics, rf_fold_scores = evaluate_with_smote_group_kfold(
        X=X_train_val,
        y=Y_train_val,
        groups=groups_train_val,
        estimator=rf_model,
        n_splits=5,
        smote_sampling_strategy="auto",  # Balance to 1:1 ratio
        smote_k_neighbors=5,
        random_state=config.RANDOM_STATE,
        verbose=True,
    )
    
    # ========================================================================
    # STEP 3b: Evaluate XGBoost with SMOTE Group K-Fold
    # ========================================================================
    print("\n" + "-" * 80)
    print("XGBoost with SMOTE Group K-Fold CV")
    print("-" * 80)
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )
    
    xgb_metrics, xgb_fold_scores = evaluate_with_smote_group_kfold(
        X=X_train_val,
        y=Y_train_val,
        groups=groups_train_val,
        estimator=xgb_model,
        n_splits=5,
        smote_sampling_strategy="auto",
        smote_k_neighbors=5,
        random_state=config.RANDOM_STATE,
        verbose=True,
    )
    
    # ========================================================================
    # STEP 4: Final model training and test evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Final Model Training and Test Evaluation")
    print("=" * 80)
    
    # Choose the best model based on CV performance
    rf_f1 = rf_metrics.get("f1_mean", 0)
    xgb_f1 = xgb_metrics.get("f1_mean", 0)
    
    if rf_f1 > xgb_f1:
        print(f"\nRandom Forest performed better (F1: {rf_f1:.4f} vs {xgb_f1:.4f})")
        best_model = rf_model
        best_name = "Random Forest"
    else:
        print(f"\nXGBoost performed better (F1: {xgb_f1:.4f} vs {rf_f1:.4f})")
        best_model = xgb_model
        best_name = "XGBoost"
    
    # Apply SMOTE to full training data
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(
        sampling_strategy="auto",
        k_neighbors=5,
        random_state=config.RANDOM_STATE,
    )
    X_train_val_resampled, Y_train_val_resampled = smote.fit_resample(
        X_train_val, Y_train_val
    )
    
    print(f"\nTraining final {best_name} model on resampled data...")
    print(f"Training samples: {len(X_train_val_resampled)}")
    
    # Train final model
    final_model = best_model.__class__(**best_model.get_params())
    final_model.fit(X_train_val_resampled, Y_train_val_resampled)
    
    # Evaluate on test set
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_test_pred = final_model.predict(X_test_selected)
    
    print("\n" + "-" * 80)
    print("Test Set Performance")
    print("-" * 80)
    print("\nClassification Report:")
    print(classification_report(Y_test, y_test_pred, target_names=["No Readmission", "Readmission"]))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(Y_test, y_test_pred)
    print(f"                 Predicted")
    print(f"                 0       1")
    print(f"Actual 0     {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       1     {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Feature reduction: {(1 - len(selected_features)/X_train.shape[1])*100:.1f}%")
    
    print(f"\nCross-Validation Results:")
    print(f"  Random Forest F1: {rf_f1:.4f} (+/- {rf_metrics.get('f1_std', 0):.4f})")
    print(f"  XGBoost F1:       {xgb_f1:.4f} (+/- {xgb_metrics.get('f1_std', 0):.4f})")
    
    from sklearn.metrics import f1_score
    test_f1 = f1_score(Y_test, y_test_pred)
    print(f"\nTest Set F1 Score: {test_f1:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        "selected_features": selected_features,
        "gwo_selector": gwo_selector,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "final_model": final_model,
        "test_predictions": y_test_pred,
    }


if __name__ == "__main__":
    results = main()
