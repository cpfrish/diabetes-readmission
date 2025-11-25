"""
Cross-validation module with SMOTE for handling class imbalance.
Provides Group K-Fold CV with synthetic minority over-sampling.
"""

from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold


class SMOTEGroupKFoldCV:
    """
    Group K-Fold Cross-Validation with SMOTE applied within each fold.
    
    This class ensures that:
    1. Data from the same patient stays in the same fold (Group K-Fold)
    2. SMOTE is applied only on training folds to avoid data leakage
    3. Class imbalance is addressed through synthetic sampling
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        smote_sampling_strategy: str = "auto",
        smote_k_neighbors: int = 5,
        random_state: int = 1234,
    ):
        """
        Initialize the SMOTE Group K-Fold CV.
        
        Args:
            n_splits: Number of folds for cross-validation
            smote_sampling_strategy: SMOTE sampling strategy
                - "auto": resample only minority class
                - float: desired ratio of minority to majority class
                - dict: target number for each class
            smote_k_neighbors: Number of nearest neighbors for SMOTE
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.smote_sampling_strategy = smote_sampling_strategy
        self.smote_k_neighbors = smote_k_neighbors
        self.random_state = random_state
        
        self.results_ = None
        self.fold_scores_ = None
        
    def fit_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        estimator,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Perform Group K-Fold CV with SMOTE and evaluate the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            groups: Group identifiers (e.g., patient IDs)
            estimator: Machine learning model to evaluate
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing mean metrics across folds
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        
        fold_results = []
        
        if verbose:
            print(f"\nStarting {self.n_splits}-Fold Group K-Fold CV with SMOTE...")
            print("=" * 70)
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            if verbose:
                print(f"\nFold {fold_idx}/{self.n_splits}")
                print("-" * 70)
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            if verbose:
                print(f"Train size: {len(X_train_fold)}, Val size: {len(X_val_fold)}")
                print(f"Train class distribution: {y_train_fold.value_counts().to_dict()}")
            
            # Apply SMOTE to training data only
            try:
                smote = SMOTE(
                    sampling_strategy=self.smote_sampling_strategy,
                    k_neighbors=self.smote_k_neighbors,
                    random_state=self.random_state + fold_idx,
                )
                X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train_fold, y_train_fold
                )
                
                if verbose:
                    print(f"After SMOTE: {len(X_train_resampled)} samples")
                    print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
            except Exception as e:
                if verbose:
                    print(f"SMOTE failed: {e}. Using original training data.")
                X_train_resampled = X_train_fold
                y_train_resampled = y_train_fold
            
            # Train model
            model = clone(estimator)
            model.fit(X_train_resampled, y_train_resampled)
            
            # Predict on validation set
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                "fold": fold_idx,
                "accuracy": accuracy_score(y_val_fold, y_pred),
                "precision": precision_score(y_val_fold, y_pred, zero_division=0),
                "recall": recall_score(y_val_fold, y_pred, zero_division=0),
                "f1": f1_score(y_val_fold, y_pred, zero_division=0),
            }
            
            if y_pred_proba is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_val_fold, y_pred_proba)
                except Exception:
                    metrics["roc_auc"] = np.nan
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred).ravel()
            metrics.update({
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            })
            
            fold_results.append(metrics)
            
            if verbose:
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
                    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Store results
        self.fold_scores_ = pd.DataFrame(fold_results)
        
        # Calculate mean metrics
        mean_metrics = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in self.fold_scores_.columns:
                mean_metrics[f"{metric}_mean"] = self.fold_scores_[metric].mean()
                mean_metrics[f"{metric}_std"] = self.fold_scores_[metric].std()
        
        self.results_ = mean_metrics
        
        if verbose:
            print("\n" + "=" * 70)
            print("Cross-Validation Results")
            print("=" * 70)
            for metric, value in mean_metrics.items():
                if "mean" in metric:
                    std_key = metric.replace("mean", "std")
                    print(f"{metric}: {value:.4f} (+/- {mean_metrics[std_key]:.4f})")
        
        return mean_metrics
    
    def get_fold_scores(self) -> pd.DataFrame:
        """
        Get detailed scores for each fold.
        
        Returns:
            DataFrame with scores for each fold
        """
        if self.fold_scores_ is None:
            raise ValueError("Cross-validation has not been performed yet. Call fit_evaluate() first.")
        return self.fold_scores_


def evaluate_with_smote_group_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator,
    n_splits: int = 5,
    smote_sampling_strategy: str = "auto",
    smote_k_neighbors: int = 5,
    random_state: int = 1234,
    verbose: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Convenience function to perform Group K-Fold CV with SMOTE.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        groups: Group identifiers (e.g., patient IDs)
        estimator: Machine learning model to evaluate
        n_splits: Number of folds
        smote_sampling_strategy: SMOTE sampling strategy
        smote_k_neighbors: Number of neighbors for SMOTE
        random_state: Random seed
        verbose: Whether to print progress
        
    Returns:
        Tuple of (mean metrics dict, fold scores DataFrame)
    """
    cv = SMOTEGroupKFoldCV(
        n_splits=n_splits,
        smote_sampling_strategy=smote_sampling_strategy,
        smote_k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )
    
    mean_metrics = cv.fit_evaluate(X, y, groups, estimator, verbose=verbose)
    fold_scores = cv.get_fold_scores()
    
    return mean_metrics, fold_scores


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data with groups
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced
        random_state=42,
    )
    
    # Create groups (simulating patient IDs)
    groups = np.repeat(np.arange(200), 5)  # 200 patients, 5 encounters each
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    groups_series = pd.Series(groups)
    
    # Create estimator
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Evaluate with SMOTE and Group K-Fold
    mean_metrics, fold_scores = evaluate_with_smote_group_kfold(
        X_df, y_series, groups_series, estimator, n_splits=5
    )
    
    print("\n\nFold-wise scores:")
    print(fold_scores)
