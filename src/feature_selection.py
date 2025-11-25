"""
Feature selection module using Grey Wolf Optimizer (GWO).
"""

from typing import Tuple, List, Callable

import numpy as np
import pandas as pd
from mealpy import FloatVar, GWO
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


class GWOFeatureSelector:
    """
    Grey Wolf Optimizer for feature selection.
    
    This class uses the Grey Wolf Optimizer metaheuristic algorithm to select
    the most important features for a given classifier.
    """
    
    def __init__(
        self,
        estimator,
        n_features: int,
        scoring: str = "f1",
        n_wolves: int = 10,
        n_iterations: int = 20,
        cv_folds: int = 3,
        threshold: float = 0.5,
        random_state: int = 1234,
    ):
        """
        Initialize the GWO Feature Selector.
        
        Args:
            estimator: The machine learning model to use for evaluation
            n_features: Total number of features in the dataset
            scoring: Scoring metric for evaluation (default: "f1")
            n_wolves: Number of wolves (population size) in GWO
            n_iterations: Number of optimization iterations
            cv_folds: Number of cross-validation folds
            threshold: Threshold for binary feature selection (0-1)
            random_state: Random seed for reproducibility
        """
        self.estimator = estimator
        self.n_features = n_features
        self.scoring = scoring
        self.n_wolves = n_wolves
        self.n_iterations = n_iterations
        self.cv_folds = cv_folds
        self.threshold = threshold
        self.random_state = random_state
        
        self.selected_features_mask_ = None
        self.selected_features_idx_ = None
        self.best_score_ = None
        self.optimizer_ = None
        
    def _fitness_function(self, solution: np.ndarray) -> float:
        """
        Fitness function to evaluate feature subset.
        
        Args:
            solution: Binary array indicating selected features
            
        Returns:
            Negative F1 score (negative because mealpy minimizes)
        """
        # Convert continuous values to binary
        mask = solution > self.threshold
        
        # Ensure at least one feature is selected
        if not np.any(mask):
            return 1.0  # Return worst score
        
        # Select features based on mask
        X_selected = self.X_train[:, mask]
        
        # Evaluate using cross-validation
        try:
            scores = cross_val_score(
                self.estimator,
                X_selected,
                self.y_train,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=-1,
            )
            # Return negative score (mealpy minimizes)
            return -np.mean(scores)
        except Exception as e:
            # If evaluation fails, return worst score
            return 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GWOFeatureSelector":
        """
        Fit the GWO feature selector.
        
        Args:
            X: Training features (numpy array)
            y: Training target (numpy array)
            
        Returns:
            self
        """
        self.X_train = X
        self.y_train = y
        
        # Define the problem bounds (0 to 1 for each feature)
        bounds = FloatVar(lb=[0.0] * self.n_features, ub=[1.0] * self.n_features)
        
        # Initialize the GWO optimizer
        problem_dict = {
            "obj_func": self._fitness_function,
            "bounds": bounds,
            "minmax": "min",  # Minimize the negative score
            "log_to": None,  # Disable logging
        }
        
        # Create and run the optimizer
        self.optimizer_ = GWO.OriginalGWO(
            epoch=self.n_iterations,
            pop_size=self.n_wolves,
        )
        
        print(f"Starting GWO feature selection with {self.n_wolves} wolves and {self.n_iterations} iterations...")
        self.optimizer_.solve(problem_dict, seed=self.random_state)
        
        # Get the best solution
        best_solution = self.optimizer_.g_best.solution
        self.selected_features_mask_ = best_solution > self.threshold
        self.selected_features_idx_ = np.where(self.selected_features_mask_)[0]
        self.best_score_ = -self.optimizer_.g_best.target.fitness
        
        print(f"GWO completed. Selected {len(self.selected_features_idx_)} out of {self.n_features} features")
        print(f"Best CV {self.scoring} score: {self.best_score_:.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the dataset by selecting features.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with only selected features
        """
        if self.selected_features_mask_ is None:
            raise ValueError("Feature selector has not been fitted yet. Call fit() first.")
        
        return X[:, self.selected_features_mask_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the selector and transform the data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self, feature_names: List[str] = None) -> List:
        """
        Get the list of selected features.
        
        Args:
            feature_names: List of feature names (optional)
            
        Returns:
            List of selected feature names or indices
        """
        if self.selected_features_idx_ is None:
            raise ValueError("Feature selector has not been fitted yet. Call fit() first.")
        
        if feature_names is not None:
            return [feature_names[i] for i in self.selected_features_idx_]
        else:
            return self.selected_features_idx_.tolist()


def select_features_gwo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator,
    n_wolves: int = 10,
    n_iterations: int = 20,
    cv_folds: int = 3,
    threshold: float = 0.5,
    random_state: int = 1234,
) -> Tuple[GWOFeatureSelector, List[str]]:
    """
    Select features using Grey Wolf Optimizer.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        estimator: Machine learning model for evaluation
        n_wolves: Number of wolves in GWO population
        n_iterations: Number of optimization iterations
        cv_folds: Number of cross-validation folds
        threshold: Threshold for feature selection
        random_state: Random seed
        
    Returns:
        Tuple of (fitted selector, list of selected feature names)
    """
    feature_names = X_train.columns.tolist()
    
    # Initialize and fit the selector
    selector = GWOFeatureSelector(
        estimator=estimator,
        n_features=X_train.shape[1],
        scoring="f1",
        n_wolves=n_wolves,
        n_iterations=n_iterations,
        cv_folds=cv_folds,
        threshold=threshold,
        random_state=random_state,
    )
    
    selector.fit(X_train.values, y_train.values)
    selected_features = selector.get_selected_features(feature_names)
    
    return selector, selected_features


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    # Create estimator
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Select features
    selector, selected_features = select_features_gwo(
        X_df, y_series, estimator, n_wolves=5, n_iterations=10
    )
    
    print(f"\nSelected features: {selected_features}")
