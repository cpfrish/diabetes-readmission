"""
Model evaluation module for diabetes readmission prediction.
Contains functions for evaluating model performance and creating reports.
"""

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorflow import keras

from . import config


def evaluate_model_performance(
    model: Any, X: pd.DataFrame, Y: pd.Series, dataset_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate a model's performance on a dataset.

    Args:
        model: Trained model
        X: Features
        Y: True labels
        dataset_name: Name of the dataset (for display)

    Returns:
        Dictionary with performance metrics
    """
    # Get predictions
    if isinstance(model, keras.Model):
        Y_pred_proba = model.predict(X, verbose=0)
        Y_pred = np.argmax(Y_pred_proba, axis=1)
    else:
        # Handle RandomizedSearchCV wrapper
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_
        else:
            estimator = model
        Y_pred = estimator.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(Y, Y_pred)
    precision = precision_score(Y, Y_pred, zero_division=0)
    recall = recall_score(Y, Y_pred, zero_division=0)
    f1 = f1_score(Y, Y_pred, zero_division=0)

    metrics = {
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Sample Size": len(Y),
    }

    return metrics


def create_confusion_matrix_plot(
    Y_true: pd.Series,
    Y_pred: np.ndarray,
    labels: list = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Create a confusion matrix plot.

    Args:
        Y_true: True labels
        Y_pred: Predicted labels
        labels: Label names for display
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if labels is None:
        labels = ["No Readmission", "Readmitted"]

    cm = confusion_matrix(Y_true, Y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(False)

    plt.tight_layout()
    return fig, cm


def print_classification_report(
    Y_true: pd.Series, Y_pred: np.ndarray, labels: list = None
):
    """
    Print a detailed classification report.

    Args:
        Y_true: True labels
        Y_pred: Predicted labels
        labels: Label names for display
    """
    if labels is None:
        labels = ["No Readmission", "Readmitted"]

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(Y_true, Y_pred, target_names=labels))
    print("=" * 70)


def calculate_detailed_metrics(
    Y_true: pd.Series, Y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate detailed metrics from confusion matrix.

    Args:
        Y_true: True labels
        Y_pred: Predicted labels

    Returns:
        Dictionary with detailed metrics
    """
    cm = confusion_matrix(Y_true, Y_pred)

    # Extract confusion matrix components
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "True Positives": int(tp),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
    }

    return metrics


def plot_roc_curve(
    model: Any,
    X: pd.DataFrame,
    Y: pd.Series,
    title: str = "ROC Curve",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curve for a binary classifier.

    Args:
        model: Trained model
        X: Features
        Y: True labels
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure and AUC score
    """
    # Get prediction probabilities
    if isinstance(model, keras.Model):
        Y_pred_proba = model.predict(X, verbose=0)[:, 1]
    else:
        # Handle RandomizedSearchCV wrapper
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_
        else:
            estimator = model
        Y_pred_proba = estimator.predict_proba(X)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(Y, Y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, roc_auc


def plot_precision_recall_curve(
    model: Any,
    X: pd.DataFrame,
    Y: pd.Series,
    title: str = "Precision-Recall Curve",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot precision-recall curve for a binary classifier.

    Args:
        model: Trained model
        X: Features
        Y: True labels
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get prediction probabilities
    if isinstance(model, keras.Model):
        Y_pred_proba = model.predict(X, verbose=0)[:, 1]
    else:
        # Handle RandomizedSearchCV wrapper
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_
        else:
            estimator = model
        Y_pred_proba = estimator.predict_proba(X)[:, 1]

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(Y, Y_pred_proba)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color="blue", lw=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def get_predictions(model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions and prediction probabilities from a model.

    Args:
        model: Trained model
        X: Features

    Returns:
        Tuple of (predictions, prediction_probabilities)
    """
    if isinstance(model, keras.Model):
        Y_pred_proba = model.predict(X, verbose=0)
        Y_pred = np.argmax(Y_pred_proba, axis=1)
    else:
        # Handle RandomizedSearchCV wrapper
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_
        else:
            estimator = model
        Y_pred = estimator.predict(X)
        Y_pred_proba = estimator.predict_proba(X)

    return Y_pred, Y_pred_proba


def evaluate_model_comprehensive(
    model: Any,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_val: pd.DataFrame,
    Y_val: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    model_name: str = "Model",
    save_plots: bool = False,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Perform comprehensive evaluation of a model on train, val, and test sets.

    Args:
        model: Trained model
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data
        model_name: Name of the model (for display and saving)
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots

    Returns:
        DataFrame with metrics for all datasets
    """
    import os

    if output_dir is None:
        output_dir = config.RESULTS_DIR

    print("\n" + "=" * 70)
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print("=" * 70)

    # Evaluate on all datasets
    train_metrics = evaluate_model_performance(model, X_train, Y_train, "Train")
    val_metrics = evaluate_model_performance(model, X_val, Y_val, "Validation")
    test_metrics = evaluate_model_performance(model, X_test, Y_test, "Test")

    # Create comparison DataFrame
    metrics_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])

    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(metrics_df.to_string(index=False))

    # Get test set predictions
    Y_test_pred, Y_test_proba = get_predictions(model, X_test)

    # Print classification report
    print_classification_report(Y_test, Y_test_pred)

    # Detailed metrics
    detailed_metrics = calculate_detailed_metrics(Y_test, Y_test_pred)
    print("\n" + "-" * 70)
    print("DETAILED TEST SET METRICS")
    print("-" * 70)
    for metric, value in detailed_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Confusion Matrix
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    fig_cm, cm = create_confusion_matrix_plot(
        Y_test, Y_test_pred, title=f"{model_name} - Confusion Matrix (Test Set)"
    )
    if save_plots:
        fig_cm.savefig(
            os.path.join(
                output_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    # ROC Curve
    print("\n" + "-" * 70)
    print("ROC CURVE")
    print("-" * 70)
    try:
        fig_roc, roc_auc = plot_roc_curve(
            model, X_test, Y_test, title=f"{model_name} - ROC Curve (Test Set)"
        )
        print(f"AUC Score: {roc_auc:.4f}")
        if save_plots:
            fig_roc.savefig(
                os.path.join(
                    output_dir, f"{model_name.replace(' ', '_')}_roc_curve.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

    # Precision-Recall Curve
    print("\n" + "-" * 70)
    print("PRECISION-RECALL CURVE")
    print("-" * 70)
    try:
        fig_pr = plot_precision_recall_curve(
            model,
            X_test,
            Y_test,
            title=f"{model_name} - Precision-Recall Curve (Test Set)",
        )
        if save_plots:
            fig_pr.savefig(
                os.path.join(
                    output_dir, f"{model_name.replace(' ', '_')}_pr_curve.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
    except Exception as e:
        print(f"Could not plot Precision-Recall curve: {e}")

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {model_name}")
    print("=" * 70)

    return metrics_df


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Available functions:")
    print("  - evaluate_model_performance")
    print("  - create_confusion_matrix_plot")
    print("  - print_classification_report")
    print("  - calculate_detailed_metrics")
    print("  - plot_roc_curve")
    print("  - plot_precision_recall_curve")
    print("  - evaluate_model_comprehensive")
