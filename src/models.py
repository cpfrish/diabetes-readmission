"""
Machine learning models module for diabetes readmission prediction.
Contains model definitions and training functions.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from xgboost import XGBClassifier

from . import config

# ============================================================================
# TRANSFORMER MODEL
# ============================================================================


def build_transformer_model(
    input_shape: int,
    num_classes: int = 2,
    embedding_dim: int = None,
    num_heads: int = None,
    key_dim: int = None,
    ffn_dim: int = None,
    dense_dim: int = None,
) -> keras.Model:
    """
    Build a transformer model for readmission prediction.

    Args:
        input_shape: Number of input features
        num_classes: Number of output classes
        embedding_dim: Dimension of embedding layer
        num_heads: Number of attention heads
        key_dim: Dimension of key in attention
        ffn_dim: Dimension of feed-forward network
        dense_dim: Dimension of final dense layer

    Returns:
        Compiled Keras model
    """
    # Use defaults from config if not provided
    if embedding_dim is None:
        embedding_dim = config.TRANSFORMER_PARAMS["embedding_dim"]
    if num_heads is None:
        num_heads = config.TRANSFORMER_PARAMS["num_heads"]
    if key_dim is None:
        key_dim = config.TRANSFORMER_PARAMS["key_dim"]
    if ffn_dim is None:
        ffn_dim = config.TRANSFORMER_PARAMS["ffn_dim"]
    if dense_dim is None:
        dense_dim = config.TRANSFORMER_PARAMS["dense_dim"]

    inputs = keras.Input(shape=(input_shape,))

    # Embedding layer
    x = keras.layers.Dense(embedding_dim, activation="relu")(inputs)
    x = keras.layers.Reshape((1, embedding_dim))(x)

    # Transformer block
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, x)
    x = keras.layers.Add()([x, attention_output])
    x = keras.layers.LayerNormalization()(x)

    # Feed-forward network
    ffn_output = keras.layers.Dense(ffn_dim, activation="relu")(x)
    ffn_output = keras.layers.Dense(embedding_dim)(ffn_output)
    x = keras.layers.Add()([x, ffn_output])
    x = keras.layers.LayerNormalization()(x)

    # Output layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(dense_dim, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_transformer_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_val: pd.DataFrame,
    Y_val: pd.Series,
    epochs: int = None,
    batch_size: int = None,
    verbose: int = 1,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train a transformer model for readmission prediction.

    Args:
        X_train: Training features
        Y_train: Training target
        X_val: Validation features
        Y_val: Validation target
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level (0, 1, or 2)

    Returns:
        Tuple of (trained model, training history)
    """
    if epochs is None:
        epochs = config.TRANSFORMER_PARAMS["epochs"]
    if batch_size is None:
        batch_size = config.TRANSFORMER_PARAMS["batch_size"]

    input_shape = X_train.shape[1]
    num_classes = len(np.unique(Y_train))

    model = build_transformer_model(input_shape, num_classes)

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train model
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=verbose,
    )

    return model, history


# ============================================================================
# BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================


def train_logistic_regression_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_val: pd.DataFrame,
    Y_val: pd.Series,
    tf_params: Dict[str, Any] = None,
    verbose: int = 1,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train a TensorFlow Keras-based logistic regression baseline.

    This builds a minimal model: Input -> Dense(1, activation='sigmoid'),
    compiles with binary_crossentropy, and trains using parameters from
    `config.LOGISTIC_TF_PARAMS` by default.

    Returns the trained Keras model and the training History object.
    """
    if tf_params is None:
        tf_params = config.LOGISTIC_TF_PARAMS

    epochs = tf_params.get("epochs", 50)
    batch_size = tf_params.get("batch_size", 64)
    learning_rate = tf_params.get("learning_rate", 0.001)
    optimizer_choice = tf_params.get("optimizer", "adam")

    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODEL: TENSORFLOW LOGISTIC REGRESSION")
    print("=" * 70)
    print(f"TF params: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, optimizer={optimizer_choice}")

    input_shape = X_train.shape[1]

    # Build model: a single sigmoid output unit
    model = keras.Sequential(
        [
            keras.Input(shape=(input_shape,)),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Choose optimizer
    if optimizer_choice.lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice.lower() == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        # Fallback to Adam for unknown names
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Evaluation
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)

    if verbose:
        print(f"Training accuracy: {train_acc:.2%}")
        print(f"Validation accuracy: {val_acc:.2%}")

    print("=" * 70)

    return model, history


# ============================================================================
# RANDOM FOREST MODEL
# ============================================================================


def train_random_forest_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_val: pd.DataFrame,
    Y_val: pd.Series,
    tuning_options: Dict[str, list] = None,
    n_iter: int = None,
    scoring: str = None,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    Train a Random Forest model with hyperparameter tuning.

    Args:
        X_train: Training features
        Y_train: Training target
        X_val: Validation features
        Y_val: Validation target
        tuning_options: Dictionary of hyperparameters to tune
        n_iter: Number of parameter settings to sample
        scoring: Scoring metric for evaluation
        verbose: Verbosity level

    Returns:
        Trained RandomizedSearchCV object
    """
    if tuning_options is None:
        tuning_options = config.RF_TUNING_OPTIONS
    if n_iter is None:
        n_iter = config.RANDOM_SEARCH_N_ITER
    if scoring is None:
        scoring = config.RANDOM_SEARCH_SCORING

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    print(f"Hyperparameter tuning options: {tuning_options}")
    print(f"Number of iterations: {n_iter}")
    print(f"Scoring metric: {scoring}")

    # Create and fit the model
    model = RandomizedSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        param_distributions=tuning_options,
        n_iter=n_iter,
        n_jobs=-1,
        scoring=scoring,
        verbose=verbose,
        random_state=config.RANDOM_STATE,
    )

    model.fit(X_train, Y_train)

    # Print results
    print(f"\nBest parameters: {model.best_params_}")
    print(f"Training accuracy: {model.score(X_train, Y_train):.2%}")
    print(f"Validation accuracy: {model.score(X_val, Y_val):.2%}")

    # Feature importance
    feature_importance = (
        pd.DataFrame(
            {
                "features": X_train.columns,
                "importance": model.best_estimator_.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 important features:")
    print(feature_importance.head(10))

    print("=" * 70)

    return model


# ============================================================================
# XGBOOST MODEL
# ============================================================================


def train_xgboost_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_val: pd.DataFrame,
    Y_val: pd.Series,
    tuning_options: Dict[str, list] = None,
    n_iter: int = None,
    scoring: str = None,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    Train an XGBoost model with hyperparameter tuning.

    Args:
        X_train: Training features
        Y_train: Training target
        X_val: Validation features
        Y_val: Validation target
        tuning_options: Dictionary of hyperparameters to tune
        n_iter: Number of parameter settings to sample
        scoring: Scoring metric for evaluation
        verbose: Verbosity level

    Returns:
        Trained RandomizedSearchCV object
    """
    if tuning_options is None:
        tuning_options = config.XGB_TUNING_OPTIONS
    if n_iter is None:
        n_iter = config.RANDOM_SEARCH_N_ITER
    if scoring is None:
        scoring = config.RANDOM_SEARCH_SCORING

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)
    print(f"Hyperparameter tuning options: {tuning_options}")
    print(f"Number of iterations: {n_iter}")
    print(f"Scoring metric: {scoring}")

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
    print(f"Scale pos weight (for class imbalance): {scale_pos_weight:.2f}")

    # Create and fit the model
    model = RandomizedSearchCV(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
        ),
        param_distributions=tuning_options,
        n_iter=n_iter,
        n_jobs=-1,
        scoring=scoring,
        verbose=verbose,
        random_state=config.RANDOM_STATE,
    )

    model.fit(X_train, Y_train)

    # Print results
    print(f"\nBest parameters: {model.best_params_}")
    print(f"Training accuracy: {model.score(X_train, Y_train):.2%}")
    print(f"Validation accuracy: {model.score(X_val, Y_val):.2%}")

    # Feature importance
    feature_importance = (
        pd.DataFrame(
            {
                "features": X_train.columns,
                "importance": model.best_estimator_.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 important features:")
    print(feature_importance.head(10))

    print("=" * 70)

    return model


# ============================================================================
# MODEL UTILITIES
# ============================================================================


def get_feature_importance(
    model: Any, feature_names: list, top_n: int = 10
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model (RandomForest, XGBoost, etc.)
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance
    """
    # Handle RandomizedSearchCV wrapper
    if hasattr(model, "best_estimator_"):
        estimator = model.best_estimator_
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df.head(top_n)


def compare_models(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_val: pd.DataFrame,
    Y_val: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compare multiple models on train, val, and test sets.

    Args:
        models_dict: Dictionary mapping model names to trained models
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        X_test, Y_test: Test data

    Returns:
        DataFrame with model comparison results
    """
    results = []

    for model_name, model in models_dict.items():
        # Handle Keras models differently
        if isinstance(model, keras.Model):
            train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        else:
            train_acc = model.score(X_train, Y_train)
            val_acc = model.score(X_val, Y_val)
            test_acc = model.score(X_test, Y_test)

        results.append(
            {
                "Model": model_name,
                "Train Accuracy": f"{train_acc:.4f}",
                "Val Accuracy": f"{val_acc:.4f}",
                "Test Accuracy": f"{test_acc:.4f}",
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Models module loaded successfully")
    print("Available models:")
    print("  - Transformer (build_transformer_model, train_transformer_model)")
    print("  - Random Forest (train_random_forest_model)")
    print("  - Logistic Regression (train_logistic_regression_model)")
    print("  - XGBoost (train_xgboost_model)")
    print("Utility functions:")
    print("  - get_feature_importance")
    print("  - compare_models")
