"""
Data processing module for diabetes readmission prediction.
Contains functions for loading, cleaning, preprocessing, and splitting data.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import config


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the diabetic patient data from CSV file.

    Args:
        filepath: Path to the CSV file. If None, uses default from config.

    Returns:
        DataFrame containing the raw data
    """
    if filepath is None:
        filepath = config.DIABETIC_DATA_CSV

    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df


def age_to_midpoint(age_bucket: str) -> float:
    """
    Convert age buckets like "[70-80)" to numeric midpoint (75).

    Args:
        age_bucket: String representing age range

    Returns:
        Numeric midpoint of the age range
    """
    if pd.isna(age_bucket):
        return np.nan
    try:
        lo, hi = age_bucket.strip("[]").split("-")
        lo = int(lo)
        hi = int(hi.strip(")"))
        return (lo + hi) / 2
    except Exception:
        return np.nan


def collapse_race_categories(df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """
    Collapse low-count race categories into 'Other'.

    Args:
        df: DataFrame with 'race' column
        top_n: Number of top race categories to keep separate

    Returns:
        DataFrame with 'race_collapsed' column
    """
    df_copy = df.copy()

    if "race" in df_copy.columns:
        df_copy["race"] = df_copy["race"].fillna("Unknown")
        top_races = df_copy["race"].value_counts().nlargest(top_n).index
        df_copy["race_collapsed"] = df_copy["race"].where(
            df_copy["race"].isin(top_races), other="Other"
        )
        df_copy.drop(columns=["race"], inplace=True)

    return df_copy


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by removing duplicates and unnecessary columns.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Remove duplicate patients (keep first encounter)
    df_clean = df_clean.drop_duplicates(subset=["patient_nbr"], keep="first")
    print(f"After removing duplicates: {df_clean.shape}")

    # Drop unnecessary columns
    df_clean = df_clean.drop(columns=config.COLUMNS_TO_DROP, axis=1)
    print(f"After dropping unnecessary columns: {df_clean.shape}")

    return df_clean


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding and ordinal encoding.

    Args:
        df: DataFrame with categorical features

    Returns:
        DataFrame with encoded features
    """
    df_encoded = df.copy()

    # Process age
    df_encoded["age_mid"] = df_encoded["age"].apply(age_to_midpoint)
    df_encoded.drop(columns=["age"], inplace=True)

    # Collapse race categories
    df_encoded = collapse_race_categories(df_encoded)

    # One-hot encode race and gender
    df_encoded = pd.get_dummies(
        df_encoded, columns=["race_collapsed", "gender"], dtype=int
    )

    # Ordinal encode medication columns
    for col in config.MEDICATION_COLUMNS:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(config.PRESCRIPTION_MAP)

    # Encode binary features
    df_encoded["change"] = np.where(df_encoded["change"] == "No", 0, 1)
    df_encoded["diabetesMed"] = np.where(df_encoded["diabetesMed"] == "No", 0, 1)

    # Encode target variable
    df_encoded["readmitted"] = df_encoded["readmitted"].map(config.READMISSION_MAP)

    return df_encoded


def shuffle_data(df: pd.DataFrame, random_state: int = None) -> pd.DataFrame:
    """
    Randomly shuffle the DataFrame.

    Args:
        df: Input DataFrame
        random_state: Random seed for reproducibility

    Returns:
        Shuffled DataFrame
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    indices = np.arange(len(df))
    shuffled_indices = np.random.RandomState(random_state).permutation(indices)
    return df.iloc[shuffled_indices].reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (Y).

    Args:
        df: DataFrame containing both features and target

    Returns:
        Tuple of (X, Y) where X is features and Y is target
    """
    X = df.drop(columns=["readmitted"], axis=1)
    Y = df["readmitted"]
    return X, Y


def split_train_val_test(
    X: pd.DataFrame,
    Y: pd.Series,
    train_size: float = None,
    val_size: float = None,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Feature DataFrame
        Y: Target Series
        train_size: Proportion of data for training (default from config)
        val_size: Proportion of remaining data for validation (default from config)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    if train_size is None:
        train_size = config.TRAIN_SIZE
    if val_size is None:
        val_size = config.VAL_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    # First split: train+val vs test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size, random_state=random_state
    )

    # Second split: train vs val
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, train_size=val_size, random_state=random_state
    )

    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def standardize_continuous_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    features: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Standardize continuous features using MinMaxScaler.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        features: List of feature names to standardize (default from config)

    Returns:
        Tuple of (X_train, X_val, X_test, scaler)
    """
    if features is None:
        features = config.CONTINUOUS_FEATURES

    # Create copies to avoid modifying originals
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    X_train_scaled[features] = scaler.fit_transform(X_train[features])
    X_val_scaled[features] = scaler.transform(X_val[features])
    X_test_scaled[features] = scaler.transform(X_test[features])

    print(f"Standardized {len(features)} continuous features")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    filepath: str = None, random_state: int = None
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """
    Complete preprocessing pipeline from raw data to train/val/test splits.

    Args:
        filepath: Path to raw data CSV
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df)
    """
    print("=" * 70)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    raw_df = load_data(filepath)

    # Clean data
    print("\n2. Cleaning data...")
    df_clean = clean_data(raw_df)

    # Encode features
    print("\n3. Encoding categorical features...")
    df_encoded = encode_categorical_features(df_clean)

    # Shuffle data
    print("\n4. Shuffling data...")
    df_shuffled = shuffle_data(df_encoded, random_state)

    # Split features and target
    print("\n5. Splitting features and target...")
    X, Y = split_features_target(df_shuffled)
    print(f"Features: {X.shape}, Target: {Y.shape}")

    # Split into train/val/test
    print("\n6. Splitting into train/val/test...")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_train_val_test(
        X, Y, random_state=random_state
    )

    # Standardize continuous features
    print("\n7. Standardizing continuous features...")
    X_train, X_val, X_test, scaler = standardize_continuous_features(
        X_train, X_val, X_test
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df = preprocess_pipeline()
    print("\nPreprocessing test successful!")
