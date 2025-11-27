"""
Data processing module for diabetes readmission prediction.
Contains functions for loading, cleaning, preprocessing, and splitting data.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from . import config
except ImportError:
    import config


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


def clean_data(df: pd.DataFrame, keep_patient_id: bool = False) -> pd.DataFrame:
    """
    Clean the raw data by removing duplicates and unnecessary columns.

    Args:
        df: Raw DataFrame
        keep_patient_id: If True, preserve patient_nbr for group-based operations

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Remove duplicate patients (keep first encounter)
    df_clean = df_clean.drop_duplicates(subset=["patient_nbr"], keep="first")
    print(f"After removing duplicates: {df_clean.shape}")

    # Drop unnecessary columns (excluding patient_nbr if requested)
    columns_to_drop = config.COLUMNS_TO_DROP.copy() if isinstance(config.COLUMNS_TO_DROP, list) else list(config.COLUMNS_TO_DROP)
    
    # If keep_patient_id is False but patient_nbr is not in the drop list, add it
    if not keep_patient_id and "patient_nbr" not in columns_to_drop:
        columns_to_drop.append("patient_nbr")
    # If keep_patient_id is True and patient_nbr is in the drop list, remove it
    elif keep_patient_id and "patient_nbr" in columns_to_drop:
        columns_to_drop.remove("patient_nbr")
    
    df_clean = df_clean.drop(columns=columns_to_drop, axis=1)
    print(f"After dropping unnecessary columns: {df_clean.shape}")

    return df_clean


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
# creating indicator variables for the categorical features (one-hot encoding)
    df_encoded = df.copy()
    # Convert age buckets to midpoints
    df_encoded['age_mid'] = df_encoded['age'].apply(age_to_midpoint)
    df_encoded = df_encoded.drop(columns=['age'], axis=1)
    # Create indicator variables for categorical features
    df_encoded = pd.get_dummies(df_encoded, columns=['race_collapsed', 'gender'], dtype=int)

    # Encode ordinal values according to the prescription status of the visit
    prescription_map = {'No': 0,       # The drug was not prescribed
                        'Down': 1,     # The dosage was decreased
                        'Steady': 2,   # The dosage did not change
                        'Up': 3}       # The dosage was increased during the encounter
    cat_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'examide',
                'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    for cat_col in cat_columns:
        df_encoded[cat_col] = df_encoded[cat_col].map(prescription_map)

    # recoding change and diabetesMed to 0 for No and 1 to Yes
    df_encoded['change'] = np.where(df_encoded['change'] == 'No', 0, 1)
    df_encoded['diabetesMed'] = np.where(df_encoded['diabetesMed'] == 'No', 0, 1)
    
    # mapping the target into three classes
    readmission_map = {
        'NO': 0,  # no readmission recorded
        '>30': 1, # readmitted (over 30 days)
        '<30': 1  # readmitted (within 30 days)
    }
    df_encoded['readmitted'] = df_encoded['readmitted'].map(readmission_map)
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


def split_features_target(
    df: pd.DataFrame, return_groups: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split DataFrame into features (X), target (Y), and optionally groups.

    Args:
        df: DataFrame containing both features and target
        return_groups: If True, return patient_nbr as groups

    Returns:
        Tuple of (X, Y) or (X, Y, groups) where X is features, Y is target,
        and groups is patient identifiers
    """
    # Extract groups if requested
    if return_groups and "patient_nbr" in df.columns:
        groups = df["patient_nbr"].copy()
        X = df.drop(columns=["readmitted", "patient_nbr"], axis=1)
    else:
        groups = None
        X = df.drop(columns=["readmitted"], axis=1)
    
    Y = df["readmitted"]
    
    if return_groups:
        return X, Y, groups
    else:
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
    filepath: str = None, random_state: int = None, keep_groups: bool = False
) -> Tuple:
    """
    Complete preprocessing pipeline from raw data to train/val/test splits.

    Args:
        filepath: Path to raw data CSV
        random_state: Random seed for reproducibility
        keep_groups: If True, preserve patient_nbr for group-based operations

    Returns:
        Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df)
        or (X_train, X_val, X_test, Y_train, Y_val, Y_test, groups_train, 
            groups_val, groups_test, raw_df) if keep_groups=True
    """
    print("=" * 70)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    raw_df = load_data(filepath)

    # Clean data
    print("\n2. Cleaning data...")
    df_clean = clean_data(raw_df, keep_patient_id=keep_groups)

    # Collapse race categories
    print("\n3. Collapsing race categories...")
    df_collapsed = collapse_race_categories(df_clean)

    # Encode features
    print("\n4. Encoding categorical features...")
    df_encoded = encode_categorical_features(df_collapsed)

    # Shuffle data
    print("\n5. Shuffling data...")
    df_shuffled = shuffle_data(df_encoded, random_state)

    # Split features and target
    print("\n6. Splitting features and target...")
    if keep_groups:
        X, Y, groups = split_features_target(df_shuffled, return_groups=True)
        print(f"Features: {X.shape}, Target: {Y.shape}, Groups: {groups.shape}")
    else:
        X, Y = split_features_target(df_shuffled, return_groups=False)
        print(f"Features: {X.shape}, Target: {Y.shape}")

    # Split into train/val/test
    print("\n7. Splitting into train/val/test...")
    if keep_groups:
        # Split groups along with features and target
        X_train, X_test, Y_train, Y_test, groups_train, groups_test = train_test_split(
            X, Y, groups, train_size=config.TRAIN_SIZE, random_state=random_state
        )
        X_train, X_val, Y_train, Y_val, groups_train, groups_val = train_test_split(
            X_train, Y_train, groups_train, train_size=config.VAL_SIZE, random_state=random_state
        )
        print(f"Train set: {X_train.shape}")
        print(f"Val set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = split_train_val_test(
            X, Y, random_state=random_state
        )

    # Standardize continuous features
    print("\n8. Standardizing continuous features...")
    X_train, X_val, X_test, scaler = standardize_continuous_features(
        X_train, X_val, X_test
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    if keep_groups:
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, groups_train, groups_val, groups_test, raw_df
    else:
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df = preprocess_pipeline()
    print("\nPreprocessing test successful!")
