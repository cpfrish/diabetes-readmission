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


def total_medications_prescribed(row: pd.Series) -> int:
    """
    Calculate the total number of medications prescribed to a patient.

    Args:
        row: DataFrame row containing medication columns

    Returns:
        Total count of medications prescribed
    """
    total = 0
    for col in config.MEDICATION_COLUMNS:
        if col in row.index and row[col] != "No":
            total += 1
    return total


def add_total_medications_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a feature for total medications prescribed to each patient.

    Args:
        df: DataFrame with medication columns

    Returns:
        DataFrame with 'total_medications_prescribed' column added
    """
    df_copy = df.copy()
    df_copy["total_medications_prescribed"] = df_copy.apply(total_medications_prescribed, axis=1)
    df_copy["total_medications_prescribed"] = df_copy["total_medications_prescribed"].fillna(0).astype(int)
    return df_copy


def add_service_utilization_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add service utilization feature as the sum of outpatient, emergency, and inpatient visits.
    This matches the feature engineering approach from the study.

    Args:
        df: DataFrame with visit count columns

    Returns:
        DataFrame with 'service_utilization' column added
    """
    df_copy = df.copy()
    
    # Calculate service utilization (sum of all prior visits)
    df_copy["service_utilization"] = (
        df_copy["number_outpatient"].fillna(0) + 
        df_copy["number_emergency"].fillna(0) + 
        df_copy["number_inpatient"].fillna(0)
    )
    df_copy["service_utilization"] = df_copy["service_utilization"].astype(int)
    
    return df_copy


def diagnosis_to_category(diag_code: int) -> int:
    """
    Map diagnosis codes to broader categories.

    Args:
        diag_code: Diagnosis code as string

    Returns:
        Diagnosis category as int (1-9, or 0 for Unknown)
    """
    if pd.isna(diag_code):
        return 0  # Unknown
    try:
        code = float(diag_code)
        if 390 <= code <= 459 or code == 785:
            return 1  # Circulatory
        elif 460 <= code <= 519 or code == 786:
            return 2  # Respiratory
        elif 520 <= code <= 579 or code == 787:
            return 3  # Digestive
        elif 250 <= code < 251:
            return 4  # Diabetes
        elif 800 <= code <= 999:
            return 5  # Injury
        elif 710 <= code <= 739:
            return 6  # Musculoskeletal
        elif 580 <= code <= 629 or code == 788:
            return 7  # Genitourinary
        elif 140 <= code <= 239:
            return 8  # Neoplasms
        else:
            return 9  # Other
    except ValueError:
        if str(diag_code).startswith("V") or str(diag_code).startswith("E"):
            return 9  # Supplementary
        return 9  # Other


def add_diagnosis_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add diagnosis category features to the DataFrame.

    Args:
        df: DataFrame with diag_1, diag_2, diag_3 columns

    Returns:
        DataFrame with diagnosis category columns added
    """
    df_copy = df.copy()
    df_copy["diagnosis_1_category"] = df_copy["diag_1"].apply(diagnosis_to_category)
    df_copy["diagnosis_2_category"] = df_copy["diag_2"].apply(diagnosis_to_category)
    df_copy["diagnosis_3_category"] = df_copy["diag_3"].apply(diagnosis_to_category)
    # Cast as int
    df_copy["diagnosis_1_category"] = df_copy["diagnosis_1_category"].astype(int)
    df_copy["diagnosis_2_category"] = df_copy["diagnosis_2_category"].astype(int)
    df_copy["diagnosis_3_category"] = df_copy["diagnosis_3_category"].astype(int)
    return df_copy


def group_discharge_dispositions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group discharge dispositions into 3 main categories to match study.
    Categories: 1=Home, 2=Skilled Nursing Facility, 3=Home Health Service

    Args:
        df: DataFrame with 'discharge_disposition_id' column

    Returns:
        DataFrame with 'discharge_disposition_grouped' column
    """
    df_copy = df.copy()

    if "discharge_disposition_id" in df_copy.columns:
        # Simplified mapping based on study's 3 main categories
        mapping = {
            1: 1,   # Home
            6: 1,   # Home (hospice)
            8: 1,   # Home (left AMA - treat as home)
            2: 3,   # Home with home health service
            3: 2,   # Skilled nursing facility  
            4: 2,   # Intermediate care facility (SNF-like)
            5: 2,   # Another type of facility (SNF-like)
            7: 2,   # Medical facility (SNF-like)
            # Others (expired, etc.) will be marked as missing/other
        }
        df_copy["discharge_disposition_grouped"] = df_copy["discharge_disposition_id"].map(mapping)
        df_copy["discharge_disposition_grouped"] = df_copy["discharge_disposition_grouped"].fillna(1)  # Default to Home
        df_copy["discharge_disposition_grouped"] = df_copy["discharge_disposition_grouped"].astype(int)
        df_copy.drop(columns=["discharge_disposition_id"], inplace=True)

    return df_copy
def group_admission_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group admission types into 3 categories to match study.
    Categories: 1=Emergency, 2=Urgent, 3=Elective

    Args:
        df: DataFrame with 'admission_type_id' column

    Returns:
        DataFrame with 'admission_type_grouped' column
    """
    df_copy = df.copy()

    if "admission_type_id" in df_copy.columns:
        # Study uses: Emergency, Urgent, Elective
        mapping = {
            1: 1,  # Emergency
            2: 2,  # Urgent
            3: 3,  # Elective
            4: 1,  # Newborn (treat as emergency)
            5: 1,  # Not Available (default to emergency)
            6: 2,  # NULL (default to urgent)
            7: 1,  # Trauma Center (emergency)
            8: 1,  # Not Mapped (default to emergency)
        }
        df_copy["admission_type_grouped"] = df_copy["admission_type_id"].map(mapping)
        df_copy["admission_type_grouped"] = df_copy["admission_type_grouped"].fillna(1)  # Default to emergency
        df_copy["admission_type_grouped"] = df_copy["admission_type_grouped"].astype(int)
        df_copy.drop(columns=["admission_type_id"], inplace=True)

    return df_copy

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

    #Remove duplicate patients (keep first encounter)
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
    """
    Encode categorical features using one-hot encoding and ordinal encoding.
    
    Strategy:
    - Group high-cardinality features to prevent feature explosion
    - Use ordinal encoding for features with natural ordering
    - Use one-hot encoding only for low-cardinality categoricals
    - Result: ~100-150 features instead of 2,400+

    Args:
        df: DataFrame with categorical features

    Returns:
        DataFrame with encoded features
    """
    df_encoded = df.copy()

    # 0. Add total medications prescribed feature
    df_encoded = add_total_medications_feature(df_encoded)

    # 0.5. Add service utilization feature (sum of outpatient, emergency, inpatient visits)
    df_encoded = add_service_utilization_feature(df_encoded)

    # 1. Process age to continuous midpoint
    df_encoded["age_mid"] = df_encoded["age"].apply(age_to_midpoint)
    df_encoded.drop(columns=["age"], inplace=True)

    # 2. Group diagnosis codes into meaningful categories (717+749+790 → 10 categories each)

    df_encoded = add_diagnosis_categories(df_encoded)
    df_encoded.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)

    # 3. Group discharge dispositions (26 → 3 categories: Home, SNF, Home Health)
    df_encoded = group_discharge_dispositions(df_encoded)

    # 4. Group admission types (8 → 3 categories: Emergency, Urgent, Elective)
    df_encoded = group_admission_types(df_encoded)

    # 5. Handle medical_specialty (73 values → collapse to top 3 + Other like study)
    # Study focuses on: Internal Medicine, Family/General Practice, Surgery-General
    if "medical_specialty" in df_encoded.columns:
        top_specialties = df_encoded["medical_specialty"].value_counts().nlargest(3).index
        df_encoded["medical_specialty"] = df_encoded["medical_specialty"].where(
            df_encoded["medical_specialty"].isin(top_specialties), other="Other"
        )

    # 6. Collapse race categories (keep top 4 + Other)
    df_encoded = collapse_race_categories(df_encoded)

    # 7. One-hot encode LOW-cardinality categorical features
    categorical_to_encode = [
        "race_collapsed",                # ~5 values
        "gender",                        # 3 values (Male/Female/Unknown)
        "admission_type_grouped",        # 3 values (Emergency/Urgent/Elective)
        "discharge_disposition_grouped", # 3 values (Home/SNF/Home Health)
        "medical_specialty",             # 4 values (Top 3 + Other)
        "diagnosis_1_category",          # 10 values (after categorization)
        "diagnosis_2_category",          # 10 values (after categorization)
        "diagnosis_3_category",          # 10 values (after categorization)
    ]
    
    cols_to_encode = [col for col in categorical_to_encode if col in df_encoded.columns]
    if cols_to_encode:
        df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, dtype=int)

    # 8. Ordinal encode A1Cresult (has natural ordering: None < Norm < >7 < >8)
    if "A1Cresult" in df_encoded.columns:
        a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
        df_encoded["A1Cresult"] = df_encoded["A1Cresult"].map(a1c_map).fillna(0)

    # 9. Ordinal encode medication columns (has natural ordering: No < Down < Steady < Up)
    for col in config.MEDICATION_COLUMNS:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(config.PRESCRIPTION_MAP)

    # 10. Binary encode change and diabetesMed
    df_encoded["change"] = np.where(df_encoded["change"] == "No", 0, 1)
    df_encoded["diabetesMed"] = np.where(df_encoded["diabetesMed"] == "No", 0, 1)

    # 11. Encode target variable (binary: <30 days = 1, else = 0)
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

    # Encode features
    print("\n3. Encoding categorical features...")
    df_encoded = encode_categorical_features(df_clean)

    # Shuffle data
    print("\n4. Shuffling data...")
    df_shuffled = shuffle_data(df_encoded, random_state)

    # Split features and target
    print("\n5. Splitting features and target...")
    if keep_groups:
        X, Y, groups = split_features_target(df_shuffled, return_groups=True)
        print(f"Features: {X.shape}, Target: {Y.shape}, Groups: {groups.shape}")
    else:
        X, Y = split_features_target(df_shuffled, return_groups=False)
        print(f"Features: {X.shape}, Target: {Y.shape}")

    # Split into train/val/test
    print("\n6. Splitting into train/val/test...")
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
    print("\n7. Standardizing continuous features...")
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
