"""
Configuration file for the diabetes readmission prediction project.
Contains all constants, file paths, and hyperparameters.
"""

import os

# ============================================================================
# FILE PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Data files - Each team member should have their own local copy
# Option 1: Use environment variable (recommended)
DIABETIC_DATA_CSV = os.environ.get(
    "DIABETIC_DATA_CSV", os.path.join(DATA_DIR, "diabetic_data.csv")
)
IDS_MAPPING_CSV = os.environ.get(
    "IDS_MAPPING_CSV", os.path.join(DATA_DIR, "IDS_mapping.csv")
)

# Option 2: Override in local_config.py (git-ignored)
try:
    from .local_config import DIABETIC_DATA_CSV as _LOCAL_DATA_CSV
    from .local_config import IDS_MAPPING_CSV as _LOCAL_IDS_CSV

    DIABETIC_DATA_CSV = _LOCAL_DATA_CSV
    IDS_MAPPING_CSV = _LOCAL_IDS_CSV
except ImportError:
    pass  # Use defaults above

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Columns to drop during preprocessing
COLUMNS_TO_DROP = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
]

# Continuous features to standardize
CONTINUOUS_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age_mid",
]

# Medication columns (for ordinal encoding)
MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Prescription status mapping
PRESCRIPTION_MAP = {
    "No": 0,  # The drug was not prescribed
    "Down": 1,  # The dosage was decreased
    "Steady": 2,  # The dosage did not change
    "Up": 3,  # The dosage was increased
}

# Readmission mapping
READMISSION_MAP = {
    "NO": 0,  # no readmission recorded
    ">30": 1,  # readmitted (over 30 days)
    "<30": 1,  # readmitted (within 30 days)
}

# Train/Val/Test split parameters
TRAIN_SIZE = 0.8
VAL_SIZE = 0.75  # Of the remaining 20% after train split
RANDOM_STATE = 1234

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random Forest hyperparameter tuning options
RF_TUNING_OPTIONS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 12, 16],
    "min_samples_split": [2, 5, 10],
}

# XGBoost hyperparameter tuning options
XGB_TUNING_OPTIONS = {
    "n_estimators": [30, 50, 100, 200, 300, 500],
    "max_depth": [3, 6, 8, 12, 16],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.4],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
    "min_child_weight": [1, 3, 5, 7],
}

# Transformer model parameters
TRANSFORMER_PARAMS = {
    "embedding_dim": 192,
    "num_heads": 8,
    "key_dim": 24,
    "ffn_dim": 2048,
    "dense_dim": 64,
    "num_classes": 2,
    "epochs": 10,
    "batch_size": 64,
}

# RandomizedSearchCV parameters
RANDOM_SEARCH_N_ITER = 30
RANDOM_SEARCH_SCORING = "f1"

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Number of columns for grid layouts in plots
PLOT_NCOLS = 4
PLOT_WIDTH = 300
PLOT_HEIGHT = 200

# Color schemes
COLOR_SCHEME = "category20"
