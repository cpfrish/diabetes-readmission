# Quick Reference: GWO + SMOTE + Group K-Fold CV

## Important: Binary Classification

This implementation uses binary classification:
- **Class 0**: Not readmitted within 30 days (includes >30 days and NO)
- **Class 1**: Readmitted within 30 days (<30)

## Additional Features

The following features are kept (not dropped) from the original dataset:
- `patient_nbr`: Used for group-based cross-validation
- `admission_type_id`: Type of admission (one-hot encoded)
- `discharge_disposition_id`: Discharge destination (one-hot encoded)  
- `admission_source_id`: Admission source (one-hot encoded)
- `A1Cresult`: A1C test result (ordinal: None=0, Norm=1, >7=2, >8=3)

## Installation

```bash
pip install imbalanced-learn>=0.10.0 mealpy>=3.0.0
# OR
./install_gwo_smote.sh
```

## Quick Test

```bash
python test_gwo_smote.py
```

## Basic Usage

### 1. Data Preprocessing with Groups

```python
from src.data_processing import preprocess_pipeline

result = preprocess_pipeline(keep_groups=True)
X_train, X_val, X_test, Y_train, Y_val, Y_test, \
    groups_train, groups_val, groups_test, raw_df = result
```

### 2. GWO Feature Selection

```python
from src.feature_selection import select_features_gwo
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(n_estimators=50, random_state=42)
selector, selected_features = select_features_gwo(
    X_train, Y_train, estimator,
    n_wolves=10, n_iterations=20, cv_folds=3
)
X_train_selected = X_train[selected_features]
```

### 3. SMOTE Group K-Fold CV

```python
from src.cross_validation import evaluate_with_smote_group_kfold
import pandas as pd

X_combined = pd.concat([X_train_selected, X_val_selected])
Y_combined = pd.concat([Y_train, Y_val])
groups_combined = pd.concat([groups_train, groups_val])

metrics, fold_scores = evaluate_with_smote_group_kfold(
    X_combined, Y_combined, groups_combined,
    estimator, n_splits=5
)
```

## Parameter Tuning

| Purpose | n_wolves | n_iterations | Time |
|---------|----------|--------------|------|
| Quick test | 5 | 10 | ~2 min |
| Development | 10 | 20 | ~10 min |
| Production | 20-30 | 50-100 | ~1 hour |

## SMOTE Strategies

```python
# Balance to 1:1
smote_sampling_strategy="auto"

# Custom ratio (1:2)
smote_sampling_strategy=0.5

# More neighbors for sparse data
smote_k_neighbors=10
```

## Complete Example

```python
# 1. Load data
result = preprocess_pipeline(keep_groups=True)
X_train, X_val, X_test, Y_train, Y_val, Y_test, \
    groups_train, groups_val, groups_test, _ = result

# 2. Select features
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
selector, features = select_features_gwo(
    X_train, Y_train, estimator, n_wolves=10, n_iterations=20
)
X_train_sel = X_train[features]
X_val_sel = X_val[features]

# 3. Cross-validate
import pandas as pd
X_cv = pd.concat([X_train_sel, X_val_sel])
Y_cv = pd.concat([Y_train, Y_val])
groups_cv = pd.concat([groups_train, groups_val])

metrics, _ = evaluate_with_smote_group_kfold(
    X_cv, Y_cv, groups_cv, estimator, n_splits=5
)

# 4. Train final model
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_final, Y_final = smote.fit_resample(X_cv, Y_cv)
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_final, Y_final)

# 5. Evaluate
X_test_sel = X_test[features]
predictions = final_model.predict(X_test_sel)
```

## Common Issues

**GWO too slow**: Reduce `n_wolves` or `n_iterations`

**SMOTE error**: Reduce `k_neighbors` or `sampling_strategy`

**Not enough folds**: Reduce `n_splits` to match number of groups

**Import error**: Add project to path: `sys.path.insert(0, '/path/to/project')`

## Files

- `src/feature_selection.py` - GWO implementation
- `src/cross_validation.py` - SMOTE + Group K-Fold
- `example_gwo_smote_cv.py` - Complete example script
- `notebooks/gwo_smote_analysis.ipynb` - Interactive notebook
- `docs/GWO_SMOTE_GUIDE.md` - Full documentation
- `test_gwo_smote.py` - Test suite

## Resources


- Example: `example_gwo_smote_cv.py`
- Notebook: `notebooks/gwo_smote_analysis.ipynb`
