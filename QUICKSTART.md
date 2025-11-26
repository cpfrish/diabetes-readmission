# Quick Start Guide

## For Team Members

### First Time Setup

1. **Clone/Pull the repository**:
   ```bash
   cd "your/project/directory"
   git pull origin main
   ```

2. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your local data files**:
   
   **Data files are NOT in Git!** to avoid GitHub's file size limits (18MB+). Each team member needs their own local copy.
   
   **Use the data/ directory:**
   ```bash
   # Copy the data files from team shared drive or existing location
   cp /path/to/diabetic_data.csv data/
   cp /path/to/IDS_mapping.csv data/
   ```

   See `data/README.md` for detailed instructions!

4. **Verify your setup**:
   ```bash
   python setup_data.py
   ```

   You should see:
   ```
   All required data files are present.
   ```

   This will check if your data files are accessible.

---

## Common Tasks

### Test in Python:
```python
from src import config
import os

print("Checking data files...")
print(f"CSV: {os.path.exists(config.DIABETIC_DATA_CSV)}")
print(f"IDS: {os.path.exists(config.IDS_MAPPING_CSV)}")
```

### Test the preprocessing pipeline:
```python
from src import data_processing as dp

X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df = dp.preprocess_pipeline()
print(f"Success! Loaded {len(raw_df)} patient records")
```

### Test a single module:
```bash
python -m src.data_processing
python -m src.models
# etc.
```

### Save results for the paper:
```python
# In the notebook, set save_plots=True
eda.generate_eda_report(X_train, Y_train, raw_df, save_plots=True)
evaluation.evaluate_model_comprehensive(..., save_plots=True)
```

---

## For the Research Paper

### Get all figures:
```python
# Run this in the notebook
eda.generate_eda_report(X_train, Y_train, raw_df, save_plots=True)
evaluation.evaluate_model_comprehensive(rf_model, ..., save_plots=True)
evaluation.evaluate_model_comprehensive(xgb_model, ..., save_plots=True)
```

Results will be in `results/`

### Get metrics tables:
```python
# Model comparison
comparison_df = models.compare_models(models_dict, X_train, Y_train, ...)
comparison_df.to_csv('results/model_comparison.csv')

# Feature importance
importance = models.get_feature_importance(model, X_train.columns)
importance.to_csv('results/feature_importance.csv')
```

---

## Directory Structure

After setup, your project should look like:

```
Final Project/
├── data/                      # ← Your local data (NOT in git)
│   ├── README.md             # (tracked)
│   ├── .gitkeep              # (tracked)
│   ├── diabetic_data.csv     # (NOT tracked - you add this)
│   └── IDS_mapping.csv       # (NOT tracked - you add this)
├── src/
│   ├── config.py
│   ├── local_config.py       # (Optional, NOT tracked)
│   └── ...
├── setup_data.py
├── migrate_data.py
└── ...
```