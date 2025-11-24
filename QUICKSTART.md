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

3. **IMPORTANT: Set up your local data files**:
   
   **Data files are NOT in Git!** Each team member needs their own local copy.
   
   **Option A - Use the data/ directory (Recommended):**
   ```bash
   # Copy the data files from team shared drive or existing location
   cp /path/to/diabetic_data.csv data/
   cp /path/to/IDS_mapping.csv data/
   ```
   
   **Option B - Use environment variables:**
   ```bash
   # macOS/Linux
   export DIABETIC_DATA_CSV="/your/path/diabetic_data.csv"
   export IDS_MAPPING_CSV="/your/path/IDS_mapping.csv"
   ```
   
   **Option C - Create local_config.py:**
   ```bash
   cp src/local_config.py.example src/local_config.py
   # Then edit src/local_config.py with your paths
   ```
   
   See `data/README.md` for detailed instructions!

4. **Verify your setup**:
   ```bash
   python setup_data.py
   ```
   
   This will check if your data files are accessible.

5. **Test the complete pipeline**:
   Open `notebooks/main_analysis.ipynb` and run the first few cells.


## Git Workflow

### Starting work:
```bash
git pull origin main
git checkout -b feature/your-name-module-update
```

### While working:
```bash
# Make changes to your assigned module
git add src/your_module.py
git commit -m "Description of what you changed"
```

### When done:
```bash
git push origin feature/your-name-module-update
# Then create a Pull Request on GitHub
```

---

## Common Tasks

### Run the complete analysis:
```bash
jupyter notebook notebooks/main_analysis.ipynb
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

