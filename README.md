# Predicting Readmission Rate for Diabetic Patients Using Machine Learning

**DS 207 Final Project**

**Team Members:**
- Colin Frishberg (cpfrish@berkeley.edu)
- Terra Jiang (yjiang66@berkeley.edu)
- Sai Sriya Mudigonda (smudigonda@berkeley.edu)
- Rahil Sharma (rahilsharma@berkeley.edu)

---

##  Project Overview

This project implements machine learning models to predict hospital readmission for diabetic patients. The codebase is modularized to facilitate team collaboration on GitHub and easy integration into a research paper.

---

## Project Structure

```
DS_207_Final_Project/
├── src/                          # Core Python modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and constants
│   ├── data_processing.py       # Data loading and preprocessing
│   ├── eda.py                   # Exploratory data analysis
│   ├── models.py                # Machine learning models
│   └── evaluation.py            # Model evaluation functions
│
├── notebooks/                    # Jupyter notebooks
│   └── main_analysis.ipynb      # Main analysis notebook
│
├── data/                         # Data directory (files NOT in git)
│   ├── README.md                # Data setup instructions
│   ├── diabetic_data.csv        # ← You add this locally
│   └── IDS_mapping.csv          # ← You add this locally
│
├── results/                    # Output directory for results
│   ├── figures/                 # Generated plots
│   ├── models/                  # Saved models
│   └── reports/                 # Analysis reports
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

##  Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages (see Installation)

### Installation

1. **Clone the repository** (or pull latest changes):
   ```bash
   git clone <repository-url>
   cd "Final Project"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. ** CRITICAL: Set up data files** (NOT in git - each member does this):
   
   **Quick setup:**
   ```bash
   # Get data files from team shared drive or download from UCI
   # Then copy to data/ directory:
   cp /path/to/diabetic_data.csv data/
   cp /path/to/IDS_mapping.csv data/
   
   # Verify setup:
   python setup_data.py
   ```
   
    **See [DATA_SETUP.md](DATA_SETUP.md) for complete instructions!**
   
4. **Verify installation**:
   ```python
   # In Python or Jupyter
   from src import config, data_processing, eda, models, evaluation
   print("✓ All modules loaded successfully!")
   
   # Test data loading
   from src import data_processing as dp
   X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df = dp.preprocess_pipeline()
   print(f"✓ Data loaded: {len(raw_df)} patient records")
   ```

---

##  Module Descriptions

### 1. **`src/config.py`** - Configuration Management


Contains all configuration parameters, file paths, and hyperparameters:
- Data file paths
- Feature lists (continuous, categorical, medications)
- Preprocessing parameters
- Model hyperparameters
- Visualization settings

**Key Constants:**
- `CONTINUOUS_FEATURES`: List of continuous features
- `MEDICATION_COLUMNS`: List of medication features
- `RF_TUNING_OPTIONS`: Random Forest hyperparameters
- `XGB_TUNING_OPTIONS`: XGBoost hyperparameters

---

### 2. **`src/data_processing.py`** - Data Pipeline


Handles all data loading, cleaning, and preprocessing operations.

**Key Functions:**
- `load_data()`: Load raw CSV data
- `clean_data()`: Remove duplicates and unnecessary columns
- `encode_categorical_features()`: One-hot and ordinal encoding
- `split_train_val_test()`: Split data into train/val/test sets
- `standardize_continuous_features()`: MinMax scaling
- `preprocess_pipeline()`: **Complete preprocessing pipeline**

**Usage Example:**
```python
from src import data_processing as dp

# Run complete pipeline
X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df = dp.preprocess_pipeline()
```

---

### 3. **`src/eda.py`** - Exploratory Data Analysis
**Owner:** *Assign team member*

Creates visualizations and statistical summaries.

**Key Functions:**
- `create_kde_plots()`: Kernel density estimates by target class
- `create_box_plots()`: Box plots for feature distributions
- `create_violin_plots()`: Violin plots for feature distributions
- `create_correlation_heatmap()`: Feature correlation matrix
- `create_target_distribution()`: Target variable distribution
- `generate_eda_report()`: **Complete EDA with all visualizations**

**Usage Example:**
```python
from src import eda

# Generate comprehensive report
eda.generate_eda_report(
    X_train=X_train,
    Y_train=Y_train,
    raw_df=raw_df,
    save_plots=True  # Save to results/
)
```

---

### 4. **`src/models.py`** - Machine Learning Models


Implements and trains machine learning models.

**Implemented Models:**
1. **Transformer** (Neural Network)
2. **Random Forest** (Ensemble)
3. **XGBoost** (Gradient Boosting)

**Key Functions:**
- `train_random_forest_model()`: Train RF with hyperparameter tuning
- `train_xgboost_model()`: Train XGBoost with hyperparameter tuning
- `train_transformer_model()`: Train neural network transformer
- `get_feature_importance()`: Extract feature importance
- `compare_models()`: Compare multiple models

**Usage Example:**
```python
from src import models

# Train Random Forest
rf_model = models.train_random_forest_model(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val
)
```

---

### 5. **`src/evaluation.py`** - Model Evaluation

Comprehensive model evaluation and metrics.

**Key Functions:**
- `evaluate_model_performance()`: Calculate accuracy, precision, recall, F1
- `create_confusion_matrix_plot()`: Visualize confusion matrix
- `plot_roc_curve()`: ROC curve and AUC score
- `plot_precision_recall_curve()`: Precision-Recall curve
- `calculate_detailed_metrics()`: Detailed metrics from confusion matrix
- `evaluate_model_comprehensive()`: **Complete evaluation report**

**Usage Example:**
```python
from src import evaluation

# Comprehensive evaluation
metrics = evaluation.evaluate_model_comprehensive(
    model=rf_model,
    X_train=X_train, Y_train=Y_train,
    X_val=X_val, Y_val=Y_val,
    X_test=X_test, Y_test=Y_test,
    model_name="Random Forest",
    save_plots=True
)
```

---

### Git Workflow for Team Members

1. **Before starting work:**
   ```bash
   git pull origin main
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # Examples:
   # git checkout -b feature/improve-data-cleaning
   # git checkout -b feature/add-new-model
   ```

3. **Work on modules**

4. **Commit your changes:**
   ```bash
   git add src/your_module.py
   git commit -m "Brief description of changes"
   ```

5. **Push and create Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub for team review.



## Using the Main Notebook

The `notebooks/main_analysis.ipynb` notebook demonstrates the complete workflow:

1. **Setup and Imports** - Load all modules
2. **Data Preprocessing** - Run preprocessing pipeline
3. **Exploratory Analysis** - Generate visualizations
4. **Model Training** - Train multiple models
5. **Model Evaluation** - Compare performance
6. **Feature Importance** - Analyze important features
7. **Conclusions** - Summarize results

**To run the notebook:**
```bash
jupyter notebook notebooks/main_analysis.ipynb
```


---

## 📈 Creating Research Paper Content

The modular structure makes it easy to generate content for your research paper:

### Generating Figures
```python
# Save all EDA plots
eda.generate_eda_report(X_train, Y_train, raw_df, save_plots=True)

# Save evaluation plots
evaluation.evaluate_model_comprehensive(
    model, X_train, Y_train, X_val, Y_val, X_test, Y_test,
    save_plots=True
)
```

### Extracting Metrics
```python
# Compare all models
comparison_df = models.compare_models(models_dict, X_train, Y_train, ...)
comparison_df.to_csv('results/model_comparison.csv')

# Feature importance
importance_df = models.get_feature_importance(model, X_train.columns)
importance_df.to_csv('results/feature_importance.csv')
```

### Code Snippets for Paper
Each module function has clear documentation that can be referenced in your methodology section.


## Customization

### Adding New Models
Add a new training function to `src/models.py`:
```python
def train_your_model(X_train, Y_train, X_val, Y_val):
    # Your implementation
    pass
```

### Modifying Hyperparameters
Update `src/config.py`:
```python
RF_TUNING_OPTIONS = {
    "n_estimators": [100, 200, 300, 500],  # Added 500
    "max_depth": [6, 8, 12, 16, 20],       # Added 20
    # ...
}
```

---

## Dependencies

Create a `requirements.txt` file:
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
altair>=4.2.0
vegafusion[embed]>=1.5.0
vl-convert-python>=1.6.0
jupyter>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

##  Troubleshooting




---

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide/keras)
- [Altair Visualization](https://altair-viz.github.io/)
- [Git Collaboration Guide](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)


## License

This project is for educational purposes as part of DS 207 coursework.

---

