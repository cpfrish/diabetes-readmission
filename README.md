## Project README: Model Development and Analysis

This document outlines the contribution distribution and key files for our project. Our workflow is divided into four main Jupyter Notebooks (.ipynb), each managed by a team member, ensuring a clear and parallelized development process.

---

### Directory Structure
```
. 
├── notebooks/ 
│ ├── rahil_dp_baseline.ipynb 
│ ├── sriya_eda_comparison.ipynb 
│ ├── colin_viz_transformer.ipynb 
│ ├── terra_rf_xgb.ipynb 
├── results/ 
│ ├── train.csv 
│ ├── val.csv 
│ ├── test.csv 
│ ├── baseline.pkl 
│ ├── transformer.h5 
│ ├── rf.pkl 
│ ├── xgb.pkl 
│ ├── stats_baseline.csv
│ ├── stats_transformer.csv
│ ├── stats_rf_xgb.csv
│ ├── final_comparison.csv 
├── src/ (Python utility files, separated by model type) 
│ ├── rahil_data_processing.py 
│ ├── rahil_baseline.py
│ ├── colin_cv.py 
│ ├── colin_model_transformer.py 
│ ├── terra_model_rf.py 
│ ├── terra_model_xgb.py 
│ ├── sriya_eda.py 
│ ├── sriya_feature_selection.py
│ ├── sriya_evaluation.py
├── README.md 
└── requirements.txt
```

---

## Contribution Distribution & Workflow

The project is structured so that each notebook's "Run All" execution follows a logical step in the overall methodology.

### 1. Rahil: Data Processing & Logistic Baseline

**`notebooks/rahil_dp_baseline.ipynb`**

* **Primary Responsibility:** **Data Preparation** and establishing the **Baseline Performance**.
* **Workflow:**
    1.  Perform initial data cleaning and pre-processing.
    2.  Split the dataset into **Train, Validation, and Test** sets.
    3.  **Output:** Save the processed data splits to `/results/` as `train.csv`, `val.csv`, and `test.csv`.
    4.  Build, train, and evaluate the **Logistic Regression Baseline Model**.
    5.  **Output:** Save the trained baseline model to `/results/baseline.pkl`.
    6.  **Output:** Save the statistics from the baseline model to `/results/stats_baseline.csv`.

### 2. Sriya: EDA & Feature Analysis

**`notebooks/sriya_eda_comparison.ipynb`**

* **Primary Responsibility:** **Exploratory Data Analysis (EDA)** and **Model Comparisons**.
* **Workflow:**
    1.  **Input:** Load processed data splits from `/results/train.csv` and `/results/test.csv`.
    2.  Perform **EDA** to understand the output distributions and key statistics.
    3.  Perform **Feature Importance Analysis** across all trained models (Logistic Baseline, Transformer, Random Forest, XGBoost).
    4.  **Input:** Import all models (`*.pkl`, `*.h5`) from `/results`.
    5.  **Input:** Import all model statistics (`*.csv`) from `/results`.
    6.  Generate a final **model comparison table/results**.
    7.  **Output:** Save the final result comparison as `/results/final_comparison.csv`.

### 3. Colin: Visualization & Transformer Model

**`notebooks/colin_viz_transformer.ipynb`**

* **Primary Responsibility:** **Data Visualization** and developing the **Transformer Model**.
* **Workflow:**
    1.  Create compelling **data visualizations**.
    2.  **Input:** Load processed data splits from `/results/*.csv`.
    3.  Build, train, and evaluate the **Transformer Model**.
    4.  **Output:** Save the trained Transformer model to `/results/transformer.h5`.
    5.  **Output:** Save the statistics from the transformer model to `/results/stats_transformer.csv`.

### 4. Terra: Random Forest & XGBoost Models

**`notebooks/terra_rf_xgb.ipynb`**

* **Primary Responsibility:** Developing and tuning 2 models - **Random Forest** and **XGBoost**.
* **Workflow:**
    1.  **Input:** Load processed data splits from `/results/*.csv`.
    2.  Build, **tune**, and evaluate the **Random Forest (RF) Model**.
    3.  **Output:** Save the trained RF model to `/results/rf.pkl`.
    4.  Build, train, and evaluate the **XGBoost (XGB) Model**.
    5.  **Output:** Save the trained XGB model to `/results/xgb.pkl`.
    6.  **Output:** Save the statistics from the 2 models to `/results/stats_rf_xgb.csv`.

---

## Execution Order

For a full project run, please execute the notebooks in the following order:

1.  **`rahil_dp_baseline.ipynb`** (Creates data splits and baseline model)
2.  **`colin_viz_transformer.ipynb`** (Needs data splits)
3.  **`terra_rf_xgb.ipynb`** (Needs data splits)
4.  **`sriya_eda_comparison.ipynb`** (Needs data splits and all trained models)