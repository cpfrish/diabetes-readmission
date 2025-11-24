# 📦 Data Setup Guide for Team Members

This guide explains how to set up the diabetes dataset on your local machine. **Data files are NOT tracked in Git** to avoid GitHub's file size limits (18MB+).

---

## 🎯 Quick Setup (Recommended)

### Step 1: Get the Data Files

Choose one option:

**Option A - Local Machine/Drive:**
- Google Drive
- Dropbox
- OneDrive
- USB drive

**Option B - Download from Source:**
Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

You need these two files:
- `diabetic_data.csv` (~18MB)
- `IDS_mapping.csv` (~1KB)

### Step 2: Place Files in data/ Directory

```bash
# Navigate to project
cd "/path/to/Final Project"

# Copy files to data/ directory
cp /path/to/download/diabetic_data.csv data/
cp /path/to/download/IDS_mapping.csv data/
```

### Step 3: Verify Setup

```bash
python setup_data.py
```

You should see:
```
✅ SUCCESS! All required data files are present.
```

**Done!** You're ready to run the analysis. 

---

---

## 🛠️ Alternative Setup Methods

### Method 1: Environment Variables


### Method 2: Create local_config.py

For a more permanent solution:

```bash
# Copy the template
cp src/local_config.py.example src/local_config.py

# Edit with your paths
nano src/local_config.py  # or use your favorite editor
```

Edit `src/local_config.py`:
```python
"""Local configuration - DO NOT COMMIT"""
import os

# Option 1: Use absolute paths
DIABETIC_DATA_CSV = "/Users/YourName/path/to/diabetic_data.csv"
IDS_MAPPING_CSV = "/Users/YourName/path/to/IDS_mapping.csv"

# Option 2: Use home directory relative paths
# DIABETIC_DATA_CSV = os.path.expanduser("~/Documents/diabetes-data/diabetic_data.csv")
# IDS_MAPPING_CSV = os.path.expanduser("~/Documents/diabetes-data/IDS_mapping.csv")
```

**Important:** `local_config.py` is in `.gitignore` and will NOT be committed!

---

## Verification Steps

### 1. Run the setup checker:
```bash
python setup_data.py
```

### 2. Test in Python:
```python
from src import config
import os

print("Checking data files...")
print(f"CSV: {os.path.exists(config.DIABETIC_DATA_CSV)}")
print(f"IDS: {os.path.exists(config.IDS_MAPPING_CSV)}")
```

### 3. Test the preprocessing pipeline:
```python
from src import data_processing as dp

X_train, X_val, X_test, Y_train, Y_val, Y_test, raw_df = dp.preprocess_pipeline()
print(f"Success! Loaded {len(raw_df)} patient records")
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



