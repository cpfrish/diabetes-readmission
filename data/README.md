# Data Directory

This directory should contain the diabetes dataset files. **Data files are NOT tracked in Git** to avoid GitHub's file size limits.

## 📦 Required Data Files

Place the following files in this directory:
- `diabetic_data.csv` - Main dataset (~18MB)
- `IDS_mapping.csv` - ID mapping file

## 🔽 How to Set Up Your Local Data

Each team member needs to set up their own local copy of the data. Choose one of the options below:

### Option 1: Download from Source (Recommended)

1. Download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) or from the shared team drive

2. Place the files in this directory:
   ```
   data/
   ├── diabetic_data.csv
   └── IDS_mapping.csv
   ```

### Option 2: Copy from Team Member

If a team member has the data:
```bash
# From their location
cd "/path/to/their/project/diabetes+130-us+hospitals+for+years+1999-2008"
cp diabetic_data.csv IDS_mapping.csv "/path/to/your/project/data/"
```

### Option 3: Use Custom Location with Environment Variable

If you prefer to keep data elsewhere on your machine:

**On macOS/Linux:**
```bash
export DIABETIC_DATA_CSV="/your/custom/path/diabetic_data.csv"
export IDS_MAPPING_CSV="/your/custom/path/IDS_mapping.csv"
```

**On Windows (PowerShell):**
```powershell
$env:DIABETIC_DATA_CSV = "C:\your\custom\path\diabetic_data.csv"
$env:IDS_MAPPING_CSV = "C:\your\custom\path\IDS_mapping.csv"
```

### Option 4: Create local_config.py

Create `src/local_config.py` (git-ignored) with your custom paths:

```python
"""
Local configuration - DO NOT COMMIT THIS FILE
Each team member can have their own data paths
"""
import os

# Customize these paths for your local machine
DIABETIC_DATA_CSV = "/Users/YourName/path/to/diabetic_data.csv"
IDS_MAPPING_CSV = "/Users/YourName/path/to/IDS_mapping.csv"
```

## ✅ Verify Your Setup

Test that your data is accessible:

```python
from src import config
import os

print("Data directory:", config.DATA_DIR)
print("Diabetic data CSV:", config.DIABETIC_DATA_CSV)
print("IDS mapping CSV:", config.IDS_MAPPING_CSV)

# Check if files exist
print("\nFile existence check:")
print("diabetic_data.csv exists:", os.path.exists(config.DIABETIC_DATA_CSV))
print("IDS_mapping.csv exists:", os.path.exists(config.IDS_MAPPING_CSV))
```

If both files exist, you're ready to go! 

## 📁 Directory Contents (Not Tracked)

This directory will contain (locally only):
```
data/
├── README.md              # This file (tracked in git)
├── .gitkeep              # Keeps directory in git (tracked)
├── diabetic_data.csv     # NOT tracked in git
└── IDS_mapping.csv       # NOT tracked in git
```

## Important Notes

1. **Never commit data files** - They are in `.gitignore`
2. **Each team member maintains their own local copy**
3. **Data files are ~18MB total** - too large for GitHub
4. **The code will work** as long as the files are in this directory or you've configured a custom path

## Troubleshooting

### Error: "FileNotFoundError: diabetic_data.csv not found"

**Solution:** Place the data files in the `data/` directory or set up environment variables/local_config.py

### Error: "No such file or directory: 'data'"

**Solution:** Make sure you're running code from the project root directory

### Need to share data with team?

Use Google Drive, Dropbox, or OneDrive to share the CSV files outside of Git.

---

**Last Updated:** November 23, 2025
