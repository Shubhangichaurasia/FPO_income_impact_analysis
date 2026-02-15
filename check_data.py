import pandas as pd
from pathlib import Path

# --- 1. Load the CLEANED data from your Desktop ---
try:
    desktop_path = Path.home() / "Desktop"
    df_clean = pd.read_csv(desktop_path / 'fpo_cleaned_data.csv')
    print("✅ Successfully loaded the cleaned data.")

except FileNotFoundError:
    print("❌ ERROR: The file 'fpo_cleaned_data.csv' was not found on your Desktop.")
    print("Please make sure you have run the data cleaning script successfully.")
    exit()

# --- 2. Check for missing values ---
print("\n--- Checking for Missing Values ---")
print("Total missing values after cleaning:", df_clean.isnull().sum().sum())
print("✅ This should be 0, confirming imputation worked.")

# --- 3. Check for outliers (by inspecting min/max values) ---
print("\n--- Inspecting Key Columns ---")
print("Crop Income - Max Value (Outliers should be capped):", round(df_clean['Crop_Income'].max(), 2))
print("Off-Farm Income - Distribution (Missing values were imputed):", df_clean['Off_Farm_Income'].describe().loc[['mean', 'std']])
print("✅ The max Crop_Income should be lower than what you might expect in the raw data.")

# --- 4. Display the first 5 rows ---
print("\n--- Preview of the Cleaned Dataset ---")
print(df_clean.head())