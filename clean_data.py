import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from pathlib import Path

print("Starting data cleaning pipeline...")

# --- 1. Load the raw dataset from the Desktop ---
try:
    desktop_path = Path.home() / "Desktop"
    df = pd.read_csv(desktop_path / 'fpo_advanced_income_data.csv')
    print("Successfully loaded raw data.")
except FileNotFoundError:
    print(f"ERROR: The file 'fpo_advanced_income_data.csv' was not found on your Desktop.")
    print("Please make sure you have run the dataset generation script successfully.")
    exit()

# --- 2. Fix Missing Values using KNN Imputation ---
imputer_cols = ['Farm_Size_Acres', 'Rainfall_mm', 'Crop_Income', 'Livestock_Income', 'Off_Farm_Income']
imputer = KNNImputer(n_neighbors=5)
df[imputer_cols] = imputer.fit_transform(df[imputer_cols])
print("Missing values have been imputed.")

# --- 3. Handle Outliers using the IQR Method ---
col = 'Crop_Income'
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
print("Outliers have been capped.")

# --- 4. Recalculate Total Income & Save the Clean File ---
df['Total_Income_INR'] = df[['Crop_Income', 'Livestock_Income', 'Off_Farm_Income']].sum(axis=1)
clean_file_path = desktop_path / "fpo_cleaned_data.csv"
df.to_csv(clean_file_path, index=False)
print(f"Processing complete. Cleaned data saved to: {clean_file_path}")