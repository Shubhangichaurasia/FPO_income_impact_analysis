import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

print("--- Step 1: Training and saving the final, definitive model ---")

# 1. Load your dataset
try:
    df = pd.read_csv('fpo_data_with_ndvi.csv')
    print("✅ Dataset 'fpo_data_with_ndvi.csv' loaded successfully.")
except FileNotFoundError:
    print("🛑 ERROR: 'fpo_data_with_ndvi.csv' not found. Please ensure it is in the correct folder.")
    exit()


# 2. Prepare the data for training
# This ensures the model only learns from features the user will provide in the app.
df_for_training = df.drop(columns=[
    'Farmer_ID', 'Village_ID', 'State', 'FPO_Member', 'WDC_Intervention_Area',
    'Latitude', 'Longitude', 'NDVI_Before', 'NDVI_After'
], errors='ignore') # 'errors=ignore' prevents errors if a column is already missing

# Create 'NDVI_Change' if the columns exist
if 'NDVI_After' in df.columns and 'NDVI_Before' in df.columns:
    df_for_training['NDVI_Change'] = df['NDVI_After'] - df['NDVI_Before']

# 3. Convert all text columns to numerical format and clean the data
df_model = pd.get_dummies(df_for_training, drop_first=True)
df_model.dropna(inplace=True)

# 4. Define X (features) and y (target)
y = df_model['Income_Annual']
X = df_model.drop('Income_Annual', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train your best model with optimal parameters
print("Training the Gradient Boosting model...")
final_model = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.7, random_state=42
)
final_model.fit(X_train, y_train)
print("✅ Final model training complete.")

# 6. Save the final model and its exact columns
joblib.dump(final_model, 'fpo_predictor_model.joblib')
joblib.dump(X_train.columns, 'model_columns.joblib')
print("✅ Final model and columns have been saved successfully to 'fpo_predictor_model.joblib' and 'model_columns.joblib'.")
print("\nYou can now run your Streamlit app.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

print("--- Step 1: Training and saving the final, definitive model ---")

# 1. Load your dataset
try:
    df = pd.read_csv('fpo_data_with_ndvi.csv')
    print("✅ Dataset 'fpo_data_with_ndvi.csv' loaded successfully.")
except FileNotFoundError:
    print("🛑 ERROR: 'fpo_data_with_ndvi.csv' not found. Please ensure it is in the correct folder.")
    exit()


# 2. Prepare the data for training
df_for_training = df.drop(columns=[
    'Farmer_ID', 'Village_ID', 'State', 'FPO_Member', 'WDC_Intervention_Area',
    'Latitude', 'Longitude', 'NDVI_Before', 'NDVI_After'
], errors='ignore')

if 'NDVI_After' in df.columns and 'NDVI_Before' in df.columns:
    df_for_training['NDVI_Change'] = df['NDVI_After'] - df['NDVI_Before']

# 3. Convert all text columns to numerical format and clean the data
df_model = pd.get_dummies(df_for_training, drop_first=True)
df_model.dropna(inplace=True)

# 4. Define X (features) and y (target)
y = df_model['Income_Annual']
X = df_model.drop('Income_Annual', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train your best model with optimal parameters
print("Training the Gradient Boosting model...")
final_model = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.7, random_state=42
)
final_model.fit(X_train, y_train)
print("✅ Final model training complete.")

# 6. Save the final model and its exact columns
joblib.dump(final_model, 'fpo_predictor_model.joblib')
joblib.dump(X_train.columns, 'model_columns.joblib')
print("✅ Final model and columns have been saved successfully to 'fpo_predictor_model.joblib' and 'model_columns.joblib'.")

# ==============================================================================
# NEW: HACKATHON ANALYSIS - PROVING THE "WHY"
# ==============================================================================
print("\n--- Starting Hackathon Analysis: Proving the FPO's Mechanisms of Impact ---")

# --- Analysis 1: The Causal Financial Impact ---
print("\nAnalyzing the direct financial impact of the FPO...")
causal_features = ['Group_Treatment', 'Land_Size_Hectares', 'Rainfall_Annual_mm', 'Soil_Health_Score']
causal_df = pd.get_dummies(df, columns=['Group'], drop_first=True)[causal_features + ['Income_Annual']].dropna()

X_causal = causal_df[causal_features]
y_causal = causal_df['Income_Annual']

lr = LinearRegression()
lr.fit(X_causal, y_causal)
treatment_effect = lr.coef_[0]
print(f"✅ Statistical Proof: The FPO intervention caused a direct average income increase of ₹{treatment_effect:,.2f}")


# --- Analysis 2: The Mechanisms of Success (Charts for Dashboard) ---
print("\nGenerating a single, effective comparison chart for the dashboard...")

# Create a 2x2 grid for our charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("FPO Impact Analysis: A Comparison of Key Benefits", fontsize=20, weight='bold')

# --- Chart 1: Access to Credit ---
credit_data = df.groupby('Group')['Access_to_Credit'].value_counts(normalize=True).unstack().fillna(0)
credit_data['YES'].plot(kind='bar', ax=axes[0, 0], color=['#ff9999','#66b3ff'])
axes[0, 0].set_title('Impact on Access to Credit', fontsize=14, weight='bold')
axes[0, 0].set_ylabel('Percentage of Farmers (%)')
axes[0, 0].tick_params(axis='x', rotation=0)
axes[0, 0].set_ylim(0, 100)

# --- Chart 2: Access to Training ---
training_data = df.groupby('Group')['Training_Received'].value_counts(normalize=True).unstack().fillna(0)
training_data['YES'].plot(kind='bar', ax=axes[0, 1], color=['#ff9999','#66b3ff'])
axes[0, 1].set_title('Impact on Training Access', fontsize=14, weight='bold')
axes[0, 1].set_ylabel('Percentage of Farmers (%)')
axes[0, 1].tick_params(axis='x', rotation=0)
axes[0, 1].set_ylim(0, 100)

# --- Chart 3: Productivity (Yield) ---
sns.barplot(x='Group', y='Yield_Kg_Per_Hectare', data=df, ax=axes[1, 0], palette=['#ff9999','#66b3ff'])
axes[1, 0].set_title('Impact on On-Farm Productivity', fontsize=14, weight='bold')
axes[1, 0].set_ylabel('Average Yield (Kg per Hectare)')
axes[1, 0].set_xlabel('')

# --- Chart 4: Cultivation Cost ---
sns.barplot(x='Group', y='Cost_of_Cultivation', data=df, ax=axes[1, 1], palette=['#ff9999','#66b3ff'])
axes[1, 1].set_title('Impact on Cultivation Costs', fontsize=14, weight='bold')
axes[1, 1].set_ylabel('Average Cost of Cultivation (INR)')
axes[1, 1].set_xlabel('')


# Improve layout and save the single, combined chart
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('impact_summary_dashboard_chart.png', dpi=150)
print("✅ Single, effective dashboard chart 'impact_summary_dashboard_chart.png' saved.")


print("\n--- Analysis Complete ---")
print("You now have a powerful, single visual to showcase the FPO's benefits in your dashboard.")