
import pandas as pd
import numpy as np
from pathlib import Path

print("Starting enhanced dataset generation with new, comprehensive variables...")

# --- 1. Configuration for the Dataset ---
NUM_FARMERS_TREATMENT = 1500
NUM_FARMERS_CONTROL = 500
ACRES_PER_HECTARE = 2.47105

# --- 2. Create Expanded Base Farmer Profiles ---
num_total_farmers = NUM_FARMERS_TREATMENT + NUM_FARMERS_CONTROL

# Environmental data mapped to districts for realism
districts = ['Muradnagar', 'Hapur', 'Meerut', 'Pune', 'Nagpur', 'Mysuru', 'Dharwad']
states = ['Uttar Pradesh', 'Uttar Pradesh', 'Uttar Pradesh', 'Maharashtra', 'Maharashtra', 'Karnataka', 'Karnataka']
state_map = dict(zip(districts, states))
rainfall_map = {'Muradnagar': 800, 'Hapur': 810, 'Meerut': 820, 'Pune': 750, 'Nagpur': 1200, 'Mysuru': 850, 'Dharwad': 900}
temp_map = {'Muradnagar': 24.5, 'Hapur': 24.6, 'Meerut': 24.4, 'Pune': 25.0, 'Nagpur': 27.0, 'Mysuru': 24.8, 'Dharwad': 24.2}
village_ids = {dist: [f"{dist[:3].upper()}{str(i).zfill(2)}" for i in range(1, 6)] for dist in districts}

# Generate base profiles
farmer_profiles = pd.DataFrame({
    'Farmer_ID': [f"FARM{str(i).zfill(4)}" for i in range(1, num_total_farmers + 1)],
    'Group': ['Treatment'] * NUM_FARMERS_TREATMENT + ['Control'] * NUM_FARMERS_CONTROL,
    'District': np.random.choice(districts, size=num_total_farmers, p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.15, 0.15]),
    'Land_Size_Hectares': np.round(np.random.gamma(shape=2.5, scale=2, size=num_total_farmers) * 0.404686 + 0.5, 2),
    'Primary_Crop': np.random.choice(['Sugarcane', 'Wheat', 'Rice', 'Maize', 'Pulses', 'Cotton', 'Soybean', 'Gram'], size=num_total_farmers),
    'Irrigation_Source_Before': np.random.choice(['Rainfed', 'Well', 'Canal'], size=num_total_farmers, p=[0.4, 0.3, 0.3]),
    'Age_of_Farmer': np.random.randint(25, 65, size=num_total_farmers),
    'Soil_Health_Score': np.round(np.random.uniform(3.5, 8.5, size=num_total_farmers), 1),
    'Access_to_Credit': np.random.choice(['YES', 'NO'], size=num_total_farmers, p=[0.6, 0.4]),
    'Distance_to_Market_KM': np.round(np.random.lognormal(mean=2.5, sigma=0.5, size=num_total_farmers), 1),
    'Off_Farm_Income': np.round(np.random.choice([0, 1], size=num_total_farmers, p=[0.3, 0.7]) * np.random.uniform(20000, 80000, size=num_total_farmers), -2)
})

# --- Add derived and mapped columns ---
farmer_profiles['State'] = farmer_profiles['District'].map(state_map)
farmer_profiles['Village_ID'] = farmer_profiles['District'].apply(lambda d: np.random.choice(village_ids[d]))
farmer_profiles['FPO_Member'] = np.where(farmer_profiles['Group'] == 'Treatment', 'YES', 'NO')
farmer_profiles['WDC_Intervention_Area'] = np.where(farmer_profiles['Group'] == 'Treatment', 'YES', 'NO')
farmer_profiles['Farm_Size_Acres'] = np.round(farmer_profiles['Land_Size_Hectares'] * ACRES_PER_HECTARE, 2)
farmer_profiles['Rainfall_Annual_mm'] = farmer_profiles['District'].map(rainfall_map) + np.random.randint(-50, 50, size=num_total_farmers)
farmer_profiles['Avg_Temp_Celsius'] = farmer_profiles['District'].map(temp_map) + np.random.uniform(-0.5, 0.5, size=num_total_farmers).round(1)

# Mechanisation is more likely on larger farms
prob_tractor = 0.1 + (farmer_profiles['Land_Size_Hectares'] / farmer_profiles['Land_Size_Hectares'].max()) * 0.7
farmer_profiles['Has_Tractor'] = np.where(np.random.rand(num_total_farmers) < prob_tractor, 'YES', 'NO')


# --- 3. Generate Before & After Data for Each Farmer ---
df_list = []
for _, row in farmer_profiles.iterrows():
    # --- BEFORE Data ---
    df_before = row.to_frame().T.copy()
    df_before['Time_Period'] = 'Before'

    # Baseline inputs
    df_before['Fertilizer_Used_Kg_per_Hectare'] = 100 + (row['Soil_Health_Score'] < 5) * 20 + np.random.uniform(-10, 10)
    df_before['Pesticide_Used_Liters_per_Hectare'] = 5 + (row['Primary_Crop'] in ['Cotton', 'Sugarcane']) * 2 + np.random.uniform(-1, 1)

    # Baseline calculations based on multiple factors
    base_yield = 3000 + (row['Soil_Health_Score'] * 150) + (row['Irrigation_Source_Before'] != 'Rainfed') * 500 - (row['Avg_Temp_Celsius'] - 25) * 100
    df_before['Yield_Kg_Per_Hectare'] = base_yield * row['Land_Size_Hectares'] + np.random.normal(0, 200)

    base_income = (base_yield * 15) - (df_before['Fertilizer_Used_Kg_per_Hectare'] * 20) - (df_before['Pesticide_Used_Liters_per_Hectare'] * 500)
    df_before['Income_Annual'] = base_income * row['Land_Size_Hectares'] + np.random.normal(0, 5000)

    df_before['Cost_of_Cultivation'] = (10000 + (row['Has_Tractor'] == 'NO') * 1500) * row['Land_Size_Hectares'] + np.random.normal(0, 1000)

    # Add empty columns for intervention-specific 'After' data
    df_before['Irrigation_Source_After'] = row['Irrigation_Source_Before'] # No change in 'before'
    df_before['Training_Received'] = 'NO'
    df_before['Crop_Diversification'] = 'NO'

    # --- AFTER Data (with project effect) ---
    df_after = df_before.copy()
    df_after['Time_Period'] = 'After'
    
    # Simulate intervention effects for the Treatment group
    if row['Group'] == 'Treatment':
        # 1. Improved Irrigation
        if row['Irrigation_Source_Before'] == 'Rainfed' and np.random.rand() < 0.6: # 60% upgrade
            df_after['Irrigation_Source_After'] = np.random.choice(['Well', 'Canal'])
        
        # 2. Training leads to better practices
        df_after['Training_Received'] = 'YES'
        efficiency_gain = np.random.uniform(1.1, 1.2) # 10-20% gain
        cost_reduction_factor = np.random.uniform(0.90, 0.98) # 2-10% cost reduction

        # 3. Crop diversification
        if np.random.rand() < 0.25: # 25% of farmers diversify
            df_after['Crop_Diversification'] = 'YES'

        # Apply effects
        df_after['Fertilizer_Used_Kg_per_Hectare'] *= np.random.uniform(0.95, 1.05) # Training might optimize usage
        df_after['Pesticide_Used_Liters_per_Hectare'] *= np.random.uniform(0.90, 1.0)
        df_after['Cost_of_Cultivation'] *= cost_reduction_factor
        
        # Recalculate yield and income based on NEW practices
        new_base_yield = 3000 + (row['Soil_Health_Score'] * 150) + (df_after['Irrigation_Source_After'].iloc[0] != 'Rainfed') * 500 - (row['Avg_Temp_Celsius'] - 25) * 100
        df_after['Yield_Kg_Per_Hectare'] = new_base_yield * row['Land_Size_Hectares'] * efficiency_gain
        
        new_base_income = (new_base_yield * 15 * efficiency_gain) - (df_after['Fertilizer_Used_Kg_per_Hectare'] * 20) - (df_after['Pesticide_Used_Liters_per_Hectare'] * 500)
        df_after['Income_Annual'] = new_base_income * row['Land_Size_Hectares'] * (1 + (df_after['Crop_Diversification'].iloc[0] == 'YES') * 0.15) # Diversification bonus
        
    else: # Control group experiences smaller, general trends
        general_trend_effect = np.random.uniform(1.02, 1.08)
        df_after['Income_Annual'] *= general_trend_effect
        df_after['Yield_Kg_Per_Hectare'] *= np.random.uniform(1.0, 1.05)
        df_after['Cost_of_Cultivation'] *= np.random.uniform(1.0, 1.03)

    df_list.extend([df_before, df_after])

final_df = pd.concat(df_list, ignore_index=True)

# --- 4. Final Polish and Save ---
final_df = final_df.sort_values(by=['Farmer_ID', 'Time_Period']).reset_index(drop=True)

# Reorder columns for logical flow
final_df = final_df[[
    # Identifiers
    'Farmer_ID', 'Village_ID', 'District', 'State',
    # Grouping
    'Group', 'FPO_Member', 'WDC_Intervention_Area',
    # Farmer & Farm Characteristics
    'Age_of_Farmer', 'Land_Size_Hectares', 'Farm_Size_Acres', 'Primary_Crop', 'Has_Tractor', 'Access_to_Credit',
    'Distance_to_Market_KM', 'Off_Farm_Income',
    # Environmental
    'Soil_Health_Score', 'Rainfall_Annual_mm', 'Avg_Temp_Celsius',
    # Time-based Variables
    'Time_Period',
    # Inputs & Interventions
    'Irrigation_Source_Before', 'Irrigation_Source_After', 'Training_Received', 'Crop_Diversification',
    'Fertilizer_Used_Kg_per_Hectare', 'Pesticide_Used_Liters_per_Hectare',
    # Outcome Metrics
    'Yield_Kg_Per_Hectare', 'Cost_of_Cultivation', 'Income_Annual'
]]

# Ensure numeric columns are rounded for cleanliness
numeric_cols = [
    'Land_Size_Hectares', 'Farm_Size_Acres', 'Soil_Health_Score', 'Distance_to_Market_KM',
    'Off_Farm_Income', 'Rainfall_Annual_mm', 'Avg_Temp_Celsius',
    'Fertilizer_Used_Kg_per_Hectare', 'Pesticide_Used_Liters_per_Hectare',
    'Yield_Kg_Per_Hectare', 'Cost_of_Cultivation', 'Income_Annual'
]
for col in numeric_cols:
    final_df[col] = pd.to_numeric(final_df[col], errors='coerce').round(2)

# Save the final dataset to the Desktop
try:
    desktop_path = Path.home() / "Desktop"
    desktop_path.mkdir(exist_ok=True)
    file_path = desktop_path / "fpo_impact_data_enhanced.csv"
    final_df.to_csv(file_path, index=False)
    print(f"\nDataset successfully generated and saved to: {file_path}")
except Exception as e:
    print(f"\nAn error occurred while saving to Desktop: {e}")
    print("Saving to the current folder instead.")
    final_df.to_csv("fpo_impact_data_enhanced.csv", index=False)
    print("Dataset successfully generated and saved to the current project folder.")