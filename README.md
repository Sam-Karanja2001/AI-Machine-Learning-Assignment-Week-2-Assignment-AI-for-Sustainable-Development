#AI Machine Learning Assignment
Week 2 Assignment: AI for Sustainable Development
Theme: "Machine Learning Meets the UN Sustainable Development Goals (SDGs)" 🌍🤖

# -*- coding: utf-8 -*-
"""
Week 2 Assignment: AI for Sustainable Development
Project: Predicting Crop Yields for Food Security (SDG 2: Zero Hunger)

This script demonstrates a supervised machine learning approach to predict crop yields
based on various agricultural factors. This can aid in proactive resource
management, logistics, and policy-making to combat food insecurity.
"""

# --- 1. Import necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully!")

# --- 2. Dataset & Tools: Synthetic Data Generation ---
# For demonstration purposes, we create a synthetic dataset.
# In a real-world scenario, this data would come from sources like FAO, World Bank,
# national agricultural surveys, or satellite imagery.

print("\n--- 2. Generating Synthetic Crop Yield Data ---")

np.random.seed(42) # for reproducibility

# Number of data points (farms/fields)
n_samples = 1000

# Features (Input Variables)
rainfall = np.random.uniform(500, 1500, n_samples) # mm per growing season
temperature = np.random.uniform(15, 35, n_samples) # Average Celsius during growing season
fertilizer_used = np.random.uniform(50, 300, n_samples) # kg/hectare
pesticide_used = np.random.uniform(0, 10, n_samples) # liters/hectare
soil_types = np.random.choice(['sandy', 'loamy', 'clay'], n_samples, p=[0.3, 0.5, 0.2])
area_hectares = np.random.uniform(1, 50, n_samples) # hectares

# Target Variable (Output: Crop Yield in tons/hectare)
# We define a simple relationship with some noise to simulate real-world variability
# Base yield + (effect of rainfall * weight) + (effect of temp * weight) + ... + noise
crop_yield = (
    5.0
    + (rainfall / 200) * 0.5   # Higher rainfall, higher yield (up to a point)
    + (temperature / 10) * 0.3 # Optimal temperature range is crucial
    + (fertilizer_used / 100) * 0.7 # Fertilizer has a strong positive effect
    - (pesticide_used * 0.2)   # Pesticides might have slight negative if overused, or positive if controlling pests
    + (area_hectares * 0.05)   # Larger area might slightly increase yield per hectare due to scale
    + np.random.normal(0, 1.5, n_samples) # Add some random noise
)

# Adjust for soil type effects (simplified)
for i in range(n_samples):
    if soil_types[i] == 'sandy':
        crop_yield[i] *= 0.8 # Sandy soil generally lower yield
    elif soil_types[i] == 'clay':
        crop_yield[i] *= 1.1 # Clay soil generally higher yield (good water retention)

# Ensure yield is not negative
crop_yield = np.maximum(0.5, crop_yield)

# Create a DataFrame
data = pd.DataFrame({
    'Rainfall_mm': rainfall,
    'Temperature_C': temperature,
    'Fertilizer_kg_ha': fertilizer_used,
    'Pesticide_L_ha': pesticide_used,
    'Soil_Type': soil_types,
    'Area_Hectares': area_hectares,
    'Crop_Yield_tons_ha': crop_yield
})

print("Synthetic data generated successfully. First 5 rows:")
print(data.head())
print(f"Dataset shape: {data.shape}")

# --- 3. Build Your Model: Preprocess Data ---
print("\n--- 3. Preprocessing Data ---")

# Define features (X) and target (y)
X = data.drop('Crop_Yield_tons_ha', axis=1)
y = data['Crop_Yield_tons_ha']

# Identify categorical and numerical features
categorical_features = ['Soil_Type']
numerical_features = ['Rainfall_mm', 'Temperature_C', 'Fertilizer_kg_ha', 'Pesticide_L_ha', 'Area_Hectares']

# Create a preprocessing pipeline
# One-hot encode categorical features, scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
print("Data split into training and testing sets.")

# --- 4. Build Your Model: Train Model ---
print("\n--- 4. Training the Machine Learning Model (RandomForestRegressor) ---")

# Create a pipeline that first preprocesses and then trains the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 uses all available cores
])

# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete!")

# --- 4. Build Your Model: Evaluate ---
print("\n--- 4. Evaluating the Model ---")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} tons/hectare")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} tons/hectare")
print(f"R-squared (R2 Score): {r2:.2f}")

# Visualize results: Actual vs. Predicted Yields
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # y=x line
plt.xlabel("Actual Crop Yield (tons/hectare)")
plt.ylabel("Predicted Crop Yield (tons/hectare)")
plt.title("Actual vs. Predicted Crop Yields")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Visualize Residuals (Errors)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals (Prediction Errors)")
plt.xlabel("Residuals (Actual - Predicted Yield)")
plt.ylabel("Frequency")
plt.show()

print("\nModel evaluation complete and visualizations displayed.")

# --- 5. Ethical Reflection ---
print("\n--- 5. Ethical Reflection ---")

print("\n**How might bias in your data affect outcomes?**")
print("""
1.  **Geographical Bias:** If training data disproportionately comes from one region (e.g., highly fertile lands with optimal climate), the model might perform poorly when applied to other regions with different soil types, climates, or farming practices (e.g., arid regions, smallholder farms). This could lead to inaccurate advice or resource allocation, exacerbating existing inequalities.
2.  **Socio-economic Bias:** Data might omit or underrepresent factors like access to irrigation, quality of seeds, financial resources for farmers, or access to agricultural extension services. A model trained without these crucial socio-economic features could unfairly disadvantage small-scale or marginalized farmers whose yields are heavily influenced by these uncaptured variables.
3.  **Crop-Specific Bias:** If the model is trained primarily on data for a few dominant crops, its predictions might be unreliable for other crops vital for local food security or biodiversity.
4.  **Measurement Bias:** Inaccurate or inconsistent data collection methods for rainfall, temperature, or yield reporting could introduce systematic errors, making the model's predictions less reliable for real-world application.
""")

print("\n**How does your solution promote fairness and sustainability?**")
print("""
1.  **Fairness (Improved Resource Allocation):** By providing more accurate and timely yield forecasts, the solution can help policymakers and aid organizations allocate resources (e.g., fertilizers, improved seeds, irrigation support, disaster relief) more equitably and efficiently to areas or farmers most in need, preventing food shortages and stabilizing incomes.
2.  **Fairness (Empowerment):** Access to predictive insights can empower smallholder farmers, allowing them to make more informed decisions about planting, harvesting, and market engagement, reducing their vulnerability to unpredictable conditions.
3.  **Sustainability (Resource Optimization):** Accurate predictions can lead to optimized use of scarce resources like water (through better irrigation scheduling) and reduced overuse of fertilizers/pesticides, minimizing environmental impact (e.g., water pollution, soil degradation). This aligns with SDG 6 (Clean Water and Sanitation) and SDG 15 (Life on Land).
4.  **Sustainability (Food Waste Reduction):** Better forecasts can help manage supply chains, reducing post-harvest losses and food waste by optimizing storage, transport, and distribution, contributing to responsible consumption and production (SDG 12).
5.  **Sustainability (Climate Resilience):** Understanding how various environmental factors influence yield allows for better adaptation strategies to climate change, ensuring agricultural resilience for future generations (SDG 13: Climate Action).
""")

print("\n--- Project Execution Complete ---")
