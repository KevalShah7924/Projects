import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ==========================================
# 1. MODEL ENGINE (Handles .pkl Generation)
# ==========================================

def train_and_save_model(csv_path):
    """
    Trains the model and saves it as crop_model_v3.pkl.
    This ensures the .pkl and the .py file are always in sync.
    """
    df = pd.read_csv(csv_path)
    
    # Internal Feature Engineering
    df['health_water_index'] = df['NDVI_index'] * df['soil_moisture_%']
    df['temp_humidity_ratio'] = df['temperature_C'] / (df['humidity_%'] + 1)

    target = 'yield_kg_per_hectare'
    cat_cols = ['region', 'crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status']
    num_cols = [
        'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm', 
        'humidity_%', 'sunlight_hours', 'pesticide_usage_ml', 'total_days', 
        'NDVI_index', 'health_water_index', 'temp_humidity_ratio'
    ]
    
    X = df[cat_cols + num_cols]
    y = df[target]

    # Preprocessing Pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    # Model Configuration
    model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=5, random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    
    # Training
    pipeline.fit(X, y)
    
    # Save the pickle file
    with open('crop_model_v3.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline

# ==========================================
# 2. STREAMLIT UI & SIDEBAR
# ==========================================

st.set_page_config(page_title="Master Crop AI", layout="wide")

# Static Column Definition (Must match Training exactly)
COLS = ['region','crop_type','irrigation_type','fertilizer_type','crop_disease_status','soil_moisture_%','soil_pH','temperature_C','rainfall_mm','humidity_%','sunlight_hours','pesticide_usage_ml','total_days','NDVI_index','health_water_index','temp_humidity_ratio']

with st.sidebar:
    st.header("📊 Project Executive Summary")
    st.markdown("""
    ---
    **Objective:** An advanced AI solution designed to predict agricultural productivity using multi-dimensional environmental and soil data.

    **Key Technical Pillars:**
    * **Algorithm:** Gradient Boosting Regressor (GBM).
    * **Feature Expansion:** 2nd-degree Polynomial transformations.
    * **Automation:** End-to-end pipeline handling missing values and feature scaling.
    * **Enterprise Ready:** Supports bulk data ingestion via CSV, JSON, and SQL.
    ---
    """)
    st.caption("🚀 Optimized for Maximum Predictive Accuracy")

st.title("🌾 Smart Farming: AI Yield Prediction System")

# Automated .pkl Management
MODEL_FILE = 'crop_model_v3.pkl'
DATA_FILE = 'Smart_Farming_Crop_Yield_2024.csv'

if not os.path.exists(MODEL_FILE):
    if os.path.exists(DATA_FILE):
        with st.spinner("Initializing Model... This only happens once."):
            train_and_save_model(DATA_FILE)
    else:
        st.error(f"Required file '{DATA_FILE}' not found in directory.")
        st.stop()

with open(MODEL_FILE, 'rb') as f:
    model_pipeline = pickle.load(f)

# Navigation
tab1, tab2 = st.tabs(["🚀 Single Prediction", "📂 Bulk Data Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        u_region = st.selectbox("Region", ["North India", "South India", "Central USA", "South USA", "East Africa"])
        u_crop = st.selectbox("Crop Type", ["Wheat", "Maize", "Rice", "Cotton", "Soybean"])
        u_disease = st.selectbox("Disease Level", ["None", "Mild", "Moderate", "Severe"])
        u_irr = st.selectbox("Irrigation", ["Drip", "Sprinkler", "Manual", "None"])
        u_fert = st.selectbox("Fertilizer", ["Organic", "Inorganic", "Mixed"])
    with col2:
        u_moist = st.slider("Soil Moisture (%)", 10.0, 50.0, 25.0, key="m_val")
        u_temp = st.slider("Temperature (°C)", 10.0, 45.0, 25.0, key="t_val")
        u_rain = st.slider("Rainfall (mm)", 50.0, 500.0, 150.0, key="r_val")
        u_hum = st.slider("Humidity (%)", 20.0, 100.0, 60.0, key="h_val")
        u_ndvi = st.slider("NDVI Index", 0.1, 1.0, 0.6, key="n_val")

    if st.button("Generate Prediction"):
        input_data = pd.DataFrame({
            'region': [u_region], 'crop_type': [u_crop], 'irrigation_type': [u_irr],
            'fertilizer_type': [u_fert], 'crop_disease_status': [u_disease],
            'soil_moisture_%': [u_moist], 'soil_pH': [6.5], 'temperature_C': [u_temp],
            'rainfall_mm': [u_rain], 'humidity_%': [u_hum], 'sunlight_hours': [8.0],
            'pesticide_usage_ml': [25.0], 'total_days': [120], 'NDVI_index': [u_ndvi],
            'health_water_index': [u_ndvi * u_moist], 
            'temp_humidity_ratio': [u_temp / (u_hum + 1)]
        })
        
        prediction = model_pipeline.predict(input_data[COLS])[0]
        st.success(f"### Predicted Yield: **{prediction:.2f} kg/ha**")

with tab2:
    st.subheader("Bulk Data Selection")
    source = st.radio("Select Input Type:", ["CSV", "JSON", "SQL"])
    
    df_bulk = None
    if source == "CSV":
        f = st.file_uploader("Upload CSV", type="csv")
        if f: df_bulk = pd.read_csv(f)
    elif source == "JSON":
        f = st.file_uploader("Upload JSON", type="json")
        if f: df_bulk = pd.read_json(f)
    elif source == "SQL":
        db_f = st.file_uploader("Upload SQLite .db", type="db")
        sql_q = st.text_input("Enter Query", "SELECT * FROM farm_data")
        if db_f and sql_q:
            with open("temp.db", "wb") as fb: fb.write(db_f.getbuffer())
            conn = sqlite3.connect("temp.db")
            df_bulk = pd.read_sql(sql_q, conn)
            conn.close()

    if df_bulk is not None:
        if st.button("Process Bulk Predictions"):
            try:
                # Add engineered features to bulk data
                df_bulk['health_water_index'] = df_bulk['NDVI_index'] * df_bulk['soil_moisture_%']
                df_bulk['temp_humidity_ratio'] = df_bulk['temperature_C'] / (df_bulk['humidity_%'] + 1)
                
                df_bulk['Predicted_Yield'] = model_pipeline.predict(df_bulk[COLS])
                st.dataframe(df_bulk[['farm_id', 'Predicted_Yield']])
                
                csv_out = df_bulk.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions (CSV)", csv_out, "bulk_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Columns in file don't match. Error: {e}")