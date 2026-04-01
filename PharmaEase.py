# PharmaEase Project - Brainy Beam InfoTech
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from faker import Faker
import warnings
import sqlite3
import json
import io
import joblib

warnings.filterwarnings('ignore')

# ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

fake = Faker()
st.set_page_config(page_title="Pharma-Ease 💊", layout="wide", page_icon="💊")

# --- DS FOUNDATION: CENTRAL PIPELINE ---
def central_pipeline(input_data, source_type="CSV"):
    """
    Standardized data flow: Ingest, Clean, and Prep.
    No direct DB access allowed after this point.
    """
    try:
        # 1. Ingestion 
        if source_type == "CSV":
            df = pd.read_csv(input_data)
        elif source_type == "JSON":
            df = pd.read_json(input_data)
        elif source_type == "SQL":
            # Temporary SQLite connection for bulk SQL files
            conn = sqlite3.connect(':memory:')
            # Simplified flow: read uploaded data as CSV into SQL then back out
            df = pd.read_csv(input_data) 
            df.to_sql('inventory', conn, index=False)
            df = pd.read_sql('SELECT * FROM inventory', conn)
        
        # 2. Cleaning 
        df = df.drop_duplicates()
        df = df.fillna(method='ffill')
        
        # 3. Feature Engineering 
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            if 'expiry_date' in df.columns:
                df['expiry_date'] = pd.to_datetime(df['expiry_date'])
                df['days_to_expiry'] = (df['expiry_date'] - df['date']).dt.days
        
        return df
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        return None

# Session State Initialization
for key in ['models_trained', 'df', 'drug_classes', 'mae', 'forecast_values', 'forecast_dates', 'prescription_model', 'vectorizer']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'models_trained' else None

@st.cache_data
def generate_pharmacy_data(n=2000):
    """Generate clean pharmacy dataset using Faker"""
    categories = ['Antibiotics', 'Pain Relief', 'Vitamins', 'Antihistamines', 'Cardiovascular']
    drugs = {
        'Antibiotics': ['Amoxicillin', 'Azithromycin'],
        'Pain Relief': ['Paracetamol', 'Ibuprofen'], 
        'Vitamins': ['Vitamin C', 'Vitamin D'],
        'Antihistamines': ['Cetirizine', 'Loratadine'],
        'Cardiovascular': ['Atenolol', 'Amlodipine']
    }
    
    data = []
    for _ in range(n):
        cat = np.random.choice(categories)
        drug = np.random.choice(drugs[cat])
        qty = np.random.randint(10, 500)
        price = np.random.uniform(5, 200)
        date = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        expiry = date + pd.Timedelta(days=np.random.randint(365, 730))
        
        data.append({
            'drug_name': drug, 
            'category': cat, 
            'quantity': qty, 
            'price': price, 
            'date': date, 
            'expiry_date': expiry,
            'usage_rate': np.random.uniform(0.5, 5.0),
            'sales': qty * np.random.uniform(0.8, 1.2)
        })
    
    df = pd.DataFrame(data)
    df['days_to_expiry'] = (df['expiry_date'] - df['date']).dt.days
    df['stock_velocity'] = df['quantity'] / (df['usage_rate'] + 1)
    return df.sort_values('date')

# --- ML CORE FUNCTIONS ---
def safe_forecast(df, steps=30):
    """Predict future sales trend"""
    sales_daily = df.groupby('date')['sales'].sum().tail(30)
    if len(sales_daily) == 0: return np.full(steps, 1000), []
    x = np.arange(len(sales_daily))
    trend = np.polyfit(x, sales_daily.values, 1)
    forecast = [max(0, sales_daily.iloc[-1] + trend[0] * (i + 1) + np.random.normal(0, 50)) for i in range(steps)]
    dates = pd.date_range(start=df['date'].max() + timedelta(days=1), periods=steps, freq='D')
    return forecast, dates.tolist()

def train_prescription_model(df):
    """NLP-based drug recommendation"""
    symptoms_dict = {
        'Amoxicillin': 'fever cough infection sore throat', 'Azithromycin': 'bacterial infection pneumonia', 
        'Paracetamol': 'headache fever pain', 'Ibuprofen': 'inflammation pain swelling',
        'Vitamin C': 'cold flu immune', 'Vitamin D': 'bone pain weakness',
        'Cetirizine': 'allergy sneezing itching', 'Loratadine': 'hay fever allergy',
        'Atenolol': 'high blood pressure hypertension', 'Amlodipine': 'hypertension angina'
    }
    available_drugs = [d for d in symptoms_dict if d in df['drug_name'].values]
    if len(available_drugs) < 3: return None, None, ['Paracetamol', 'Amoxicillin']
    
    df_filtered = df[df['drug_name'].isin(available_drugs)].copy()
    df_filtered['symptoms'] = df_filtered['drug_name'].map(symptoms_dict)
    df_filtered = df_filtered.dropna(subset=['symptoms'])
    
    le = LabelEncoder()
    y = le.fit_transform(df_filtered['drug_name'])
    vec = TfidfVectorizer(max_features=15, stop_words='english')
    X = vec.fit_transform(df_filtered['symptoms'])
    
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model, vec, le.classes_.tolist()

# === SIDEBAR: BULK INGESTION ===
st.sidebar.title("📥 Data Ingestion")
ingestion_type = st.sidebar.selectbox("Source Type", ["Synthetic (Faker)", "CSV", "JSON", "SQL (via Buffer)"])

if ingestion_type == "Synthetic (Faker)":
    if st.sidebar.button("Generate New Dataset"):
        st.session_state.df = generate_pharmacy_data()
        st.sidebar.success("Generated 2000 records!")
else:
    uploaded_file = st.sidebar.file_uploader(f"Upload {ingestion_type} file", type=['csv', 'json', 'sql', 'txt'])
    if uploaded_file:
        processed_df = central_pipeline(uploaded_file, source_type=ingestion_type.split()[0])
        if processed_df is not None:
            st.session_state.df = processed_df
            st.sidebar.success(f"Successfully processed {len(processed_df)} rows!")

# Navigation
page = st.sidebar.selectbox("📂 Select Module", ["📊 Dashboard", "📦 Inventory", "💉 Prescriptions", "💰 Sales", "👥 Employees", "📈 Reports"])

if st.session_state.df is None:
    st.session_state.df = generate_pharmacy_data()

df = st.session_state.df.copy()

# TRAIN MODELS BUTTON
if st.sidebar.button("🚀 **TRAIN ML MODELS**", type="primary"):
    with st.spinner("🔄 Training DS/ML Pipeline..."):
        st.session_state.forecast_values, st.session_state.forecast_dates = safe_forecast(df)
        st.session_state.prescription_model, st.session_state.vectorizer, st.session_state.drug_classes = train_prescription_model(df)
        st.session_state.models_trained = True
        if st.session_state.prescription_model:
            joblib.dump(st.session_state.prescription_model, 'prescription_model.pkl')
    st.sidebar.success("✅ MODELS TRAINED & SAVED!")

# === MODULE PAGES ===
if page == "📊 Dashboard":
    st.title("PharmaEase Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧪 Total Stock", f"{df['quantity'].sum():,.0f}")
    col2.metric("💰 Total Sales", f"Rs.{df['sales'].sum():,.0f}")
    col3.metric("⚠️ Low Stock", len(df[df['quantity'] < 50]))
    col4.metric("⏳ Expiring Soon", len(df[df['days_to_expiry'] < 30]))
    fig = px.bar(df.groupby('category')['quantity'].sum().reset_index(), x='category', y='quantity', title="Stock Distribution")
    st.plotly_chart(fig, use_container_width=True)

elif page == "📦 Inventory":
    st.header("📦 Inventory Management")
    st.dataframe(df[df['quantity'] < 50].head(10), use_container_width=True)
    if st.session_state.models_trained:
        st.subheader("📈 30-Day Sales Forecast")
        fig = px.line(x=st.session_state.forecast_dates, y=st.session_state.forecast_values, title="Predicted Sales Trend")
        st.plotly_chart(fig)

elif page == "💉 Prescriptions":
    st.header("💉 AI Drug Recommendation")
    symptoms = st.text_input("🏥 Enter patient symptoms:", value="fever cough").strip().lower()
    if st.button("🔮 RECOMMEND DRUG"):
        if st.session_state.models_trained and st.session_state.vectorizer:
            vec_input = st.session_state.vectorizer.transform([symptoms])
            prediction = st.session_state.prescription_model.predict(vec_input)[0]
            recommended = st.session_state.drug_classes[prediction]
            st.success(f"✅ RECOMMENDED DRUG: {recommended.upper()}")
        else:
            st.error("Please train models first!")

elif page == "💰 Sales":
    st.header("💰 Point of Sale")
    items = st.multiselect("🛒 Add to cart:", df['drug_name'].unique()[:10])
    if items:
        total = df[df['drug_name'].isin(items)]['price'].sum()
        st.metric("💳 TOTAL", f"Rs.{total:.2f}")
        if st.button("🧾 GENERATE BILL"): st.balloons()

elif page == "👥 Employees":
    st.header("👥 Employee Management & Performance Matrix")
    
    # DS Focus: Statistical analysis of performance
    staff_names = [f"Staff_{i}" for i in range(1, 11)]
    emp_data = [{'employee_id': name, 'performance_score': np.random.randint(60, 100),
                 'attendance': np.random.uniform(0.7, 1.0), 'workload': np.random.randint(20, 50)} for name in staff_names]
    df_emp = pd.DataFrame(emp_data)
    
    scaler = MinMaxScaler()
    df_emp['norm_score'] = scaler.fit_transform(df_emp[['performance_score']])
    
    st.subheader("📈 Performance Matrix")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_perf = px.scatter(df_emp, x="workload", y="performance_score", size="attendance", 
                             color="norm_score", hover_name="employee_id", title="Efficiency Matrix")
        st.plotly_chart(fig_perf, use_container_width=True)
    with col2:
        st.write("**Top Performers**")
        st.table(df_emp.nlargest(3, 'performance_score')[['employee_id', 'performance_score']])

    # ML Focus: Predict staffing needs
    st.divider()
    st.subheader("🤖 ML Staffing Predictor")
    if st.session_state.models_trained and st.session_state.forecast_values:
        avg_forecast = np.mean(st.session_state.forecast_values)
        needed = int(avg_forecast / 5000) + 1
        st.info(f"ML Prediction: **{needed}** staff members needed based on sales forecast.")
        st.write("**Predicted Shift Schedule**")
        st.dataframe(df_emp.sample(min(len(df_emp), needed))[['employee_id', 'workload']], use_container_width=True)

elif page == "📈 Reports":
    st.header("📈 Business Intelligence")
    st.plotly_chart(px.pie(df, values='sales', names='category', title="Sales by Category"))

st.sidebar.markdown("---")
st.sidebar.info("👈 Use Ingestion to load new data, then click Train!")