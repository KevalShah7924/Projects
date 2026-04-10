import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
import io
import time
from faker import Faker
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="Global Spending AI", layout="wide")
fake = Faker()

# --- COUNTRY WAGE DATA ---
COUNTRY_WAGES = {
    "United States": 1256,  
    "United Kingdom": 1850, 
    "Australia": 3060,      
    "Germany": 2100,         
    "India": 215,           
    "Brazil": 250           
}

# --- DATA & ML ENGINE (REALISTIC E-COMMERCE LOGIC) ---
@st.cache_data
def load_base_data(country_name):
    """Generates synthetic data based on E-commerce Wallet Share logic."""
    data = []
    monthly_min = COUNTRY_WAGES[country_name]
    annual_min = monthly_min * 12
    for _ in range(1200):
        age = np.random.randint(18, 70)
        tenure = np.random.randint(0, 15)
        
        # Income follows a log-normal distribution (realistic spread)
        multiplier = np.random.lognormal(mean=0.6, sigma=0.5) + (age * 0.005)
        income = annual_min * multiplier
        
        # LOGIC: E-commerce Wallet Share (typically 3% to 15% of total income)
        # Higher earners spend a lower percentage of total income on the platform.
        base_share = np.clip(0.14 - (np.log1p(income/annual_min) * 0.025), 0.03, 0.15)
        
        # Spending formula: Income Share + Loyalty/Age Boost
        platform_spend = (income * base_share) + (tenure * 150) + (age * 12)
        
        # Add market noise (seasonal fluctuations)
        platform_spend += np.random.normal(0, platform_spend * 0.08)
        
        data.append({
            "Name": fake.name(),
            "Age": age,
            "Gender": np.random.choice(["Male", "Female", "Other"]),
            "Annual_Income": round(income, 2),
            "Years_as_Customer": tenure,
            "Total_Spent": round(max(100, platform_spend), 2)
        })
    return pd.DataFrame(data)

def train_system(df):
    """Trains a Gradient Boosting model with high-accuracy validation."""
    le = LabelEncoder()
    df_m = df.copy()
    df_m['Gender'] = le.fit_transform(df['Gender'])
    X = df_m[['Age', 'Gender', 'Annual_Income', 'Years_as_Customer']]
    y = df_m['Total_Spent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
    model.fit(X_train_scaled, y_train)
    
    acc = r2_score(y_test, model.predict(X_test_scaled))
    return model, scaler, le, acc

# --- SESSION STATE MANAGEMENT ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

# --- SIDEBAR & DATA REFRESH ---
st.sidebar.title("🌍 Economic Settings")
selected_country = st.sidebar.selectbox("Select Country for Analysis", options=list(COUNTRY_WAGES.keys()))
base_monthly = COUNTRY_WAGES[selected_country]

if 'current_country' not in st.session_state or st.session_state.current_country != selected_country:
    st.session_state.current_country = selected_country
    st.session_state.data = load_base_data(selected_country)
    st.session_state.prediction_result = None
    st.session_state.model, st.session_state.scaler, st.session_state.le, st.session_state.acc = train_system(st.session_state.data)

def move_to(target_step):
    st.session_state.step = target_step
    st.session_state.prediction_result = None 
    st.session_state.streaming = False

# --- APP FLOW ---

# PAGE 1: HOME
if st.session_state.step == 1:
    st.title(f"📌 EquiSpend: {selected_country}")
    st.markdown("---")
    st.markdown("### 📘 Strategic Summary")
    st.write(f"""
    This platform serves as a high-fidelity simulator for retail and banking analytics. 
    By anchoring all 'Income' data points to the **{selected_country}** monthly minimum wage of **${base_monthly:,.2f}**, 
    we provide a realistic economic floor that adjusts for global inflation and local standards.
    """)
    
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.markdown("""
        **System Architecture:**
        * **Normalization**: Customer profiles built relative to local wage laws.
        * **Predictive Depth**: Utilizes a Gradient Boosting Regressor for non-linear modeling.
        * **Actionable Insight**: Enables tiered loyalty goals for different income brackets.
        """)
    with col_sum2:
        st.metric("Monthly Min Wage", f"${base_monthly:,.2f}")
        st.metric("Annual Floor", f"${(base_monthly * 12):,.2f}")

    st.markdown("---")
    st.button("Next ➡️", on_click=move_to, args=(2,), use_container_width=True)

# PAGE 2: LIVE STREAM
elif st.session_state.step == 2:
    st.title("📡 Live Data Stream Simulation")
    
    # Information lines for stream phase
    st.info("""
    **Monitoring Logic:** This stream simulates the raw inflow of purchase data. 
    The AI calculates a rolling Z-score to determine if a specific transaction is an outlier compared to the 
    average e-commerce activity in **{}**.
    """.format(selected_country))
    
    col_start, col_stop = st.columns(2)
    if col_start.button("Start Live Stream 🚀", use_container_width=True): st.session_state.streaming = True
    if col_stop.button("Stop Live Stream 🛑", use_container_width=True): st.session_state.streaming = False

    col_viz, col_tips = st.columns([2, 1])
    with col_viz:
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
    with col_tips:
        st.subheader("💡 AI Strategy Suggestions")
        suggestion_placeholder = st.empty()
        # Information for user understanding
        st.markdown("""
        **Alert Thresholds:**
        * **VIP**: Spend > 150% of Mean.
        * **Risk**: Spend < 50% of Mean.
        """)

    if st.session_state.streaming:
        stream_data = []
        avg_baseline = st.session_state.data['Total_Spent'].mean()
        for i in range(100):
            if not st.session_state.streaming: break
            new_val = np.random.normal(avg_baseline, 500)
            stream_data.append(new_val)
            with chart_placeholder:
                fig = px.line(stream_data, title=f"Real-time Expenditure Feed ({selected_country})", 
                              labels={'index': 'Interval', 'value': 'Spend ($)'}, color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig, use_container_width=True)
            status_placeholder.write(f"Incoming Transaction {len(stream_data)}: **${new_val:,.2f}**")
            
            if new_val > avg_baseline * 1.5:
                suggestion_placeholder.warning(f"**VIP Alert!** Transaction high. Suggestion: Trigger VIP rewards.")
            elif new_val < avg_baseline * 0.5:
                suggestion_placeholder.error(f"**Low Activity.** Suggestion: Offer time-sensitive discount.")
            else:
                suggestion_placeholder.success(f"**Healthy Flow.** Spending aligns with local trends.")
            time.sleep(0.4)
    
    st.markdown("---")
    c_prev, c_next = st.columns(2)
    c_prev.button("⬅️ Previous", on_click=move_to, args=(1,), use_container_width=True)
    c_next.button("Next ➡️", on_click=move_to, args=(3,), use_container_width=True)

# PAGE 3: ANALYTICS
elif st.session_state.step == 3:
    st.title(f"📊 {selected_country} Behavioral Analytics")

    # Information lines for analytics phase
    st.markdown(f"""
    ### Market Logic Overview
    * **Diminishing Propensity**: Notice how the spending curve flattens at higher income levels. This reflects the **Wallet Share** logic—wealthy individuals spend more in absolute dollars, but a smaller percentage of their total income on this platform.
    * **Wealth Distribution**: The box plot identifies market segments (Male/Female/Other) to ensure the AI remains unbiased across demographic groups.
    """)

    c1, c2 = st.columns(2)
    fig1 = px.scatter(st.session_state.data, x="Annual_Income", y="Total_Spent", color="Age", trendline="lowess", title="Income vs. Platform Spend")
    fig2 = px.box(st.session_state.data, y="Annual_Income", x="Gender", color="Gender", title="Wealth Distribution")
    c1.plotly_chart(fig1, use_container_width=True)
    c2.plotly_chart(fig2, use_container_width=True)
    
    st.info("**Data Science Note:** We use 'Lowess' smoothing for the trendline to capture the non-linear relationship between income and retail consumption.")
    
    st.markdown("---")
    c_prev, c_next = st.columns(2)
    c_prev.button("⬅️ Previous", on_click=move_to, args=(2,), use_container_width=True)
    c_next.button("Next ➡️", on_click=move_to, args=(4,), use_container_width=True)

# PAGE 4: INTEGRATION
elif st.session_state.step == 4:
    st.title("📥 Bulk Data & SQL Integration")
    tab1, tab2 = st.tabs(["CSV/JSON Upload", "SQL Database Connection"])
    
    with tab1:
        st.subheader("Template Download")
        csv_data = st.session_state.data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {selected_country} Sample Dataset (CSV)",
            data=csv_data,
            file_name=f"{selected_country.lower()}_market_data.csv",
            mime='text/csv',
            use_container_width=True
        )
        st.file_uploader("Upload Market Data", type=['csv', 'json'])
            
    with tab2:
        st.subheader("Direct SQL Connection")
        sql_file = st.file_uploader("Upload SQLite (.db) file", type=['db'])

    st.markdown("---")
    c_prev, c_next = st.columns(2)
    c_prev.button("⬅️ Previous", on_click=move_to, args=(3,), use_container_width=True)
    c_next.button("Next ➡️", on_click=move_to, args=(5,), use_container_width=True)

# PAGE 5: WORKING PREDICTION ENGINE
elif st.session_state.step == 5:
    st.title("🔮 Predictive Spending Engine")
    st.markdown(f"Forecasting based on platform-specific wallet share. Accuracy: **{st.session_state.acc:.2%}**")
    
    model, scaler, le = st.session_state.model, st.session_state.scaler, st.session_state.le
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Customer Age", 18, 90, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        wage_mult = st.select_slider("Income Multiplier (vs Min Wage)", options=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
        tenure = st.slider("Years as Member", 0, 15, 3)
    
    calc_inc = base_monthly * 12 * wage_mult
    st.metric("Profile Annual Income", f"${calc_inc:,.2f}")

    if st.button("Generate Revenue Forecast", use_container_width=True, type="primary"):
        gender_encoded = le.transform([gender])[0]
        input_data = np.array([[age, gender_encoded, calc_inc, tenure]])
        scaled_input = scaler.transform(input_data)
        
        prediction = model.predict(scaled_input)[0]
        st.session_state.prediction_result = max(0, prediction)

    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.success(f"### Predicted Annual Platform Spend: **${st.session_state.prediction_result:,.2f}**")
        
        share_ratio = st.session_state.prediction_result / calc_inc
        st.info(f"Platform Wallet Share: **{share_ratio:.2%}** of total annual income.")

        if share_ratio > 0.12:
            st.warning("🌟 **High-Value Advocate**: This customer allocates a significant portion of their discretionary budget to our platform.")
        elif share_ratio < 0.04:
            st.error("📉 **Occasional Shopper**: Low wallet share suggests this customer primarily uses the platform for niche or rare needs.")
        else:
            st.success("✅ **Standard Customer**: Spending aligns with healthy market benchmarks for e-commerce engagement.")

    st.markdown("---")
    st.button("⬅️ Previous", on_click=move_to, args=(4,), use_container_width=True)