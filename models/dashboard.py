import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Aircraft RUL Prognostics", layout="wide")

st.title("✈️ Aircraft Engine Prognostics Dashboard")
st.markdown("Automated Remaining Useful Life Prediction with Risk Assessment")

# =============================
# LOAD MODEL
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "final_rul_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found")
    st.stop()

model = joblib.load(model_path)

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader("📂 Upload Engine Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully")

    # =============================
    # AUTO FORMAT (CMAPSS SAFE)
    # =============================
    num_cols = df.shape[1]

    df.columns = (
        ["unit_number", "setting_1", "setting_2", "time_in_cycles"] +
        [f"sensor_{i}" for i in range(1, num_cols - 3)]
    )

    # =============================
    # RUL CALCULATION
    # =============================
    df['RUL'] = df.groupby('unit_number')['time_in_cycles'].transform('max') - df['time_in_cycles']
    df['RUL'] = df['RUL'].clip(upper=125)

    # =============================
    # FEATURE ENGINEERING (MATCH TRAINING)
    # =============================
    sensor_cols = [col for col in df.columns if "sensor" in col]

    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    df.fillna(0, inplace=True)

    # =============================
    # STRICT FEATURE ALIGNMENT (AUTO FIX)
    # =============================
    model_features = list(model.feature_names_in_)

    # Add missing features as zeros
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = 0

    # Remove extra columns (only keep required)
    X = df[model_features]

    # =============================
    # PREDICTION
    # =============================
    df['Predicted_RUL'] = model.predict(X)

    # =============================
    # ENGINE SELECTION
    # =============================
    engine_ids = df['unit_number'].unique()
    selected_engine = st.selectbox("Select Engine", engine_ids)

    engine_df = df[df['unit_number'] == selected_engine]
    latest = engine_df.iloc[-1]

    # =============================
    # KPI SECTION
    # =============================
    st.subheader("📊 Engine Health Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Cycle", int(latest['time_in_cycles']))
    col2.metric("Actual RUL", int(latest['RUL']))
    col3.metric("Predicted RUL", int(latest['Predicted_RUL']))

    # =============================
    # RISK SCORE
    # =============================
    rul = latest['Predicted_RUL']
    risk_score = max(0, min(100, 100 - rul))

    st.subheader("⚠️ Risk Assessment")

    col4, col5 = st.columns(2)

    col4.metric("Risk Score", f"{risk_score:.1f}/100")

    if rul > 80:
        status = "🟢 Healthy"
        recommendation = "No immediate action required"
    elif rul > 40:
        status = "🟡 Moderate"
        recommendation = "Schedule inspection"
    else:
        status = "🔴 Critical"
        recommendation = "Immediate maintenance required"

    col5.metric("Status", status)
    st.info(f"📌 Recommendation: {recommendation}")

    # =============================
    # TREND
    # =============================
    st.subheader("📉 Degradation Trend")

    trend = engine_df[['time_in_cycles', 'RUL', 'Predicted_RUL']].tail(60)
    trend = trend.set_index('time_in_cycles')

    st.line_chart(trend)

    # =============================
    # ERROR ANALYSIS
    # =============================
    residuals = engine_df['RUL'] - engine_df['Predicted_RUL']

    st.subheader("📈 Prediction Error")

    col6, col7 = st.columns(2)
    col6.metric("Mean Error", f"{residuals.mean():.2f}")
    col7.metric("Std Deviation", f"{residuals.std():.2f}")

    # =============================
    # DOWNLOAD
    # =============================
    st.download_button(
        "📥 Download Results",
        df.to_csv(index=False),
        "rul_predictions.csv"
    )

else:
    st.info("📂 Upload dataset to start analysis")