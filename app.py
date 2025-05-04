'''For running the Streamlit app, run this command in terminal:
streamlit run app.py
Then go to: http://localhost:8501 '''

import streamlit as st
import pandas as pd
import joblib
import datetime
from glob import glob
import os

# ========= Load latest model and feature list ==========
model_files = glob("task2_models/sales_model_*.pkl")
if not model_files:
    st.error("❌ No model found!")
    st.stop()

latest_model_path = max(model_files, key=os.path.getctime)
model, expected_features = joblib.load(latest_model_path)
st.sidebar.success(f"✅ Loaded model: {os.path.basename(latest_model_path)}")

# ========= App UI ==========
st.title("🛒 Rossmann Sales Prediction")

# Sidebar input fields
store_id = st.sidebar.number_input("Store ID", min_value=1, value=1)
date = st.sidebar.date_input("Date", datetime.date.today())
promo = st.sidebar.selectbox("Promo Running?", [0, 1])
state_holiday = st.sidebar.selectbox("State Holiday?", [0, 1])
school_holiday = st.sidebar.selectbox("School Holiday?", [0, 1])

# Extract date features
dow = date.weekday()
month = date.month
year = date.year
week = date.isocalendar()[1]

# Create initial input DataFrame
input_df = pd.DataFrame([{
    "Store": store_id,
    "DayOfWeek": dow,
    "Promo": promo,
    "Month": month,
    "Year": year,
    "WeekOfYear": week,
    "StateHoliday": state_holiday,
    "SchoolHoliday": school_holiday
}])

# ========= File Upload (optional) ==========
st.markdown("### 📁 Or Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

# ========= Align input to expected features ==========
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns

input_df = input_df[expected_features]  # Keep only expected columns

# ========= Predict ==========
if st.button("🔮 Predict Sales"):
    try:
        predictions = model.predict(input_df)
        input_df["PredictedSales"] = predictions.astype(int)
        st.success("✅ Prediction successful!")
        st.write(input_df)

        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predicted_sales.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
