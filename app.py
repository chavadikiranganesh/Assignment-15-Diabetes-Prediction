import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

@st.cache_resource
def load_models():
    lgbm = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
    xgb = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    return lgbm, xgb

st.title("🩺 Diabetes Prediction System")
st.write("LightGBM and XGBoost based diabetes classification.")

try:
    lgbm_model, xgb_model = load_models()
except Exception as e:
    st.error("Model files are missing. Please run the training script before starting the app.")
    st.stop()

st.sidebar.header("Prediction Settings")
model_name = st.sidebar.selectbox("Choose model", ["LightGBM", "XGBoost"])

st.subheader("Enter Patient Information")

c1, c2 = st.columns(2)
with c1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=250.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
with c2:
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

input_df = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age
}])

if st.button("Predict Diabetes", type="primary", use_container_width=True):
    model = lgbm_model if model_name == "LightGBM" else xgb_model
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    if prediction == 1:
        st.error(f"Prediction: Diabetes (class 1)\n\nEstimated probability: {probability:.2%}")
    else:
        st.success(f"Prediction: No Diabetes (class 0)\n\nEstimated probability: {probability:.2%}")

st.caption("")
