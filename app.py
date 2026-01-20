import streamlit as st
import numpy as np
import joblib
import pandas as pd

model = joblib.load("ev_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🚗 Electric Vehicle Prediction App")

input_data = {}

for feature in feature_names:
    if feature.lower() == "cluster":
        input_data[feature] = st.selectbox(feature, [0, 1, 2])
    else:
        input_data[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    df_input = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)

    st.success(f"Price: {prediction[0][0]:,.2f}")
    st.success(f"Demand: {prediction[0][1]:.2f}")
    st.success(f"Count: {prediction[0][2]:.2f}")
