import streamlit as st
import pandas as pd
import joblib

st.title("Cyber Threat Prediction Dashboard")

model = joblib.load('models/cyber_threat_model.pkl')

uploaded_file = st.file_uploader("Upload network data CSV")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", data.head())
    predictions = model.predict(data)
    st.write("Predicted Threat Levels:", predictions)
