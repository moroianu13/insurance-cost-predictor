import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("insurance_model.pkl")

st.title("💰 Insurance Cost Predictor")
st.markdown("Enter your details to estimate your medical insurance cost.")

# Collect user input
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Encode inputs
sex_val = 1 if sex == "Male" else 2
smoker_val = 1 if smoker == "Yes" else 0
region_dict = {"Northeast": 1, "Northwest": 2, "Southeast": 3, "Southwest": 4}
region_val = region_dict[region]

# Make prediction
input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
predicted_cost = model.predict(input_data)[0]

st.subheader(f"💸 Predicted Insurance Cost: ${predicted_cost:,.2f}")
