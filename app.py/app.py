import streamlit as st
import numpy as np
import joblib

# Load models
lr_model = joblib.load("model/lr_model.pkl")
lda_model = joblib.load("model/lda_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Diabetes Prediction System")
st.write("Enter patient details to assess diabetes risk")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)
    
    lr_pred = lr_model.predict(input_scaled)[0]
    lda_pred = lda_model.predict(input_scaled)[0]
    
    final_pred = 1 if (lr_pred == 1 or lda_pred == 1) else 0
    
    if final_pred == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Non-Diabetic")

st.warning("This tool is for educational purposes only.")
