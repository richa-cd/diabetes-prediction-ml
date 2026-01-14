# Diabetes Prediction using Classical Machine Learning

## Overview
This project focuses on predicting whether a person is diabetic or non-diabetic using classical supervised machine learning techniques. The project is built using the Pima Indians Diabetes Dataset and applies Logistic Regression and Linear Discriminant Analysis (LDA).

To improve healthcare reliability, both models are combined using an ensemble approach that prioritizes recall and reduces missed diabetic cases.


## Problem Statement
Early detection of diabetes is critical for effective healthcare management. Medical diagnostic data contains complex relationships between multiple features such as glucose level, BMI, age, and blood pressure. This project aims to develop a machine learning–based decision support system to predict diabetes status.


## Dataset
- **Name:** Pima Indians Diabetes Dataset  
- **Number of Features:** 8  
- **Target Variable:** Outcome  
  - 0 → Non-Diabetic  
  - 1 → Diabetic  

**Features used:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age


## Methodology
1. Data preprocessing and handling invalid zero values
2. Feature scaling using StandardScaler
3. Model training using:
   - Logistic Regression
   - Linear Discriminant Analysis (LDA)
4. Model evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC Curve and AUC
5. Model comparison using ROC curves
6. Ensemble strategy combining Logistic Regression and LDA using OR logic to improve recall


## Key Results
- Logistic Regression achieved higher recall
- LDA achieved higher AUC, indicating better class separation
- The combined model reduced false negatives and improved recall, making it more suitable for healthcare screening


## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit


## Deployment
The trained models were deployed using Streamlit as a web application. Users can input patient details and receive real-time predictions using a recall-focused ensemble model.


## Ethical Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis or treatment.

## Deployment

The trained machine learning models were deployed using Streamlit Cloud.
The deployment process involved serializing trained models, managing dependencies,
and hosting the application as a web-based interface.

The application allows users to input medical parameters and receive real-time
diabetes prediction using an ensemble of Logistic Regression and LDA models.

Live Application:
https://diabetes-prediction-ml-suhpleqhaqttbmkgkwwmvj.streamlit.app/

## Author
Richa Dhiman
