# Diabetes Prediction using Classical Machine Learning

## Overview
This project focuses on predicting whether a person is diabetic or non-diabetic using classical supervised machine learning techniques. The Pima Indians Diabetes Dataset is used to build, evaluate, and compare Logistic Regression and Linear Discriminant Analysis (LDA) models.

To improve healthcare reliability, an ensemble strategy is implemented to prioritize recall and reduce missed diabetic cases.



## Problem Statement
Early detection of diabetes is crucial for effective healthcare management. Medical diagnostic data contains complex relationships between multiple features such as glucose level, BMI, age, and blood pressure. This project aims to develop a reliable machine learning-based decision support system for diabetes screening.



## Dataset
- **Name:** Pima Indians Diabetes Dataset  
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
- **Target:** Outcome (0 = Non-Diabetic, 1 = Diabetic)



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
5. Ensemble strategy combining LR and LDA using OR logic to improve recall



## Key Results
- Logistic Regression achieved better recall
- LDA achieved higher AUC (better class separation)
- The combined model reduced false negatives and improved healthcare reliability


## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit (for deployment)


## Deployment
The trained models were deployed using Streamlit to provide an interactive web application where users can input patient details and receive diabetes predictions.



## Ethical Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis.


## Author
Richa Dhiman
