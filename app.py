import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model, scaler, and encoders
scale = joblib.load('scaler.pkl')
model = joblib.load('Disease_Risk_model.pkl')
gender_le = joblib.load('gender_le.pkl')
smoking_le = joblib.load('smoking_le.pkl')
activity_le = joblib.load('activity_le.pkl')

st.title("Disease Risk Prediction App")

with st.form("risk_form"):
    Age = st.number_input("Age", min_value=0, max_value=120, value=0)
    Gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    BMI = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
    Smoking_Status = st.selectbox("Smoking Status", options=["Never", "Former", "Current"])
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", options=["Moderate", "High"])
    Physical_Activity_Level = st.selectbox("Physical Activity Level", options=["Low", "Moderate", "High"])
    Blood_Pressure_Systolic = st.number_input("Systolic Blood Pressure", min_value=0, max_value=200, value=0)
    Blood_Pressure_Diastolic = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=130, value=0)
    Cholesterol_Level = st.number_input("Cholesterol Level", min_value=0, max_value=300, value=0)
    Glucose_Level = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
    Family_History = st.selectbox("Family History of Disease", options=["No", "Yes"])
    Genetic_Risk_Score = st.number_input("Genetic Risk Score", min_value=0.0, max_value=1.0, value=0.0)
    Previous_Diagnosis = st.selectbox("Previous Diagnosis", options=["Pre-disease", "Diagnosed"])

    if st.form_submit_button("Predict Risk"):
        # Prepare the input data as DataFrame
        input_data = pd.DataFrame({
            'Age': [Age],
            'Gender': [Gender],
            'BMI': [BMI],
            'Smoking_Status': [Smoking_Status],
            'Alcohol_Consumption': [Alcohol_Consumption],
            'Physical_Activity_Level': [Physical_Activity_Level],
            'Blood_Pressure_Systolic': [Blood_Pressure_Systolic],
            'Blood_Pressure_Diastolic': [Blood_Pressure_Diastolic],
            'Cholesterol_Level': [Cholesterol_Level],
            'Glucose_Level': [Glucose_Level],
            'Family_History': [Family_History],
            'Genetic_Risk_Score': [Genetic_Risk_Score],
            'Previous_Diagnosis': [Previous_Diagnosis]
        })

        # Map binary categorical features
        input_data['Alcohol_Consumption'] = input_data['Alcohol_Consumption'].map({'Moderate': 0, 'High': 1})
        input_data['Family_History'] = input_data['Family_History'].map({'No':0, 'Yes':1})
        input_data['Previous_Diagnosis'] = input_data['Previous_Diagnosis'].map({'Pre-disease': 0, 'Diagnosed': 1})

        # Encode LabelEncoder features (single row)
        input_data['Gender'] = gender_le.transform([Gender])[0]
        input_data['Smoking_Status'] = smoking_le.transform([Smoking_Status])[0]
        input_data['Physical_Activity_Level'] = activity_le.transform([Physical_Activity_Level])[0]

        # Scale numerical inputs
        input_data_scaled = scale.transform(input_data)


        # Predict disease risk
        risk_prediction = model.predict(input_data_scaled)

        # Display result
        st.write(f"Predicted Disease Risk: {'Yes' if risk_prediction[0]==1 else 'No'}")
