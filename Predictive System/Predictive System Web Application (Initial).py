import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('D:/Python Projects/Heart Disease Prediction System/Google Colab/trained_model_1.sav', 'rb'))

# Creating a function to make predictions
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)

    if prediction[0] == 0:
        return "The person does not have a Heart Disease"
    else:
        return "The person has a Heart Disease"

# Main function
def main():
    # Page title
    st.title("Heart Disease Prediction System")

    # Collecting user input
    age = st.number_input("Age (in years)", min_value=1, max_value=120, value=30)
    
    sex = st.selectbox("Sex", ("Male", "Female"))
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest Pain Type", ("0 : Typical Angina", "1 : Atypical Angina", "2 : Non-anginal Pain", "3 : Asymptomatic"))
    cp = int(cp[0])

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("0 : False", "1 : True"))
    fbs = int(fbs[0])

    restecg = st.selectbox("Resting ECG Results", ("0 : Normal", "1 : ST-T wave abnormality", "2 : Left ventricular hypertrophy"))
    restecg = int(restecg[0])

    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    
    exang = st.selectbox("Exercise Induced Angina", ("0 : No", "1 : Yes"))
    exang = int(exang[0])

    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ("0 : Upsloping", "1 : Flat", "2 : Downsloping"))
    slope = int(slope[0])

    ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)
    
    thal = st.selectbox("Thalassemia", ("1 : Normal", "2 : Fixed Defect", "3 : Reversible Defect"))
    thal = int(thal[0])

    # Prediction
    diagnosis = ''
    if st.button('Predict Heart Disease'):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        diagnosis = heart_disease_prediction(input_data)
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
