import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('D:/Python Projects/Heart Disease Prediction System/Google Colab/trained_model_1.sav', 'rb'))

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)
    return "The person has a Heart Disease" if prediction[0] == 1 else "The person does not have a Heart Disease"

def main():
    st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

    st.title("Heart Disease Prediction System")
    st.write("Provide the following information to predict the likelihood of heart disease")

    # Personal Info
    st.header("Personal Information")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, help="Enter your age in years")
    sex = st.selectbox("Sex", ("Male", "Female"), help="Select your biological sex")
    sex = 1 if sex == "Male" else 0

    # Clinical Measurements in two columns
    st.header("Clinical Measurements")
    col1, col2 = st.columns(2)

    cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}

    fbs_dict = {"False": 0, "True": 1}

    restecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}

    exang_dict = {"No": 0, "Yes": 1}

    slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    with col1:
        cp = st.selectbox("Chest Pain Type", list(cp_dict.keys()), help="Type of chest pain experienced")
        cp = cp_dict[cp]

        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, help="Measured in millimeters of mercury (mm Hg)")

        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Serum cholesterol in mg/dl")

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_dict.keys()), help="Is fasting blood sugar > 120 mg/dl?")
        fbs = fbs_dict[fbs]

        restecg = st.selectbox("Resting ECG Results", list(restecg_dict.keys()), help="Resting electrocardiographic results")
        restecg = restecg_dict[restecg]

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150, help="Maximum heart rate achieved during exercise")

        exang = st.selectbox("Exercise Induced Angina", list(exang_dict.keys()), help="Whether exercise induced angina is present")
        exang = exang_dict[exang]

        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, format="%.1f", help="ST depression caused by exercise relative to rest")

        slope = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_dict.keys()), help="The slope of the peak exercise ST segment")
        slope = slope_dict[slope]

        ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0, help="Number of major vessels colored by fluoroscopy")

        thal = st.selectbox("Thalassemia", list(thal_dict.keys()), help="Type of thalassemia condition")
        thal = thal_dict[thal]

    if st.button("Predict Heart Disease"):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        diagnosis = heart_disease_prediction(input_data)
        st.success(diagnosis)

if __name__ == "__main__":
    main()
