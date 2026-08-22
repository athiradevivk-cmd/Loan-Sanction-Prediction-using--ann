import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("loan_ann_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
st.write("Encoder keys:", list(encoders.keys()))
# Page title
st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details to predict loan approval")

# User inputs
person_age= st.number_input(
    "person_age",
    min_value=18,
    max_value=100
)
person_gender= st.selectbox(
    "person_gender",
    ["female", "male"]
)
person_education= st.selectbox(
    "person_education",
    [
        "Associate",
        "Bachelor",
        "Doctorate",
        "High School",
        "Master"
    ]
)


person_income= st.number_input("person_income")

person_home_ownership= st.selectbox(
    "person_home_ownership",
    ["MORTAGE","OTHER","OWN","RENT"]
)

person_emp_exp = st.number_input("Employment Experience")

loan_intent = st.selectbox(
    "Loan intent",
    ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL",
 "VENTURE"]
)

loan_amnt = st.number_input("loan_amnt")

loan_percent_income = st.number_input("Loan Percent Income")

loan_int_rate = st.number_input("Interest Rate")


previous_loan_defaults_on_file = st.selectbox(
    "Previous Default",
    ["No", "Yes"])

cb_person_cred_hist_length= st.number_input("Credit History Length")

credit_score = st.number_input("Credit Score")


# Prediction
if st.button("Predict Loan Status"):

    # Encode categorical variables
    person_gender_encoded = encoders["person_gender"].transform([person_gender])[0]
    person_education_encoded = encoders["person_education"].transform([person_education])[0]
    person_home_ownership_encoded = encoders["person_home_ownership"].transform([person_home_ownership])[0]
    loan_intent_encoded = encoders["loan_intent"].transform([loan_intent])[0]
    previous_loan_defaults_on_file=encoders["previous_loan_defaults_on_file"].transform([previous_loan_defaults_on_file])[0]

    # Create input data
    input_data = np.array([[
        person_age,
        person_gender_encoded,
        person_education_encoded,
        person_income,
        person_emp_exp,
        person_home_ownership_encoded,
        loan_amnt,
        loan_intent_encoded,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_loan_defaults_on_file,
        
      ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    if probability > 0.5:
        st.success(
            f"Loan Approved ✅ Probability: {probability:.2f}"
        )
    else:
        st.error(
            f"Loan Rejected ❌ Probability: {probability:.2f}"
        )