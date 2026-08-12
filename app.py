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
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100
)
gender = st.selectbox(
    "Gender",
    ["female", "male"]
)
education = st.selectbox(
    "Education",
    [
        "Associate",
        "Bachelor",
        "Doctorate",
        "High School",
        "Master"
    ]
)


income = st.number_input("Annual Income")

home = st.selectbox(
    "Home Ownership",
    ["MORTAGE","OTHER","OWN","RENT"]
)

employment = st.number_input("Employment Experience")

loan_intent = st.selectbox(
    "Loan intent",
    ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL",
 "VENTURE"]
)

loan_amount = st.number_input("Loan Amount")

loan_percent_income = st.number_input("Loan Percent Income")

interest = st.number_input("Interest Rate")

credit_history=st.selectbox(" credit history",[0,1])

prevoius_loan_default = st.selectbox(
    "Previous Default",
    [0, 1])

credit_length = st.number_input("Credit History Length")

credit_score = st.number_input("Credit Score")


# Prediction
if st.button("Predict Loan Status"):

    # Encode categorical variables
    gender_encoded = encoders["person_gender"].transform([gender])[0]
    education_encoded = encoders["person_education"].transform([education])[0]
    home_encoded = encoders["person_home_ownership"].transform([home])[0]
    loan_intent_encoded = encoders["loan_intent"].transform([loan_intent])[0]

    # Create input data
    input_data = np.array([[
        age,
        gender_encoded,
        education_encoded,
        income,
        home_encoded,
        employment,
        loan_intent_encoded,
        loan_amount,
        interest,
        credit_history,
        prevoius_loan_default,
        credit_length,
        credit_score
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