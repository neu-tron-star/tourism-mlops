"""Streamlit version of the frontend, retained for the assignment requirement.
The free Hugging Face deployment uses deployment/app.py (Gradio) because the
current Hugging Face account shown in the assignment UI does not permit Docker Spaces.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Purchase Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Wellness Tourism Package Predictor")
st.write("Predict whether a customer is likely to purchase the package before contacting them.")

MODEL_REPO = os.getenv("HF_MODEL_REPO", "Ms21063/tourism-purchase-model")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib", repo_type="model")
except Exception:
    model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")
model = joblib.load(model_path)

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 100, 35)
    contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
    persons = st.number_input("Number of Persons Visiting", 1, 20, 2)
    stars = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
with c2:
    trips = st.number_input("Number of Trips", 0.0, 20.0, 3.0)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    children = st.number_input("Number of Children Visiting", 0, 10, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 1000.0, 100000.0, 50000.0)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 4)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    followups = st.number_input("Number of Followups", 0, 10, 4)
    duration = st.number_input("Duration of Pitch (minutes)", 1.0, 60.0, 15.0)

if st.button("Predict Purchase Probability", type="primary"):
    row = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city, "Occupation": occupation,
        "Gender": gender, "NumberOfPersonVisiting": persons, "PreferredPropertyStar": stars,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport, "OwnCar": car,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income, "PitchSatisfactionScore": pitch_score, "ProductPitched": product,
        "NumberOfFollowups": followups, "DurationOfPitch": duration
    }])
    prob = float(model.predict_proba(row)[:, 1][0])
    pred = int(prob >= 0.5)
    st.metric("Purchase Probability", f"{prob:.1%}")
    if pred:
        st.success("Likely to purchase — prioritize this customer.")
    else:
        st.info("Less likely to purchase — consider lower-priority outreach.")
