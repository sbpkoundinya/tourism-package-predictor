
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ===============================
# Load trained model from Hugging Face
# ===============================
model_path = hf_hub_download(
    repo_id="sbpkoundinya/tourism-package-predictor",
    filename="wellness_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

# ===============================
# Streamlit UI
# ===============================
st.title("Wellness Tourism Package Purchase Prediction")

st.write("""
This application predicts whether a customer is **likely to purchase**
the **Wellness Tourism Package** based on customer demographics and
interaction details.

Please provide the customer information below.
""")

# ===============================
# User Inputs
# ===============================

# Categorical inputs
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)
designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "VP", "AVP"]
)

# Ordinal inputs
city_tier = st.selectbox("City Tier", [1, 2, 3])
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

# Binary inputs
passport = st.selectbox("Passport Available", [0, 1])
own_car = st.selectbox("Owns a Car", [0, 1])

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=3)
num_children = st.number_input("Number of Children (Below 5)", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=50000, step=1000)
pitch_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=20)

# ===============================
# Assemble Input DataFrame
# ===============================
input_data = pd.DataFrame([{
    "TypeofContact": typeof_contact,
    "Occupation": occupation,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "ProductPitched": product_pitched,
    "Designation": designation,
    "CityTier": city_tier,
    "PreferredPropertyStar": preferred_property_star,
    "Passport": passport,
    "OwnCar": own_car,
    "Age": age,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfTrips": num_trips,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch
}])

# ===============================
# Prediction
# ===============================
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ Customer is **LIKELY** to purchase the package\n\n"
            f"**Purchase Probability:** {probability:.2%}"
        )
    else:
        st.warning(
            f"❌ Customer is **UNLIKELY** to purchase the package\n\n"
            f"**Purchase Probability:** {probability:.2%}"
        )
