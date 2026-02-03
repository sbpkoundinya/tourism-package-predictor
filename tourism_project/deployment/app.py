"""Streamlit app for Wellness Tourism Package purchase prediction.
- Loads the best model from Hugging Face model hub (or local fallback).
- Loads encoders/feature_cols from HF dataset hub (or local fallback).
- Collects user inputs, builds a dataframe, and predicts purchase probability.
"""
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")  # e.g. "sbpkoundinya/tourism-wellness-model"
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")  # e.g. "sbpkoundinya/tourism-dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")


def hf_or_local_model() -> str:
    """Return path to model, preferring HF model hub when configured."""
    if HF_MODEL_REPO and HF_TOKEN:
        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="model.joblib",
                repo_type="model",
                token=HF_TOKEN,
            )
            return path
        except Exception as e:
            st.warning(f"Could not load model from HF model hub ({e}). Falling back to local file.")
    for p in [
        "model_building/model.joblib",
        "../model_building/model.joblib",
        "model.joblib",
    ]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Model file not found locally or on HF model hub.")


def hf_or_local_dataset_artifact(filename: str, local_candidates: list[str]) -> str:
    """Return path to dataset artifact, preferring HF dataset hub when configured."""
    if HF_DATASET_REPO and HF_TOKEN:
        try:
            path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            return path
        except Exception as e:
            st.warning(f"Could not load {filename} from HF dataset hub ({e}). Falling back to local file.")
    for p in local_candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{filename} not found locally or on HF dataset hub.")


@st.cache_resource
def load_artifacts():
    model_path = hf_or_local_model()
    encoders_path = hf_or_local_dataset_artifact(
        "label_encoders.joblib",
        [
            "model_building/label_encoders.joblib",
            "../model_building/label_encoders.joblib",
            "label_encoders.joblib",
        ],
    )
    feature_cols_path = hf_or_local_dataset_artifact(
        "feature_cols.joblib",
        [
            "model_building/feature_cols.joblib",
            "../model_building/feature_cols.joblib",
            "feature_cols.joblib",
        ],
    )

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    feature_cols = joblib.load(feature_cols_path)
    return model, encoders, feature_cols


st.set_page_config(page_title="Wellness Package Predictor", page_icon="✈️")
st.title("Wellness Tourism Package – Purchase Prediction")
st.markdown("Predict whether a customer will purchase the Wellness Tourism Package.")

try:
    model, encoders, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model or preprocessing artifacts: {e}")
    st.stop()

with st.form("customer_input"):
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch (mins)", min_value=1, max_value=60, value=10)
    occupation = st.selectbox(
        "Occupation",
        list(encoders["Occupation"].classes_) if "Occupation" in encoders else ["Salaried", "Free Lancer", "Small Business", "Large Business"],
    )
    gender = st.selectbox(
        "Gender",
        list(encoders["Gender"].classes_) if "Gender" in encoders else ["Male", "Female"],
    )
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox(
        "Product Pitched",
        list(encoders["ProductPitched"].classes_) if "ProductPitched" in encoders else ["Basic", "Standard", "Deluxe", "Super Deluxe"],
    )
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital = st.selectbox(
        "Marital Status",
        list(encoders["MaritalStatus"].classes_) if "MaritalStatus" in encoders else ["Single", "Married", "Divorced", "Unmarried"],
    )
    num_trips = st.number_input("Number of Trips (annual)", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Passport (1=Yes, 0=No)", [0, 1])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Own Car (1=Yes, 0=No)", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox(
        "Designation",
        list(encoders["Designation"].classes_) if "Designation" in encoders else ["Executive", "Manager", "Senior Manager"],
    )
    monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=100000, value=20000)
    typeof_contact = st.selectbox(
        "Type of Contact",
        list(encoders["TypeofContact"].classes_) if "TypeofContact" in encoders else ["Company Invited", "Self Enquiry"],
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "TypeofContact": typeof_contact,
    }
    df_input = pd.DataFrame([row])
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))
    X = df_input[feature_cols]
    proba = model.predict_proba(X)[0, 1]
    pred = 1 if proba >= 0.5 else 0
    st.success(f"**Prediction:** {'Will purchase' if pred == 1 else 'Will not purchase'} (probability: {proba:.2%})")
