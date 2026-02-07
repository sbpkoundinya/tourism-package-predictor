# ===============================
# Imports
# ===============================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ===============================
# Hugging Face setup
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

DATASET_PATH = "hf://datasets/sbpkoundinya/tourism-package-predictor/tourism.csv"
DATASET_REPO_ID = "sbpkoundinya/tourism-package-predictor"

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ===============================
# Drop identifier column
# ===============================
df.drop(columns=["CustomerID"], inplace=True)

# ===============================
# Feature groups
# ===============================
nominal_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation"
]

ordinal_cols = [
    "CityTier",
    "PreferredPropertyStar"
]

binary_cols = [
    "Passport",
    "OwnCar"
]

numeric_cols = [
    "Age",
    "NumberOfPersonVisiting",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

target_col = "ProdTaken"

# ===============================
# Handle missing values
# ===============================
for col in nominal_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ordinal_cols + numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in binary_cols:
    df[col] = df[col].fillna(0)

# ===============================
# Train–test split
# ===============================
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Save locally
# ===============================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train/test splits saved locally.")

# ===============================
# Upload to Hugging Face Dataset
# ===============================
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset"
    )

print("Train/test datasets uploaded to Hugging Face successfully.")
