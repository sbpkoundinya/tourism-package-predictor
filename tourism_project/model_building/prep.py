# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sbpkoundinya/tourism-package-predictor/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True) // CustomerID is a unique identifier and does not carry predictive information

# Nominal categorical columns (NO inherent order)
nominal_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation"
]

# Ordinal categorical columns (ORDER matters)
ordinal_cols = [
    "CityTier",               # Tier 1 > Tier 2 > Tier 3
    "PreferredPropertyStar"   # 1 to 5
]

# Binary columns (already encoded)
binary_cols = [
    "Passport",
    "OwnCar"
]

# Handle missing values
# For categorical columns
for col in nominal_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# For ordinal columns
for col in ordinal_cols:
    df[col] = df[col].fillna(df[col].median())

# Encode ordinal categorical columns
# Explicit ordinal mapping (optional but clear)
city_tier_mapping = {1: 1, 2: 2, 3: 3}
property_star_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

df["CityTier"] = df["CityTier"].map(city_tier_mapping)
df["PreferredPropertyStar"] = df["PreferredPropertyStar"].map(property_star_mapping)

# One-hot encode nominal categorical columns
df = pd.get_dummies(
    df,
    columns=nominal_cols,
    drop_first=True  # avoids dummy variable trap
)

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sbpkoundinya/tourism-package-predictor",
        repo_type="dataset",
    )
