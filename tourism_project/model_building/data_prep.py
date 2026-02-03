"""Data preparation script.
- Loads raw data (preferably from Hugging Face dataset hub).
- Cleans and encodes features.
- Splits into train/test and saves locally.
- Uploads prepared splits and metadata back to the HF dataset hub.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi
import argparse

# CLI args (allow CI to pass values) with env var fallback
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dataset-repo", help="Hugging Face dataset repo (owner/name)")
parser.add_argument("--hf-token", help="Hugging Face token")
args, _ = parser.parse_known_args()
DATASET_REPO = args.dataset_repo or os.environ.get("HF_DATASET_REPO")
HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN")


def load_raw_data() -> pd.DataFrame:
    """Load raw tourism dataset, preferring HF dataset hub if configured."""
    if DATASET_REPO and HF_TOKEN:
        try:
            csv_path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename="tourism.csv",
                repo_type="dataset",
                token=HF_TOKEN,
            )
            print(f"Loaded dataset from HF dataset repo: {DATASET_REPO}")
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to load from HF dataset repo ({e}), falling back to local file.")

    # Local fallbacks for development
    for path in [
        "tourism_project/data/tourism.csv",
        "tourism.csv",
    ]:
        if os.path.exists(path):
            print(f"Loaded dataset from local path: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("tourism.csv not found locally or on Hugging Face dataset hub.")


def main() -> None:
    df = load_raw_data()

    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    # Basic cleaning: fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            modes = df[col].mode()
            df[col] = df[col].fillna(modes.iloc[0] if len(modes) > 0 else "Unknown")

    # Encode categorical variables
    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    feature_cols = [c for c in df_encoded.columns if c not in ["ProdTaken", "CustomerID"]]
    X = df_encoded[feature_cols]
    y = df_encoded["ProdTaken"]

    # Combine into one frame with target for saving
    df_prepared = X.copy()
    df_prepared["ProdTaken"] = y

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df_prepared.drop(columns=["ProdTaken"]),
        df_prepared["ProdTaken"],
        test_size=0.2,
        random_state=42,
        stratify=df_prepared["ProdTaken"],
    )

    train_df = X_train.copy()
    train_df["ProdTaken"] = y_train
    test_df = X_test.copy()
    test_df["ProdTaken"] = y_test

    os.makedirs("tourism_project/model_building", exist_ok=True)
    train_path = "tourism_project/model_building/train_prepared.csv"
    test_path = "tourism_project/model_building/test_prepared.csv"
    encoders_path = "tourism_project/model_building/label_encoders.joblib"
    feature_cols_path = "tourism_project/model_building/feature_cols.joblib"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(label_encoders, encoders_path)
    joblib.dump(feature_cols, feature_cols_path)

    print("Data preparation complete.")
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    # Upload prepared artifacts back to HF dataset hub
    if DATASET_REPO and HF_TOKEN:
        api = HfApi(token=HF_TOKEN)
        for local, remote in [
            (train_path, "train_prepared.csv"),
            (test_path, "test_prepared.csv"),
            (encoders_path, "label_encoders.joblib"),
            (feature_cols_path, "feature_cols.joblib"),
        ]:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=DATASET_REPO,
                repo_type="dataset",
            )
        print(f"Uploaded prepared train/test and metadata to HF dataset repo: {DATASET_REPO}")
    else:
        print("HF_DATASET_REPO or HF_TOKEN not set; skipping upload to HF dataset hub.")


if __name__ == "__main__":
    main()
