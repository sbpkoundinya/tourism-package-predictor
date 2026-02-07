
# ===============================
# Imports
# ===============================

# for data manipulation
import pandas as pd

# for preprocessing and pipeline creation
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# for model training, tuning, and evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# for model serialization
import joblib

# for creating folders
import os

# for Hugging Face authentication & upload
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# for experiment tracking
import mlflow
import mlflow.sklearn

# ===============================
# MLflow Configuration
# ===============================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wellness-tourism-purchase-prediction")

# ===============================
# Hugging Face API
# ===============================
api = HfApi()

# ===============================
# Load data from Hugging Face
# ===============================
Xtrain_path = "hf://datasets/sbpkoundinya/tourism-package-predictor/Xtrain.csv"
Xtest_path  = "hf://datasets/sbpkoundinya/tourism-package-predictor/Xtest.csv"
ytrain_path = "hf://datasets/sbpkoundinya/tourism-package-predictor/ytrain.csv"
ytest_path  = "hf://datasets/sbpkoundinya/tourism-package-predictor/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest  = pd.read_csv(ytest_path).values.ravel()

print("Training and testing data loaded successfully")

# ===============================
# Feature Groups
# ===============================
numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

nominal_categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation"
]

ordinal_features = [
    "CityTier",
    "PreferredPropertyStar"
]

binary_features = [
    "Passport",
    "OwnCar"
]

# ===============================
# Preprocessor
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("nom_cat", OneHotEncoder(handle_unknown="ignore"), nominal_categorical_features),
        ("ord", "passthrough", ordinal_features),
        ("bin", "passthrough", binary_features)
    ]
)

# ===============================
# Base Model
# ===============================
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# ===============================
# Hyperparameter Grid
# ===============================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8]
}

# ===============================
# Pipeline
# ===============================
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)

# ===============================
# Training + Experiment Tracking
# ===============================
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_roc_auc", results["mean_test_score"][i])

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    y_pred = best_model.predict(Xtest)
    y_proba = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    accuracy = accuracy_score(ytest, y_pred)
    roc_auc  = roc_auc_score(ytest, y_proba)
    f1       = f1_score(ytest, y_pred)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "f1_score": f1
    })

    # ===============================
    # Save Model
    # ===============================
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/wellness_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print("Model saved and logged to MLflow")

# ===============================
# Upload Model to Hugging Face Hub
# ===============================
repo_id = "<HF_USERNAME>/wellness-tourism-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists")
except RepositoryNotFoundError:
    print(f"Creating model repo '{repo_id}'")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="wellness_tourism_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Model uploaded to Hugging Face Model Hub")
