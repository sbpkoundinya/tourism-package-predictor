"""Model training with experimentation tracking and HF model registration.
- Loads prepared train/test splits (prefer HF dataset hub).
- Performs simple hyperparameter tuning for RandomForest.
- Logs all tuned parameters and metrics to MLflow.
- Saves best model locally and registers it to HF model hub.
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from huggingface_hub import hf_hub_download, HfApi
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# CLI args (allow CI to pass values) with env var fallback
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dataset-repo", help="Hugging Face dataset repo (owner/name)")
parser.add_argument("--model-repo", help="Hugging Face model repo (owner/name)")
parser.add_argument("--hf-token", help="Hugging Face token")
args, _ = parser.parse_known_args()
DATASET_REPO = args.dataset_repo or os.environ.get("HF_DATASET_REPO")
MODEL_REPO = args.model_repo or os.environ.get("HF_MODEL_REPO")
HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN")


def hf_or_local_dataset(filename: str, local_path: str) -> str:
    """Return a local path to the given artifact, preferring HF dataset hub when configured."""
    if DATASET_REPO and HF_TOKEN:
        try:
            path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            print(f"Loaded {filename} from HF dataset repo: {DATASET_REPO}")
            return path
        except Exception as e:
            print(f"Failed to load {filename} from HF dataset repo ({e}); falling back to {local_path}")
    if os.path.exists(local_path):
        return local_path
    raise FileNotFoundError(f"{filename} not found on HF dataset hub or at {local_path}")


def main() -> None:
    # Load prepared data and metadata
    train_csv = hf_or_local_dataset("train_prepared.csv", "tourism_project/model_building/train_prepared.csv")
    test_csv = hf_or_local_dataset("test_prepared.csv", "tourism_project/model_building/test_prepared.csv")
    feature_cols_path = hf_or_local_dataset("feature_cols.joblib", "tourism_project/model_building/feature_cols.joblib")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    feature_cols = joblib.load(feature_cols_path)

    X_train = train_df[feature_cols]
    y_train = train_df["ProdTaken"]
    X_test = test_df[feature_cols]
    y_test = test_df["ProdTaken"]

    # MLflow setup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("tourism_wellness_package")

    param_grid = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 100, "max_depth": None},
    ]

    best_auc = -1.0
    best_model = None
    best_params = None
    best_metrics = None

    with mlflow.start_run(run_name="rf_wellness_tuning"):
        for params in param_grid:
            with mlflow.start_run(run_name=f"rf_{params['n_estimators']}_{params['max_depth']}", nested=True):
                model = RandomForestClassifier(random_state=42, **params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                roc_auc = roc_auc_score(y_test, y_proba)

                mlflow.log_params(params)
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                })

                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_model = model
                    best_params = params
                    best_metrics = (accuracy, f1, roc_auc)

        # Log best params/metrics on the parent run
        if best_params is not None:
            acc, f1_val, roc_auc_val = best_metrics
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metrics({
                "best_accuracy": acc,
                "best_f1_score": f1_val,
                "best_roc_auc": roc_auc_val,
            })

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    os.makedirs("tourism_project/model_building", exist_ok=True)
    model_path = "tourism_project/model_building/model.joblib"
    joblib.dump(best_model, model_path)

    print("Best model training complete.")
    print("Best params:", best_params)
    print("Classification report:")
    y_best_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_best_pred, target_names=["No Purchase", "Purchase"]))
    print(f"Best ROC-AUC: {best_auc:.4f}")

    # Register best model in HF model hub
    if MODEL_REPO and HF_TOKEN:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.joblib",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"Uploaded best model to HF model repo: {MODEL_REPO}")
    else:
        print("HF_MODEL_REPO or HF_TOKEN not set; skipping model upload to HF model hub.")


if __name__ == "__main__":
    main()
