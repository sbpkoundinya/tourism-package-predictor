"""Register local tourism.csv to Hugging Face dataset hub.
- Expects HF_DATASET_REPO and HF_TOKEN to be set (in CI).
- Looks for tourism.csv under tourism_project/data/ or current directory.
"""
import os
from huggingface_hub import HfApi


def main() -> None:
    # Allow passing via CLI or environment variables
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-repo", help="Hugging Face dataset repo (owner/name)")
    parser.add_argument("--hf-token", help="Hugging Face token")
    args, _ = parser.parse_known_args()

    dataset_repo = args.dataset_repo or os.environ.get("HF_DATASET_REPO")
    token = args.hf_token or os.environ.get("HF_TOKEN")

    if not dataset_repo or not token:
        raise ValueError("HF_DATASET_REPO and HF_TOKEN must be set to register the dataset. Provide via --dataset-repo/--hf-token or environment variables.")

    candidates = [
        "tourism_project/data/tourism.csv",
        "tourism.csv",
    ]
    local_path = None
    for p in candidates:
        if os.path.exists(p):
            local_path = p
            break

    if local_path is None:
        raise FileNotFoundError("tourism.csv not found locally. Place it under tourism_project/data/ or project root.")

    api = HfApi(token=token)
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="tourism.csv",
        repo_id=dataset_repo,
        repo_type="dataset",
    )

    print(f"Uploaded {local_path} to dataset repo {dataset_repo} as tourism.csv")


if __name__ == "__main__":
    main()
