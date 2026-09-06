
import os
from huggingface_hub import HfApi, create_repo
DATASET_REPO=os.getenv("HF_DATASET_REPO")
HF_TOKEN=os.getenv("HF_TOKEN")
if not DATASET_REPO or not HF_TOKEN:
    raise RuntimeError("Set HF_DATASET_REPO and HF_TOKEN.")
api=HfApi(token=HF_TOKEN)
api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
api.upload_file(path_or_fileobj="data/tourism.csv", path_in_repo="tourism.csv",
                repo_id=DATASET_REPO, repo_type="dataset",
                commit_message="Register tourism dataset")
print("Dataset registered:", DATASET_REPO)
