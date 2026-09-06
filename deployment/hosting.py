import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO = os.getenv("HF_SPACE_REPO")
if not HF_TOKEN or not SPACE_REPO:
    raise RuntimeError("Set HF_TOKEN and HF_SPACE_REPO (e.g. Ms21063/tourism-wellness-predictor).")

api = HfApi(token=HF_TOKEN)
api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(
    folder_path="deployment/hf_space",
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy tourism purchase predictor frontend",
)
print("Gradio frontend pushed to:", SPACE_REPO)
