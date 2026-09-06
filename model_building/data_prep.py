
import os, io
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN")

def load_data():
    if DATASET_REPO and HF_TOKEN:
        path = hf_hub_download(repo_id=DATASET_REPO, filename="tourism.csv",
                               repo_type="dataset", token=HF_TOKEN)
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "tourism.csv")
    return pd.read_csv(path)

df = load_data()
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")
df = df.drop_duplicates().reset_index(drop=True)

# Standardize obvious text fields
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
train = X_train.copy()
train["ProdTaken"] = y_train.values
test = X_test.copy()
test["ProdTaken"] = y_test.values

out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_dir, exist_ok=True)
train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

if DATASET_REPO and HF_TOKEN:
    api = HfApi(token=HF_TOKEN)
    api.upload_file(path_or_fileobj=os.path.join(out_dir, "train.csv"),
                    path_in_repo="train.csv", repo_id=DATASET_REPO,
                    repo_type="dataset", commit_message="Upload prepared training data")
    api.upload_file(path_or_fileobj=os.path.join(out_dir, "test.csv"),
                    path_in_repo="test.csv", repo_id=DATASET_REPO,
                    repo_type="dataset", commit_message="Upload prepared test data")
print(f"Prepared data: {train.shape[0]} training rows, {test.shape[0]} test rows.")
