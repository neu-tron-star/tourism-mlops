# Exact setup sequence

## 1. Hugging Face repositories
Create these public repositories under your HF account:

- Dataset: `tourism-dataset`
- Model: `tourism-purchase-model`
- Space: `tourism-wellness-predictor`

If your HF username is `Ms21063`, the final names are:

- `Ms21063/tourism-dataset`
- `Ms21063/tourism-purchase-model`
- `Ms21063/tourism-wellness-predictor`

## 2. Space creation
On the current HF New Space page, select **Gradio** (not Docker if Docker shows a Paid badge). Choose Blank and make the Space public. The pipeline will overwrite the Space files.

## 3. GitHub repository
Create a public GitHub repository, then push this entire project folder including `.github/workflows/pipeline.yml`.

## 4. GitHub Actions secrets
Add repository secrets:

- `HF_TOKEN` = Hugging Face write token
- `HF_DATASET_REPO` = `Ms21063/tourism-dataset`
- `HF_MODEL_REPO` = `Ms21063/tourism-purchase-model`
- `HF_SPACE_REPO` = `Ms21063/tourism-wellness-predictor`

Replace `Ms21063` if your HF username differs.

## 5. Trigger pipeline
Push to the `main` branch. GitHub Actions executes:

1. Dataset registration
2. Data preparation
3. Hyperparameter tuning + model evaluation + MLflow logging
4. Model registration
5. Gradio frontend deployment

## 6. Verify
Check all four:

- HF Dataset contains `tourism.csv`, `train.csv`, `test.csv`
- HF Model contains `best_model.joblib` and `model_comparison.csv`
- GitHub Actions workflow is green
- HF Space opens and predicts purchase probability

## 7. Submission evidence
Capture:

- GitHub repository folder structure
- Successful GitHub Actions workflow
- Hugging Face Space prediction screen

Keep the executed `Tourism_MLOps_Assignment.html` as the primary notebook submission.
