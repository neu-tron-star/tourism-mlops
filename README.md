# Visit with Us — Wellness Tourism Purchase Prediction

End-to-end MLOps project using scikit-learn/XGBoost, MLflow, Hugging Face Hub, GitHub Actions and a public Hugging Face frontend.

## Important deployment note

Hugging Face's current Space creation UI may show Docker as paid for free personal accounts. The project therefore includes **both**:

- `deployment/streamlit_app.py` — the Streamlit frontend required by the assignment and the Docker configuration.
- `deployment/app.py` — a Gradio frontend used for the free Hugging Face Space deployment when Docker is unavailable.

The assignment's required `Dockerfile` remains in `deployment/Dockerfile`. If Docker Spaces are enabled on your HF account, the Dockerfile can be used directly. Otherwise, use the included Gradio Space deployment.

## Required GitHub secrets

- `HF_TOKEN`
- `HF_DATASET_REPO` — e.g. `Ms21063/tourism-dataset`
- `HF_MODEL_REPO` — e.g. `Ms21063/tourism-purchase-model`
- `HF_SPACE_REPO` — e.g. `Ms21063/tourism-wellness-predictor`

## Local run

```bash
pip install -r deployment/requirements.txt
python model_building/data_prep.py
python model_building/train.py
cd deployment
streamlit run streamlit_app.py
```

For the free Hugging Face frontend:

```bash
cd deployment/hf_space
pip install -r requirements.txt
python app.py
```

## Executed benchmark

On the supplied dataset (4,128 rows), the executed notebook selected Random Forest by test ROC-AUC (0.9681). Test accuracy was 0.8991, precision 0.9375, recall 0.5097 and F1 0.6611.
