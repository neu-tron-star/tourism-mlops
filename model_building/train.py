
import os, json, joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
train_path = os.path.join(BASE, "..", "data", "train.csv")
test_path = os.path.join(BASE, "..", "data", "test.csv")
train = pd.read_csv(train_path); test = pd.read_csv(test_path)
X_train, y_train = train.drop(columns="ProdTaken"), train["ProdTaken"]
X_test, y_test = test.drop(columns="ProdTaken"), test["ProdTaken"]

cat = X_train.select_dtypes(include="object").columns.tolist()
num = X_train.select_dtypes(exclude="object").columns.tolist()
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

models = {
"RandomForest": (RandomForestClassifier(random_state=42, class_weight="balanced"),
 {"model__n_estimators":[200,300],"model__max_depth":[None,8,12,16],
  "model__min_samples_split":[2,5,10],"model__min_samples_leaf":[1,2,4],
  "model__max_features":["sqrt","log2"]}),
"GradientBoosting": (GradientBoostingClassifier(random_state=42),
 {"model__n_estimators":[100,150,200],"model__learning_rate":[.03,.05,.1],
  "model__max_depth":[2,3,4],"model__min_samples_leaf":[1,2,4]}),
"XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=2),
 {"model__n_estimators":[150,250,350],"model__max_depth":[2,3,4,5],
  "model__learning_rate":[.03,.05,.1],"model__subsample":[.8,1.0],
  "model__colsample_bytree":[.8,1.0]})
}

os.makedirs(BASE, exist_ok=True)
rows=[]; best=None; best_auc=-1
try:
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","file:./mlruns"))
    mlflow.set_experiment("tourism_purchase_prediction")
    use_mlflow=True
except Exception:
    use_mlflow=False

for name,(est,params) in models.items():
    pipe=Pipeline([("preprocessor",pre),("model",est)])
    search=RandomizedSearchCV(pipe, params, n_iter=8, cv=3, scoring="roc_auc",
                              random_state=42, n_jobs=-1)
    if use_mlflow:
        with mlflow.start_run(run_name=name):
            search.fit(X_train,y_train)
            pred=search.predict(X_test); proba=search.predict_proba(X_test)[:,1]
            metrics={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred),
                     "recall":recall_score(y_test,pred),"f1":f1_score(y_test,pred),
                     "roc_auc":roc_auc_score(y_test,proba),"cv_roc_auc":search.best_score_}
            mlflow.log_params(search.best_params_); mlflow.log_metrics(metrics)
    else:
        search.fit(X_train,y_train)
        pred=search.predict(X_test); proba=search.predict_proba(X_test)[:,1]
        metrics={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred),
                 "recall":recall_score(y_test,pred),"f1":f1_score(y_test,pred),
                 "roc_auc":roc_auc_score(y_test,proba),"cv_roc_auc":search.best_score_}
    row={"model":name, **metrics, "best_params":str(search.best_params_)}
    rows.append(row)
    if metrics["roc_auc"] > best_auc:
        best_auc=metrics["roc_auc"]; best=search.best_estimator_

comparison=pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
comparison.to_csv(os.path.join(BASE,"model_comparison.csv"), index=False)
joblib.dump(best, os.path.join(BASE,"best_model.joblib"))

MODEL_REPO=os.getenv("HF_MODEL_REPO","")
HF_TOKEN=os.getenv("HF_TOKEN")
if MODEL_REPO and HF_TOKEN:
    from huggingface_hub import HfApi
    api=HfApi(token=HF_TOKEN)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=os.path.join(BASE,"best_model.joblib"),
                    path_in_repo="best_model.joblib", repo_id=MODEL_REPO,
                    repo_type="model", commit_message="Register best tourism model")
    api.upload_file(path_or_fileobj=os.path.join(BASE,"model_comparison.csv"),
                    path_in_repo="model_comparison.csv", repo_id=MODEL_REPO,
                    repo_type="model", commit_message="Upload experiment comparison")
print(comparison[["model","accuracy","precision","recall","f1","roc_auc","cv_roc_auc"]].to_string(index=False))
print("Best model:", comparison.iloc[0]["model"])
