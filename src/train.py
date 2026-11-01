"""Step 2 - Training pipeline with MLflow tracking.
Set MLFLOW_TRACKING_URI=http://<VM_IP>:5000 to log to the cloud server; default is local sqlite file mlflow.db."""
import os
import sys
import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES, TARGET, MODEL_PATH

C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MIN_F1 = 0.99  # quality gate used by CI

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
mlflow.set_experiment("aws-mlops-demo")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

with mlflow.start_run():
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    acc, f1 = accuracy_score(test[TARGET], pred), f1_score(test[TARGET], pred)
    mlflow.log_param("C", C)
    mlflow.log_metrics({"accuracy": acc, "f1": f1})
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    print(f"C={C} accuracy={acc:.3f} f1={f1:.3f}")

if f1 < MIN_F1:
    sys.exit(f"Quality gate failed: f1 {f1:.3f} < {MIN_F1}")
