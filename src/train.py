import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import os
from feast import FeatureStore
from datetime import datetime
import argparse
from google.cloud import storage

def upload_to_gcs(bucket_name, source_path, dest_blob):
    """Uploads a file to the specified GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_path)
    print(f"✅ Uploaded model to gs://{bucket_name}/{dest_blob}")

def train_model_with_feast(data_path: str):
    """
    Generates training data from Feast, trains a model, and logs the
    experiment to MLflow.
    """
    print("--- Training Model with Features from Feast ---")
    
    store = FeatureStore(repo_path="feature_repo")

    feature_names = [
        "heart_features:age",
        "heart_features:gender",
        "heart_features:cp",
        "heart_features:trestbps",
        "heart_features:chol",
        "heart_features:fbs",
        "heart_features:restecg",
        "heart_features:thalach",
        "heart_features:exang",
        "heart_features:oldpeak",
        "heart_features:slope",
        "heart_features:ca",
        "heart_features:thal",
    ]


    raw_data = pd.read_parquet(data_path)
    entity_df = raw_data[["patient_id", "event_timestamp", "target"]]
    entity_df.loc[:, "event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    print("📡 Retrieving training data from Feast...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_names,
    ).to_df()

    mlflow.set_experiment("OPPE2-Production-Experiment")
    with mlflow.start_run():
        X = training_df.drop(columns=["patient_id", "event_timestamp", "target"])
        y = training_df["target"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        params = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": 42
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # ---------------- Evaluate ----------------
        acc = metrics.accuracy_score(y_val, preds)
        prec = metrics.precision_score(y_val, preds, average="weighted", zero_division=0)
        rec = metrics.recall_score(y_val, preds, average="weighted", zero_division=0)
        f1 = metrics.f1_score(y_val, preds, average="weighted", zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print(f"✅ Metrics: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}")

        mlflow.log_params(params)

        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name="oppe2-production-model",
        )

        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Model saved locally: {model_path}")

        # Upload to GCS
        upload_to_gcs("iitmbs-mlops-21f1000344", model_path, "my_models/model.joblib")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:8100")
    parser = argparse.ArgumentParser(description="Train Logistic Regression Model with Feast Features")
    parser.add_argument("--data_path", type=str, default="data/data.parquet", help="Path to the parquet file")
    args = parser.parse_args()
    train_model_with_feast(args.data_path)