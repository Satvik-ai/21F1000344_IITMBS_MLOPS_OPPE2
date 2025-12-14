import pandas as pd
import joblib
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from feast import FeatureStore

def test_artifact_exists():
    """Tests if the essential model artifact from training exists."""
    assert os.path.exists("artifacts/model.joblib"), "Model artifact 'model.joblib' not found."

def test_model_performance():
    """
    Tests if the trained model's F1-score on the test set is above a reasonable threshold.
    """
    # 1. Load the trained model
    model = joblib.load("artifacts/model.joblib")

    # 2. Prepare data and make predictions
    data = pd.read_parquet("data/heart.parquet")
    X = data.drop(columns=["target", "patient_id","event_timestamp"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    
    print(f"Model F1-Score on Feast-generated test set: {f1:.4f}")
    
    # Set a reasonable minimum performance threshold
    assert f1 > 0.1, f"F1 Score ({f1}) is below the 0.1 threshold."