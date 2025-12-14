import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from mlflow.models import infer_signature
import joblib

import shap
import mlflow
import mlflow.sklearn

from fairlearn.metrics import demographic_parity_difference

SEED = 42
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------
# 1) Load dataset
# -------------------------
data_path = "data/heart.parquet"
df = pd.read_parquet(data_path)

TARGET_COL = "target"

# -------------------------
# 3) Prepare features + labels
# -------------------------
X = df.drop(columns=[TARGET_COL, "event_timestamp", "patient_id"])
y = df[TARGET_COL].copy()

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

age_val = df.loc[X_val.index, "age"].reset_index(drop=True)

age_binned = pd.cut(
    age_val,
    bins=[0, 40, 55, 100],
    labels=["young", "middle", "old"]
)
y_val = y_val.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)

# -------------------------
# 4) Load trained model
# -------------------------

params = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "liblinear",
        "random_state": 42
    }
model = joblib.load("artifacts/model.joblib")

# -------------------------
# 5) Validation metrics
# -------------------------
y_val_pred = model.predict(X_val)

f1 = f1_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, zero_division=0)
recall = recall_score(y_val, y_val_pred, zero_division=0)

# -------------------------
# 6) SHAP explainability (beeswarm)
# -------------------------

# Unified SHAP API – works for any model
explainer = shap.Explainer(model.predict_proba, X_val)

# SHAP values for positive class (index=1)
shap_values = explainer(X_val)
shap_pos = shap_values[:, :, 1]   # instances × features

# Prepare shap.Explanation object
shap_exp = shap.Explanation(
    values=shap_pos.values,
    base_values=shap_pos.base_values,
    data=X_val.values,
    feature_names=X_val.columns,
)

shap_path = os.path.join(ARTIFACT_DIR, "shap_summary.png")

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_exp, show=False, max_display=30)
plt.savefig(shap_path, bbox_inches="tight", dpi=300)
plt.close()

# -------------------------
# 7) Fairness audit
# -------------------------
dpd = demographic_parity_difference(
    y_true=y_val,
    y_pred=y_val_pred,
    sensitive_features=age_binned
)

# -------------------------
# 8) MLflow logging
# -------------------------
mlflow.set_tracking_uri("http://127.0.0.1:8100")
mlflow.set_experiment("OPPE2-Explainability-Fairness-Test")

with mlflow.start_run():
    # Log params
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("f1_val", float(f1))
    mlflow.log_metric("precision_val", float(precision))
    mlflow.log_metric("recall_val", float(recall))
    mlflow.log_metric("demographic_parity_difference", float(dpd))

    # Log SHAP plot
    mlflow.log_artifact(shap_path, artifact_path="explainability")

    # Log model
    signature = infer_signature(X_train, model.predict(X_train))
        
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="oppe2-location-dt"
    )

print("\nCompleted Explainability + Fairness workflow (Decision Tree).")
print("SHAP plot saved at:", shap_path)
print(f"F1 Score (v0): {f1:.4f}")
print(f"Demographic Parity Difference: {dpd:.6f}")