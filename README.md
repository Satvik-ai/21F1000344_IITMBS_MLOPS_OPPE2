# Heart Disease ML Model — End-to-End MLOps with CI/CD, Monitoring & Governance

## 📌 Overview
This project demonstrates a **production-grade MLOps pipeline** for a Heart Disease prediction model.  
It covers the full lifecycle — from **data versioning and training** to **deployment on Kubernetes**, **monitoring**, **fairness analysis**, and **explainability**.

The pipeline is designed to simulate a real-world ML system with automated testing, deployment, stress testing, and model governance.

---

## 🎯 Key Capabilities
- End-to-end CI/CD pipeline with GitHub Actions  
- Automated model training and experiment tracking  
- Containerized deployment on Kubernetes  
- Real-time inference via FastAPI  
- Stress testing and autoscaling validation  
- Data drift detection and monitoring  
- Fairness analysis across age groups  
- Model explainability using SHAP  

---

## 🧱 Project Architecture

```
├── raw_data/
│ └── heart.csv
│
├── data/
│ └── heart.parquet
│
├── artifacts/
│ ├── model.pkl
│ └── shap_summary.png
│
├── app/
│ ├── main.py
│ ├── Dockerfile
│ ├── requirements.txt
│ └── k8s/
│ ├── deployment.yaml
│ ├── service.yaml
│ └── hpa.yaml
│
├── src/
│ ├── train.py
│ ├── prep_data_feast.py
│ ├── generate_test_data.py
│ └── explainability_fairness.py
│
├── tests/
├── .github/workflows/
├── feature_repo/
├── create_gke_cluster.sh
└── README.md
```


---

## 🛠️ Tech Stack

- **ML & Data:** Scikit-learn, DVC, MLflow  
- **Serving:** FastAPI, Docker  
- **Infrastructure:** Google Kubernetes Engine, Artifact Registry  
- **CI/CD:** GitHub Actions  
- **Monitoring:** Google Cloud Logging, Trace, Monitoring  
- **Governance:** Evidently, Fairlearn, SHAP  
- **Feature Store:** Feast  

---

## 📂 Project Components

### 📁 `raw_data`
Stores the original dataset (`heart.csv`) before preprocessing.

### 📁 `data`
Contains processed dataset (`heart.parquet`) used for training.

### 📁 `artifacts`
Stores:
- Trained model  
- SHAP explainability plots  

### ⚙️ Training (`src/train.py`)
- Loads processed data  
- Trains **Logistic Regression** model  
- Logs parameters, metrics, and model to MLflow  

### 🧹 Data Preparation (`src/prep_data_feast.py`)
- Cleans raw dataset  
- Encodes categorical features  
- Adds `patient_id` and timestamp  
- Saves as parquet  

### 🧪 Testing (`tests/`)
- Data validation tests  
- Model evaluation tests  

### 🔍 Explainability & Fairness (`src/explainability_fairness.py`)
- Generates SHAP beeswarm plot  
- Computes fairness metric (demographic parity difference across age)  

---

## 🚀 Deployment Pipeline

### 🔁 Continuous Integration
Workflows:
- `ci-dev.yml`
- `ci-main.yml`

**Steps**
1. Pull data & model from DVC  
2. Run tests with pytest  
3. Publish reports using CML  

---

### 🚀 Continuous Deployment (`cd.yml`)
Triggered after successful CI on main branch:

1. Build Docker image  
2. Push image to Artifact Registry  
3. Deploy to GKE  

---

### 📦 Batch Inference (`batch_inference.yml`)
- Generates random test data  
- Sends requests to deployed API  
- Collects predictions  

---

### ⚡ Stress Testing (`stress_test.yml`)
- Simulates high-load traffic using **wrk**  
- Demonstrates autoscaling from 1 → 3 pods  
- Tests bottlenecks under restricted scaling  

---

## 📊 Monitoring & Observability

- Request logs and traces via :contentReference[oaicite:0]{index=0}  
- Drift detection using :contentReference[oaicite:1]{index=1}  
- Performance tracking via :contentReference[oaicite:2]{index=2}  

---

## ⚖️ Responsible AI

### Fairness
Evaluates model bias across **age groups** using Fairlearn.

### Explainability
SHAP summary plot shows:

- Feature importance  
- Direction of impact on predictions

---

## 🎥 Video Presentation  
[▶️ Click Here](https://drive.google.com/file/d/1DWVUCL1RrnMdETyQMVAhNdhiMSggI2NP/view?usp=drive_link)

