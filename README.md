# End-to-End MLOps project 

# 🚀 MLOps Project: Used Car Price Prediction

A MLOps pipeline showcasing end-to-end machine learning workflow from data preprocessing to deployment. This project predicts used car prices using Random Forest Regression with comprehensive ML lifecycle management.

Dataset is downloaded from kaggle
https://www.kaggle.com/datasets/pushpakhinglaspure/used-car-price-prediction

🎯 Project Overview
This project demonstrates enterprise-level MLOps practices including:

1. Automated ML Pipeline with experiment tracking (MLflow)
2. Containerized Microservices with Docker
3. CI/CD Pipeline with GitHub Actions
4. RESTful API for model serving
5. Data versioning and model artifacts management


📊 Dataset & Problem Statement

1. Dataset: CarDekho Used Car Price Prediction (Kaggle)
2. Objective: Predict selling price based on car features
3. Model: Random Forest Regressor with hyperparameter tuning
4. Features: Car Name, Year, Present Price, Kilometers Driven, Fuel Type, Seller Type, Transmission


🏗️ Architecture
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw Data  │───▶│ Preprocessing │───▶│   Training  │
│  (CSV/DB)   │     │   Pipeline   │     │   Pipeline  │
└─────────────┘     └──────────────┘     └─────────────┘
                                              │
┌─────────────┐    ┌──────────────┐     ┌─────▼─────┐
│   Client    │◀───│  REST API    │◀───│   Model   │
│ Application │    │  (Flask)     │     │ Artifacts │
└─────────────┘    └──────────────┘     └───────────┘
                           │
                    ┌──────▼──────┐
                    │   Docker    │
                    │  Container  │
                    └─────────────┘


⚙️ Key Features
🔬 ML Pipeline

Data Preprocessing: MinMax scaling, One-hot encoding
Model Selection: Random Forest with optimized hyperparameters
Model Validation: Train/Validation/Test split with cross-validation
Feature Engineering: Automated categorical and numerical feature handling

📈 Experiment Tracking

MLflow Integration: Model versioning and experiment tracking
Metrics Logging: Model performance and hyperparameter tracking
Model Registry: Centralized model artifact management

🐳 Containerization

Docker: Lightweight Python 3.9-slim container
Multi-stage Build: Optimized image size
Health Checks: Service monitoring endpoints

🔄 CI/CD Pipeline

Automated Testing: pytest with coverage reporting
Docker Build: Automated image building and pushing
Quality Gates: Code quality checks and testing
Deployment: Ready for cloud deployment
---

## 📂 Project Structure

.
├── src/
│   ├── data_loader.py          # Loads dataset
│   ├── preprocessing_data.py   # Data cleaning & feature engineering
│   ├── prediction.py           # Script for making predictions
│   ├── encoder.joblib          # Trained encoder
│   ├── scaler.joblib           # Trained scaler
│   ├── model.joblib            # Trained ML model
│   └── __init__.py             
│
├── score.py                    # Flask API
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization
└── README.md                   # Documentation


## 🔧 Setup & Run Locally

### 1️⃣ Clone repo
bash

git clone https://github.com/Harshith121212/end-toend-MLOps-project-RandomForestRegression-.git
cd end-toend-MLOps-project-RandomForestRegression-

### Install Dependencies
pip install -r requirements.txt

### Run Flask App
python score.py

### Test health endpoint
curl http://127.0.0.1:5000/health

### Test Prediction
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{parameters for prediction}'


### 📊 MLflow Tracking

The project uses MLflow for comprehensive experiment management:

Experiment Tracking: All model runs with parameters and metrics
Model Registry: Versioned model artifacts
Reproducibility: Complete experiment lineage
Comparison: Easy model performance comparison

All experiments & models are logged using MLflow.
<img width="1860" height="612" alt="Screenshot 2025-07-17 192620" src="https://github.com/user-attachments/assets/977875c8-9e76-4407-94b7-bebf3b1db8a3" />
<img width="985" height="689" alt="Screenshot 2025-07-17 192801" src="https://github.com/user-attachments/assets/6dfbf7ca-1a67-469d-b1a1-91257f9d1cde" />
<img width="1793" height="738" alt="Screenshot 2025-07-17 192821" src="https://github.com/user-attachments/assets/05eae2b9-baa3-4936-8477-449b4b7f1bc4" />

### Sample Prediction of model
<img width="915" height="532" alt="Screenshot 2025-08-25 234025" src="https://github.com/user-attachments/assets/cbca2359-41fd-4cb3-8514-d3eda47c0897" />


📊 Model Performance

1. Algorithm: Random Forest Regressor
2. Training Score: 95.8%
3. Validation Score: 87.2%
4. Hyperparameters: 400 estimators, max_depth=25, min_samples_split=10


🚧 Roadmap & Future Enhancements

 Complete API: Implement /predict endpoint with input validation
 Model Monitoring: Add data drift detection and model performance monitoring using evidently AI
 Advanced ML: Hyperparameter tuning with Optuna
 Cloud Deployment: AWS/Azure deployment with Kubernetes
 Security: API authentication and rate limiting







