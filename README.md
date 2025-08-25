# End-to-End MLOps project 

# 🚀 MLOps Project: Used Car Price Prediction

This project demonstrates a complete **MLOps workflow** — from model training and versioning to containerization and deployment.  
It uses **scikit-learn**, **MLflow** for experiment tracking, and a **Flask REST API** served in Docker. This model uses RandomForestRegressor from scikit-learn to estimate price of used car based on data provided from cardekho. Dataset is downloaded from kaggle
https://www.kaggle.com/datasets/pushpakhinglaspure/used-car-price-prediction

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



---

## ⚙️ Features

- ✅ Data preprocessing & feature engineering  
- ✅ Model training (Linear Regression / Random Forest / etc.)  
- ✅ Model & artifacts tracked in **MLflow**  
- ✅ REST API endpoint using **Flask** (`/predict`, `/health`)  
- ✅ Containerized with **Docker**  
- ✅ CI/CD pipeline with **GitHub Actions**  
- ✅ Future-ready for deployment to **Kubernetes / Cloud**  

---

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

All experiments & models are logged using MLflow.
<img width="1860" height="612" alt="Screenshot 2025-07-17 192620" src="https://github.com/user-attachments/assets/977875c8-9e76-4407-94b7-bebf3b1db8a3" />
<img width="985" height="689" alt="Screenshot 2025-07-17 192801" src="https://github.com/user-attachments/assets/6dfbf7ca-1a67-469d-b1a1-91257f9d1cde" />
<img width="1793" height="738" alt="Screenshot 2025-07-17 192821" src="https://github.com/user-attachments/assets/05eae2b9-baa3-4936-8477-449b4b7f1bc4" />

### Sample Prediction of model
<img width="915" height="532" alt="Screenshot 2025-08-25 234025" src="https://github.com/user-attachments/assets/cbca2359-41fd-4cb3-8514-d3eda47c0897" />







