# 🏥 Breast Cancer Classification - Complete MLOps Pipeline

![CI Pipeline](https://github.com/Krishmalik786/rajuDI-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Production-grade MLOps pipeline** for breast cancer classification with experiment tracking, automated deployment, monitoring, and CI/CD integration.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [CI/CD Pipeline](#️-cicd-pipeline)
- [Results & Screenshots](#-results--screenshots)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

### Problem Statement

Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate diagnosis is crucial for effective treatment. This project builds an end-to-end machine learning system to classify breast tumors as **Malignant (Cancerous)** or **Benign (Non-cancerous)** based on 30 numerical features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

### Dataset

- **Source:** Wisconsin Breast Cancer Dataset (sklearn)
- **Samples:** 569 instances
- **Features:** 30 numerical features (mean, standard error, and worst values of 10 characteristics)
- **Target:** Binary classification (0 = Malignant, 1 = Benign)
- **Distribution:** 357 Benign (62.7%), 212 Malignant (37.3%)

### Solution

A comprehensive MLOps pipeline that includes:
- Automated data preprocessing and versioning
- Multi-model training with experiment tracking
- RESTful API and interactive UI for predictions
- Containerized deployment
- Continuous integration and delivery
- Production monitoring for data drift

---

## ✨ Key Features

### 🔄 Complete ML Pipeline
- ✅ Automated data download and validation
- ✅ Feature engineering and scaling
- ✅ Train-test split with stratification
- ✅ Multi-algorithm training (Logistic Regression, Random Forest, XGBoost)
- ✅ Hyperparameter tracking
- ✅ Best model selection based on metrics

### 🛠️ MLOps Best Practices
- ✅ **Data Versioning** - DVC for reproducible datasets
- ✅ **Experiment Tracking** - MLflow for all training runs
- ✅ **Pipeline Orchestration** - Automated workflow execution
- ✅ **Model Registry** - Version-controlled model artifacts
- ✅ **Deployment** - Production-ready FastAPI + Streamlit
- ✅ **Containerization** - Docker for environment consistency
- ✅ **CI/CD** - GitHub Actions for automated testing
- ✅ **Monitoring** - Data drift detection with Evidently

### 🚀 Production-Ready
- REST API with auto-generated documentation
- Interactive web UI for non-technical users
- Health checks and error handling
- Logging and monitoring
- Docker containerization
- Automated testing pipeline

---

## 🏗️ Architecture
┌──────────────────────────────────────────────────────────────┐
│ DATA PIPELINE │
│ │
│ Raw Data (DVC) → Preprocessing → Feature Engineering │
│ ↓ ↓ ↓ │
│ Train/Test Split → Scaling → Validation │
└──────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────┐
│ MODEL TRAINING │
│ │
│ Logistic Regression ┐ │
│ Random Forest ├→ MLflow Tracking → Best Model │
│ XGBoost ┘ │
└──────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────┐
│ DEPLOYMENT │
│ │
│ Best Model → FastAPI (REST) ──→ Docker Container │
│ ↘ Streamlit (UI) ──→ Docker Container │
└──────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────┐
│ MONITORING & CI/CD │
│ │
│ Evidently (Drift) → Reports │
│ GitHub Actions → Tests → Build → Deploy │
└──────────────────────────────────────────────────────────────┘


---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Programming** | Python 3.11 | Core development language |
| **ML Frameworks** | scikit-learn, XGBoost | Model training |
| **Data Processing** | pandas, numpy | Data manipulation |
| **Visualization** | matplotlib, seaborn | EDA and analysis |
| **Experiment Tracking** | MLflow | Track experiments, parameters, metrics |
| **Data Versioning** | DVC | Version control for datasets |
| **API Framework** | FastAPI | REST API development |
| **UI Framework** | Streamlit | Interactive web interface |
| **Containerization** | Docker, docker-compose | Environment isolation |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **Monitoring** | Evidently | Data drift detection |
| **Code Quality** | flake8 | Linting and style checks |

---

## 🚀 Quick Start

### Prerequisites

Python 3.11 or higher

Docker Desktop (optional, for containerization)

Git

### Installation

git clone https://github.com/Krishmalik786/rajuDI-project.git
cd rajuDI-project

2. Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

4. Run the complete pipeline
python workflows/pipeline.py

### output

============================================================
🚀 ML PIPELINE ORCHESTRATION
Pipeline Steps:

Download Data

Preprocess Data

Train Models with MLflow
============================================================

✅ Download Data completed in 0.45s
✅ Preprocess Data completed in 1.23s
✅ Train Models completed in 45.67s

============================================================
✅ PIPELINE COMPLETED SUCCESSFULLY!
⏱️ Total execution time: 47.35s
## 📖 Detailed Usage

### 1. Data Pipeline
Download dataset
python src/download_data.py

Preprocess data
python src/preprocess.py

Verify outputs
ls data/processed/

Output: X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl

### 2. Model Training
Train all models with MLflow tracking
python src/train.py

View experiment results
mlflow ui

Open: http://localhost:5000

**Training Output:**
🔵 Training Logistic Regression...
✅ Accuracy: 0.9825 | Precision: 0.9873 | Recall: 0.9873 | F1: 0.9873

🟢 Training Random Forest...
✅ Accuracy: 0.9737 | Precision: 0.9873 | Recall: 0.9746 | F1: 0.9809

🟡 Training XGBoost...
✅ Accuracy: 0.9825 | Precision: 1.0000 | Recall: 0.9746 | F1: 0.9871

🏆 BEST MODEL: Logistic Regression

### 3. Deployment

#### Start API Server
Run FastAPI server
uvicorn deployment.app:app --reload

Access:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health


#### Start Streamlit UI

Run Streamlit interface
streamlit run deployment/streamlit_app.py

Open: http://localhost:8501


### 4. Docker Deployment

Build image
docker build -f docker/Dockerfile -t mlops-api:latest .

Run container
docker run -d -p 8000:8000 --name mlops-api mlops-api:latest

Or use docker-compose (runs API + Streamlit)
docker-compose up -d

Stop containers
docker-compose down

### 5. Monitoring
Generate monitoring report
python src/monitor.py

## 📁 Project Structure
rajuDI-project/
│
├── .github/ # GitHub Actions workflows
│ └── workflows/
│ └── ci.yml # CI/CD pipeline configuration
│
├── data/ # Data directory (gitignored except .dvc)
│ ├── raw/ # Original datasets
│ │ ├── breast_cancer.csv
│ │ └── breast_cancer.csv.dvc
│ └── processed/ # Preprocessed data
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ ├── y_test.csv
│ └── scaler.pkl
│
├── deployment/ # Deployment scripts
│ ├── app.py # FastAPI REST API
│ └── streamlit_app.py # Streamlit web interface
│
├── docker/ # Docker configurations
│ ├── Dockerfile # API container
│ └── Dockerfile.streamlit # Streamlit container
│
├── models/ # Trained models
│ ├── best_model.pkl
│ └── model_info.txt
│
├── notebooks/ # Jupyter notebooks
│ └── eda.ipynb # Exploratory Data Analysis
│
├── reports/ # Monitoring reports
│ └── data_monitoring_report.html
│
├── src/ # Source code
│ ├── download_data.py # Data acquisition
│ ├── preprocess.py # Data preprocessing
│ ├── train.py # Model training with MLflow
│ ├── predict.py # Prediction utilities
│ └── monitor.py # Monitoring report generation
│
├── workflows/ # Pipeline orchestration
│ └── pipeline.py # End-to-end workflow
│
├── .dockerignore # Docker ignore patterns
├── .gitignore # Git ignore patterns
├── docker-compose.yml # Multi-container orchestration
├── dvc.yaml # DVC pipeline definition
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)


---

## 📊 Model Performance

### Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **98.25%** | 98.73% | 98.73% | 98.73% | 99.81% |
| **Random Forest** | 97.37% | 98.73% | 97.46% | 98.09% | 99.79% |
| **XGBoost** | **98.25%** | **100.0%** | 97.46% | 98.71% | **99.87%** |

### Model Selection

- **Selected Model:** Logistic Regression (98.25% accuracy)
- **Rationale:** 
  - Tied accuracy with XGBoost
  - Faster inference time
  - More interpretable for medical applications
  - Lower computational requirements

### Test Set Performance

- **Test Samples:** 114 (20% of dataset)
- **True Positives:** 77
- **True Negatives:** 35
- **False Positives:** 1
- **False Negatives:** 1
- **Confidence:** 95%+ on most predictions

---

## 🔌 API Reference

### Base URL
http://localhost:8000

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | **98.25%** | 98.73% | 98.73% | 98.73% | 99.81% |
| Random Forest | 97.37% | 98.73% | 97.46% | 98.09% | 99.79% |
| XGBoost | 98.25% | 100.0% | 97.46% | 98.71% | 99.87% |

**Selected:** Logistic Regression (best balance of accuracy and interpretability)

---

## 🔌 API

### Health Check
GET http://localhost:8000/health

### predict
POST http://localhost:8000/predict
Content-Type: application/json

{
"mean_radius": 17.99,
"mean_texture": 10.38,
"mean_perimeter": 122.8,
...
}

Response:
{
"prediction_label": "Malignant",
"confidence": 0.9523
}

**Interactive Docs:** http://localhost:8000/docs

---

## 🏗️ Architecture

Data (DVC) → Preprocessing → Training (MLflow) → Best Model
↓
FastAPI + Streamlit (Docker)
↓
Monitoring (Evidently)
↓
CI/CD (GitHub Actions)


---

## ⚙️ CI/CD

GitHub Actions runs on every push:
- ✅ Code quality checks (flake8)
- ✅ Pipeline validation
- ✅ Docker build test
- ✅ Security scan

**Status:** [View Actions](https://github.com/Krishmalik786/rajuDI-project/actions)

---

## 📈 Monitoring

Automated drift detection with Evidently:
- Tracks feature distribution changes
- Detects data quality issues
- Alerts on target drift



---

## 🎯 MLOps Features

- [x] Data versioning (DVC)
- [x] Experiment tracking (MLflow)
- [x] Pipeline automation
- [x] REST API (FastAPI)
- [x] Web UI (Streamlit)
- [x] Containerization (Docker)
- [x] CI/CD (GitHub Actions)
- [x] Monitoring (Evidently)

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning (Optuna)
- [ ] Model explainability (SHAP)
- [ ] Cloud deployment (AWS/GCP)
- [ ] API authentication
- [ ] Real-time monitoring dashboard

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create feature branch
3. Submit pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 👥 Author

**Divyansh Malik**  
GitHub: [@Krishmalik786](https://github.com/Krishmalik786)

---

## 🙏 Acknowledgments

- Wisconsin Breast Cancer Dataset (UCI)
- MLOps tools: MLflow, DVC, FastAPI, Docker
- Open-source ML community

---

**⭐ If you found this helpful, please star the repo!**

*Last Updated: November 27, 2025*
