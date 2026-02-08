# Wellness Tourism Package – MLOps Pipeline

This project implements an **end-to-end MLOps pipeline** to predict whether a customer is likely to purchase the newly introduced **Wellness Tourism Package** for the company *“Visit with Us”*.  
The solution automates data ingestion, preprocessing, model training, experiment tracking, deployment, and CI/CD using modern MLOps tools.

---

## 📌 Project Objective
To build a scalable and automated machine learning system that:
- Predicts customer purchase likelihood (`ProdTaken`)
- Improves marketing targeting efficiency
- Enables continuous model improvement through CI/CD

---

## 🏗️ Project Architecture
- **Data Storage:** Hugging Face Datasets  
- **Model Training & Tracking:** XGBoost + MLflow  
- **Model Registry:** Hugging Face Model Hub  
- **Deployment:** Streamlit on Hugging Face Spaces  
- **CI/CD:** GitHub Actions (end-to-end automation)

---

## 📂 Repository Structure
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv
│   ├── deployment/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── hosting/
│   │   └── hosting.py
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   └── train.py
│   └── requirements.txt
└── README.md
