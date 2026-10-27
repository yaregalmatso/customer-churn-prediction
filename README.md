
# 📊 Telecom Customer Churn Prediction & Actionable Retention Dashboard

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/classifier-XGBoost-1E88E5.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade machine learning system designed to predict customer churn, diagnose risk factor drivers using **SHAP**, and provide dynamic, strategic retention plays. This project addresses the classic subscription service churn problem using robust statistical balancing, advanced boosting algorithms, and professional software implementation workflows.

### 🔗 Key Highlights
* **Recall-Optimized SMOTE-XGBoost Architecture:** Reaches an outstanding **76.47% Recall** to successfully identify high-risk customer dropouts before they cancel.
* **Strategic Decision Thresholding:** Implements a custom decision boundary of `0.39` to balance precision-recall trade-offs based on financial retention costs.
* **Feature-Engineered Pipeline:** Includes high-fidelity variables such as value-to-tenure interaction and charge-interaction multipliers.
* **SHAP Explainability Engine:** Incorporates Game Theory-driven explainability to identify that month-to-month contracts and electronic check payments are the top churn drivers.
* **Interactive Dashboard:** Equipped with a Streamlit interface containing dynamic input forms and situational billing insights.

---

## 🪐 Project Architecture & Workflow

```
 ├── data/
 │    ├── raw/              <- Original Telco dataset
 │    └── processed/        <- Cleaned & engineered prepared_dataset.csv
 ├── models/
 │    ├── final_model_pipeline.joblib   <- Full Preprocessing + SMOTE + XGBoost pipeline
 │    └── model_metadata.json           <- Evaluation metrics and custom thresholds
 ├── app.py                 <- Interactive Streamlit dashboard application
 ├── train.py               <- Fully automated data pipeline and model retraining script
 └── requirements.txt       <- Python dependencies mapped for stable deployment
```

### 🔄 The End-to-End Pipeline
1. **Data Cleaning & Engineering:** Coerces structural types, imputes missing records cleanly with the median, and engineers interactive metrics (e.g. `ValueRatio`, `Tenure_Charges_Interaction`).
2. **Preprocessing Isolation:** Builds separate numerical scaling (`StandardScaler`) and nominal encoding (`OneHotEncoder`) sequences to prevent downstream data leakage.
3. **Addressing Class Imbalance:** Integrates Synthetic Minority Over-sampling Technique (**SMOTE**) inside a unified pipeline to handle the target label imbalance (~26% Churn).
4. **Algorithmic Search & Calibration:** Tests Logistic Regression, Random Forest, LightGBM, and XGBoost. Fine-tunes custom parameters and models to maximize true positive classification.
5. **Model Serialization:** Exports production pipelines and telemetry metadata into secure serializations (`.joblib` and `.json`).

---

## 📈 Performance Summary

Through careful threshold calibration on the validation sets, our custom **SMOTE-XGBoost** model substantially outperforms traditional algorithms in catch-rate metrics:

| Model Group | Model Algorithm | Accuracy | Precision | Recall (Catch Rate) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Logistic Regression | 80.91% | 67.21% | 54.81% | 0.6038 | 0.8476 |
| **Candidate** | LightGBM | 78.92% | 62.15% | 52.67% | 0.5702 | 0.8336 |
| **Production** | **SMOTE-XGBoost (Threshold = 0.39)** | **76.79%** | **54.48%** | **76.47%** | **0.6363** | **0.8456** |

> **Business Impact:** While baseline approaches only capture about half of churners, our production SMOTE-XGBoost model identifies **over 76% of at-risk customers**, allowing the marketing and retention teams to save valuable contracts before they drop off.

---

## 🛠️ Installation & Local Setup

Run the machine learning dashboard locally in just 5 steps:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/telco-churn-predictor.git
cd telco-churn-predictor
```

### 2. Set up virtual environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Optional: Run training workflow
If you wish to retrain the pipeline from scratch and reserialise models:
```bash
python train.py
```

### 5. Start the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to **`http://localhost:8501`**!

---

## 🔮 Explainable AI (SHAP Summary)

To ensure our model's predictions are interpretable and actionable for customer support teams, we use **SHAP (SHapley Additive exPlanations)** values to display feature contributions:

* **Month-to-Month Contracts:** Unquestionably the strongest positive driver of churn risk.
* **Online Security / Tech Support (No):** Customers without these protective add-ons exhibit significantly higher churn risk.
* **Electronic Check Payment Method:** Strongly correlates with highly active, short-term churn-prone behaviors.

--- 

## 🤝 Contacts & Authors
This project was designed and developed by a dedicated AI engineer:

* **Yaregal Matso**
```
