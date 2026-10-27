# Project Technical Introduction

**End-to-End Telecom Churn Prediction Pipeline** This project implements a complete machine learning lifecycle for predicting customer churn in telecommunications using the Telco Customer Churn dataset ([https\://www\.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA\_Fn-UseC\_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)). The project focuses on data loading, preprocessing, handling class imbalance with SMOTE, model training ( Random Fores,Logistic Regression, LighGBM and XGBoost), rigorous evaluation, and SHAP explainability.

# Refine the Problem

Customer churn is a major challenge for subscription-based businesses because it can lead to lost revenue, reduced customer loyalty, and higher costs associated with acquiring new customers.  In subscription-based business models, acquiring a new customer can cost 5x to 25x more than retaining an existing one. Unidentified churn silently erodes Customer Lifetime Value (LTV) and inflates Customer Acquisition Costs (CAC).

&nbsp;When customers cancel their subscriptions, businesses may struggle to maintain stable revenue and long-term customer relationships.

This project aims to develop a machine learning model that predicts whether a customer is likely to cancel their subscription or continue using the service. Using our chosen customer data, the model will identify patterns and relationships that may indicate a higher risk of churn.

Predicting churn after a cancellation request is usually too late. This pipeline provides actionable risk scores that enable automated or targeted retention campaigns like customized discount offers, VIP support routing, or feature re-engagement drips to high-risk customers prior to their contract renewal window.

***Primary Objective:*** Maximize high-risk churn detection (Recall) to save recurring revenue while keeping target marketing spend manageable.

***Financial Impact:*** Capturing 76% of churners allows the retention team to preserve substantial LTV that would otherwise be lost to competitors.

The predictions can help customer-retention teams identify at-risk customers, prioritize follow-up activities, and develop suitable retention strategies, such as personalized offers or improved customer support. However, the model provides an estimate of churn risk rather than a guarantee that a customer will cancel their subscription.

***Why Threshold Tuning (0.39) Matters?***

By default, classifiers evaluate positive predictions at a probability threshold of 0.50. In churn prediction, a false negative (missing a churner) is vastly more expensive than a false positive (sending a retention coupon to a loyal customer).

By lowering the decision threshold to 0.39, we shift the decision boundary to prioritize Recall without sacrificing model reliability:

**Data Pipeline & Feature Engineering**&nbsp;

Raw customer telemetry and billing data undergo a reproducible transformation pipeline to prepare features for inference:

***Imputation & Cleaning:*** Missing values in numerical features (e.g., total charges) are imputed using median strategy; categorical missing values are tagged with an Unknown indicator.

***Categorical Encoding***: One-Hot Encoding (OHE) applied to multi-class nominal features (e.g., payment methods, internet service types); Binary encoding applied to boolean flags.

***Feature Scaling:*** Robust Scaling applied to numeric features (tenure, monthly charges) to handle skewed distributions and outliers effectively without leaking statistical properties across splits.

***Imbalance Handling:***

***SMOTE*** (Synthetic Minority Over-sampling Technique): Applied strictly to the training split to synthesize minority class samples and prevent the model from biasing toward non-churners.

**Model Evaluation & Tradeoffs**&nbsp;

Standard accuracy can be misleading on imbalanced churn data—a naive model predicting "no churn" for everyone could achieve high baseline accuracy while completely failing the business.

&nbsp;

# Telecom Customer Churn Predictor

An end-to-end machine learning system built to predict customer churn in subscription services. By combining synthetic oversampling (SMOTE), boosted decision trees (XGBoost), and customized decision threshold tuning, this engine identifies at-risk accounts early enough for proactive retention intervention.

This  Machine Learning pipeline and interactive Streamlit web application designed to predict telecom customer churn using an optimized **SMOTE-XGBoost** model.

---

## 📁 Project Structure

├── app/

│   └── app.py                  \# Interactive Streamlit Web Application

├── data/

│   ├── raw/

│   │   └── original\_dataset.csv \# Raw Kaggle Telco Churn Dataset

│   └── processed/

│       └── prepared\_dataset.csv \# Preprocessed and Cleaned Dataset

├── models/

│   ├──baseline\_pipeline.joblib    \#trained Logistic Regression & Random Forest models

│   ├──candidate\_pipeline.joblib  \#trained LightGBM & XGBoost models

│   ├── final\_model\_pipeline.joblib \# Trained SMOTE-XGBoost Imblearn Pipeline

│   └── model\_metadata.json      \# Evaluation metrics and schema thresholds

├── outputs/

│   ├── figures/

│   ├── Amounts of churn & retain datasets.png

│   ├── clean\_Dashboard.png

│   ├──Comparison of ROC-AUC Curves Across All Models.png

│   ├──correlations of numerical features.png

│   ├──HIgh\_Risk\_Dashboard.png

│   ├──Low\_Risk\_Dashboard.png

│   ├──monthly charge density by churn.png

│   ├──Overall comparisions of models.png

│   └── SHAP Explainability Analysis.png

│   ├── report/

│   ├── model\_evaluation\_report.csv \# Final Performance Metrics report

│   └── experiment\_log.csv       \# Experiment iteration records

├──Customer Churn Prediction.ipynb  \#root code&nbsp;

├── train.py                    \# Unified execution script for training the pipeline

├── requirements.txt            \# Python dependencies configuration

└── README.md                   \# Installation and usage instructions

&nbsp;

---

## ⚙️ Installation & Setup

**Prerequisites&nbsp;**

* &nbsp;&nbsp;Python 3.11 or higher installed


# 📖 Local User Guide: Telecom Customer Churn Prediction Dashboard

Welcome to the **Telecom Customer Churn Prediction Dashboard**! This interactive web application is designed for customer-retention teams, marketing analysts, and managers to proactively evaluate customer churn risks and receive dynamic, strategic retention suggestions.

Since this application runs entirely on your local machine, your data remains secure, private, and offline.

---

## 🚀 Features
* **Real-Time Risk Analysis:** Input customer demographics, subscribed services, and billing configurations to see their immediate calculated risk.
* **SMOTE-XGBoost Architecture:** Uses an advanced machine learning model optimized to catch over **76% of actual churners** (Recall: `0.7647`).
* **Optimized Decision Threshold:** Tuned to a custom decision threshold of `0.39` to catch high-risk accounts before they cancel.
* **Interactive Actionable Insights:** Highlights specific features of the customer's profile and proposes strategic paths (e.g., offering automatic payments or long-term contract migrations).

---

## 🛠️ Local Installation & Running Instructions

Follow these quick steps to set up and launch the dashboard locally on your computer.

### Step 1: Clone or Copy the Project Files
Ensure you have the following files in your project directory:
* `app.py` (The Streamlit web application code)
* `requirements.txt` (List of required software packages)
* `models/final_model_pipeline.joblib` (The pre-trained machine learning model pipeline)

### Step 2: Open your Terminal / Command Prompt
Navigate to your project directory. For example:
```bash
cd path/to/your/project_directory
```

### Step 3: Set Up a Virtual Environment (Recommended)
Using a virtual environment keeps your Python packages isolated and prevents version conflicts.

* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
Install the exact, stable package versions required to run the pipeline model:
```bash
pip install -r requirements.txt
```

### Step 5: Start the Streamlit Application
Run the local webserver by executing:
```bash
streamlit run app.py
```

### Step 6: Access the Dashboard
Once the command executes, Streamlit will automatically open your default web browser to:  
👉 **`http://localhost:8501`**

If it does not open automatically, copy and paste that address into your web browser.

---

## 💡 Tips for Testing Customer Profiles
To explore how the model responds, test these extreme profiles in the left sidebar:

1. **High-Risk Profile (Expected output: 🚨 HIGH CHURN RISK):**
   * **Contract Type:** Month-to-month
   * **Payment Method:** Electronic check
   * **Internet Service:** Fiber optic
   * **Online Security / Tech Support:** No
   * **Tenure:** 1-5 months

2. **Low-Risk Profile (Expected output: ✅ LOW RISK):**
   * **Contract Type:** Two year
   * **Payment Method:** Bank transfer (automatic)
   * **Internet Service:** DSL or No internet service
   * **Online Security / Tech Support:** Yes
   * **Tenure:** 60+ months
```

## 📊 Features & Takeaways

- **SMOTE Resampling:** Tackles training target class imbalance ratio (from 26% to balanced status).  
- **Optimal Thresholding:** Fine-tuned probability decision threshold configured at `0.39` to yield maximum Recall (\~76.47%) and high ROC-AUC (\~84.56%).  
- **Real-time Inference:** Input demographics (tenure, charges, payment methods) to predict high-risk classifications and generate immediate retention actions.
