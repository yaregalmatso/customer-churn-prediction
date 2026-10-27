
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
