import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# CRITICAL: Import imblearn explicitly so Python's pickle find_class 
# can successfully resolve and deserialize the SMOTE pipeline classes.
import imblearn
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Dynamic and robust model path resolution for Streamlit Cloud deployment
def resolve_model_path():
    # List of possible relative path structures on Streamlit Cloud and Local testing
    possible_paths = [
        os.path.join(os.getcwd(), 'models', 'final_model_pipeline.joblib'),
        os.path.join(os.getcwd(), 'My project', 'models', 'final_model_pipeline.joblib'),
        os.path.join(os.getcwd(), 'My project', 'app', 'models', 'final_model_pipeline.joblib'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'final_model_pipeline.joblib') if '__file__' in globals() else '',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'final_model_pipeline.joblib') if '__file__' in globals() else ''
    ]

    for path in possible_paths:
        if path and os.path.exists(path):
            return path

    # Fallback default if not found yet
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return os.path.join(base, 'models', 'final_model_pipeline.joblib')

MODEL_PATH = resolve_model_path()
OPTIMAL_THRESHOLD = 0.39

@st.cache_resource
def load_model_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading the model pipeline file: {str(e)}. This can happen if there is a library version mismatch between Google Colab and Streamlit Cloud.")
            return None
    else:
        st.error(f"Model pipeline not found. We searched several paths, current resolved path: `{MODEL_PATH}`. Please verify your `final_model_pipeline.joblib` file is committed to your GitHub repository inside a `models/` directory.")
        return None

loaded_pipeline = load_model_pipeline()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
This app uses a custom-threshold **SMOTE-XGBoost Pipeline** to predict customer churn risk.
Adjust customer demographics and services in the sidebar to run interactive inference.
""")

# Sidebar input fields
st.sidebar.header("👤 Customer Profile")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen (65+)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

st.sidebar.header("📞 Services & Subscriptions")
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.sidebar.header("💳 Billing & Payment")
Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * MonthlyCharges))

# Submit button
if st.sidebar.button("Predict Churn Risk", type="primary"):
    if loaded_pipeline is not None:
        # Construct raw input DataFrame
        input_dict = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        input_df = pd.DataFrame([input_dict])

        # Replicate dynamic Feature Engineering pipeline
        input_df["ValueRatio"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
        input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=[0,12,24,36,72], labels=["New","Short","Medium","Long"])
        # Handle edge-case categorical bins if tenure was 0
        input_df["TenureGroup"] = input_df["TenureGroup"].astype(object).fillna("New")
        input_df['Monthly_to_Total_Ratio'] = input_df['MonthlyCharges'] / (input_df['TotalCharges'] + 1)
        input_df['Tenure_Charges_Interaction'] = input_df['tenure'] * input_df['MonthlyCharges']
        input_df['Total_Charges_to_Monthly_Ratio'] = input_df['TotalCharges'] / (input_df['MonthlyCharges'] + 1)

        # Run prediction
        with st.spinner("Calculating Risk score..."):
            churn_probability = loaded_pipeline.predict_proba(input_df)[0, 1]
            is_churn = int(churn_probability >= OPTIMAL_THRESHOLD)

        # Display Results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric(label="Churn Probability", value=f"{churn_probability * 100:.2f}%")

        with col2:
            if is_churn == 1:
                st.error("🚨 HIGH RISK: Customer is likely to churn!")
            else:
                st.success("✅ LOW RISK: Customer is likely to stay retained.")

        # Highlighting Risk Profiles
        st.subheader("ℹ️ Strategic Key Takeaways")
        if Contract == "Month-to-month" and PaymentMethod == "Electronic check":
            st.warning("⚠️ This profile exhibits high-risk properties (Month-to-month contract & Electronic check payments). Offer incentives to migrate to a long-term contract or register for automatic payment methods.")
        else:
            st.info("💡 Customer features indicate stable subscription behavior. Keep engaging them with relevant upgrade offers.")
    else:
        st.error("Unable to run prediction. The pipeline model was not loaded.")
else:
    st.info("👈 Configure parameters in the sidebar and click **Predict Churn Risk** to run inference.")
