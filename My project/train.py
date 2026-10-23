# Consolidating all backend steps into a single unified training script
import os
import json
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROJECT_PATH = "My project"
RAW_DATA_PATH = f"{PROJECT_PATH}/data/raw/original_dataset.csv"
PROCESSED_DATA_DIR = f"{PROJECT_PATH}/data/processed"
MODELS_DIR = f"{PROJECT_PATH}/models"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load Data
print("Loading dataset...")
data = pd.read_csv(RAW_DATA_PATH)
df = data.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 2. Feature Engineering
print("Performing Feature Engineering...")
df["ValueRatio"] = df["TotalCharges"] / (df["tenure"] + 1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[0,12,24,36,72], labels=["New","Short","Medium","Long"])
df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
df['Tenure_Charges_Interaction'] = df['tenure'] * df['MonthlyCharges']
df['Total_Charges_to_Monthly_Ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

# Save clean data
df.to_csv(f"{PROCESSED_DATA_DIR}/prepared_dataset.csv", index=False)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# 3. Model Pipeline Definition (SMOTE + XGBoost)
print("Fitting SMOTE-XGBoost pipeline...")
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_pos_weight = num_neg / num_pos

tuned_model = XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.01, subsample=0.6,
    colsample_bytree=0.8, random_state=42, eval_metric='logloss'
)

final_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.6)),
    ('model', tuned_model)
])

final_pipeline.fit(X_train, y_train)

# 4. Save Artifacts
joblib.dump(final_pipeline, f"{MODELS_DIR}/final_model_pipeline.joblib")

# Save metadata
metadata = {
    "project_name": "Customer Churn Prediction",
    "optimal_decision_threshold": 0.39,
    "features": {
        "numerical_columns": list(numerical_cols),
        "categorical_columns": list(categorical_cols)
    }
}
with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Pipeline training complete! Saved model pipeline and metadata successfully.")
