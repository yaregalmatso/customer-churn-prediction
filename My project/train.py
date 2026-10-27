import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from lgbm import LGBMClassifier
from xgboost import XGBClassifier

# Dynamically calculate absolute paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'original_dataset.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading dataset...")
# Fallback to absolute Google Drive mounted path if missing locally
if not os.path.exists(RAW_DATA_PATH):
    RAW_DATA_PATH = '/content/drive/My Drive/My_AI_Project/data/raw/original_dataset.csv'

data = pd.read_csv(RAW_DATA_PATH)
df = data.drop("customerID", axis=1, errors='ignore')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Feature Engineering
df["ValueRatio"] = df["TotalCharges"] / (df["tenure"], + 1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[0,12,24,36,72], labels=["New","Short","Medium","Long"])
df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
df['Tenure_Charges_Interaction'] = df['tenure'] * df['MonthlyCharges']
df['Total_Charges_to_Monthly_Ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

df.to_csv(os.path.join(PROCESSED_DIR, 'prepared_dataset.csv'), index=False)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Preprocessing Pipeline Setup
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute weight class ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("Training models...")
# 1. Baseline Pipeline
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
baseline_pipeline.fit(X_train, y_train)
joblib.dump(baseline_pipeline, os.path.join(MODELS_DIR, "baseline_pipeline.joblib"))

# 2. Candidate Pipeline
candidate_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1))
])
candidate_pipeline.fit(X_train, y_train)
joblib.dump(candidate_pipeline, os.path.join(MODELS_DIR, "candidate_pipeline.joblib"))

# 3. Final Model Pipeline (SMOTE + XGBoost)
tuned_model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.01, subsample=0.6, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
final_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.6)),
    ('model', tuned_model)
])
final_pipeline.fit(X_train, y_train)
joblib.dump(final_pipeline, os.path.join(MODELS_DIR, "final_model_pipeline.joblib"))

print(f"Training completed successfully. All models written relative to: {MODELS_DIR}")
