%%writefile train.py
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
from xgboost import XGBClassifier

# Dynamically determine the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure relative paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'original_dataset.csv')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Using relative base directory: {BASE_DIR}")

# 1. Load Data
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}. Please place original_dataset.csv in data/raw/")

data = pd.read_csv(RAW_DATA_PATH)
df = data.drop("customerID", axis=1, errors="ignore")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 2. Feature Engineering
df["ValueRatio"] = df["TotalCharges"] / (df["tenure"] + 1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[0, 12, 24, 36, 72], labels=["New", "Short", "Medium", "Long"])
df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
df['Tenure_Charges_Interaction'] = df['tenure'] * df['MonthlyCharges']
df['Total_Charges_to_Monthly_Ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

# Save processed data relatives
df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'prepared_dataset.csv'), index=False)

# 3. Preprocessing Setup
X = df.drop('Churn', axis=1)
y = df['Churn']

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

# Calculate Class Weights
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_pos_weight = num_neg / num_pos

# 4. Define and Train SMOTE-XGBoost ImbPipeline
tuned_model = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.6,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

final_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.6)),
    ('model', tuned_model)
])

final_pipeline.fit(X_train, y_train)

# Save Pipeline
final_model_path = os.path.join(MODELS_DIR, 'final_model_pipeline.joblib')
joblib.dump(final_pipeline, final_model_path)
print(f"Saved Final Pipeline to: {final_model_path}")
