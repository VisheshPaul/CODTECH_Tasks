# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import joblib

# ===============================
# LOAD DATA (EXTRACT)
# ===============================
# Replace with your dataset path
df = pd.read_csv("data.csv")

print("Initial Data:")
print(df.head())

# ===============================
# DEFINE TARGET & FEATURES
# ===============================
TARGET_COLUMN = "result"   # change if required

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ===============================
# IDENTIFY COLUMN TYPES
# ===============================
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumerical Columns:", list(numerical_features))
print("Categorical Columns:", list(categorical_features))

# ===============================
# NUMERICAL PIPELINE
# ===============================
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# ===============================
# CATEGORICAL PIPELINE
# ===============================
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ===============================
# COMBINE PIPELINES
# ===============================
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# FIT & TRANSFORM (TRANSFORMATION)
# ===============================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed.")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

# ===============================
# SAVE PIPELINE (LOADING)
# ===============================
joblib.dump(preprocessor, "data_preprocessing_pipeline.pkl")
print("\nPipeline saved as 'data_preprocessing_pipeline.pkl'")

# ===============================
# OPTIONAL: SAVE PROCESSED DATA
# ===============================
pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed)\
    .to_csv("X_train_processed.csv", index=False)

pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed)\
    .to_csv("X_test_processed.csv", index=False)

print("Processed datasets saved successfully.")
