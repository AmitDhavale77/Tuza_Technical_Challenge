import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


pd.set_option('display.max_rows', None)  # No limit on rows
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Auto-detect width
pd.set_option('display.max_colwidth', None)  # No limit on column width


file_path = 'updated_transaction_data_withlabels.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

print(data.head())

print(data.columns)

label_column = 'Current pay'  
X = data.drop(columns=[label_column, 'Transaction_per_Unit_Turnover_RobustScaled'])  # Features
y = data[label_column]  # Target labels (already encoded)


from sklearn.preprocessing import LabelEncoder

# Encode the target variable 'y'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label Encoding Mapping:")
for class_label, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{class_label} -> {encoded_value}")


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded , test_size=0.2, random_state=42, stratify=y)

# ----------------- XGBoost -----------------
print("Training LGBMClassifier Classifier...")
lgbm = LGBMClassifier(random_state=42)

# Hyperparameter Grid for LightGBM
lgbm_param_grid = {
    'n_estimators': [1, 8, 16, 32, 64, 100, 200],
    'boosting_type': ['gbdt', 'dart']
}

# Perform Grid Search Cross Validation for LightGBM
lgbm_grid_search = GridSearchCV(estimator=lgbm, param_grid=lgbm_param_grid,
                                scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
lgbm_grid_search.fit(X_train, y_train)

# Best LightGBM Model
lgbm_best_model = lgbm_grid_search.best_estimator_
print(f"Best Parameters for LightGBM: {lgbm_grid_search.best_params_}")

# Predict and Evaluate LightGBM
lgbm_predictions = lgbm_best_model.predict(X_test)
print("LightGBM Classification Report:")
print(classification_report(y_test, lgbm_predictions))
print("LightGBM Accuracy:", accuracy_score(y_test, lgbm_predictions))

# ----------------- Comparison -----------------
