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
print("Training XGBoost Classifier...")
xgb = XGBClassifier(random_state=42)

# Hyperparameter Grid for XGBoost
xgb_param_grid = {
    'n_estimators': [1, 8, 16, 32, 64, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.2],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform Grid Search Cross Validation for XGBoost
xgb_grid_search = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, 
                               scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

# Best XGBoost Model
xgb_best_model = xgb_grid_search.best_estimator_
print(f"Best Parameters for XGBoost: {xgb_grid_search.best_params_}")

# Predict and Evaluate XGBoost
xgb_predictions = xgb_best_model.predict(X_test)
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_predictions))

