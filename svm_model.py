import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


file_path = 'updated_transaction_data_withlabels3.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

#print(data["Current Provider"].value_counts())

print(data.columns)

label_column = 'Current pricing'  
#X = data.drop(columns=[label_column])  # Features
data['encoded__Miscellaneous Stores'] = data['encoded__Miscellaneous Stores'].astype(int)

# 'Annual Card Turnover Scaled',
# 'Visa Debit Scaled',
# 'Visa Credit Scaled',
# 'Visa Business Debit Scaled',

X = data.drop(columns=[label_column,
'Transaction Fees per Unit Turnover_Scaled',
'Visa Debit Scaled',
'Total Annual Transaction Fees Scaled'
])

print(X.head())
  # Features
y = data[label_column]  # Target labels (already encoded)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

le = LabelEncoder()
y = le.fit_transform(y)

# Data preprocessing: Scale the features

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10],                # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],   # Kernel types
    'degree': [2, 3],                   # Degree of the polynomial kernel (if using 'poly')
    'gamma': ['scale', 'auto'] # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
}

# Initialize the SVM classifier
svm = SVC()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

best_svm = grid_search.best_estimator_
best_svm.fit(X_train, y_train)

# Evaluate the model
y_pred = best_svm.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()