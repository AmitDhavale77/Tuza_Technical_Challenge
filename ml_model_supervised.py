import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


pd.set_option('display.max_rows', None)  # No limit on rows
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Auto-detect width
pd.set_option('display.max_colwidth', None)  # No limit on column width


file_path = 'updated_transaction_data_withlabels1.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

#print(data["Current Provider"].value_counts())

print(data.columns)

label_column = 'Current pricing'  
#X = data.drop(columns=[label_column])  # Features


X = data.drop(columns=[label_column,'Transaction Fees per Unit Turnover_Scaled'])  # Features
y = data[label_column]  # Target labels (already encoded)

len(X.iloc[0])
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the lengths of the training and test sets
print(f"Length of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of y_test: {len(y_test)}")

print(y_test)
# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42) #random_state=42 a seed for random number generator
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print(y_test)
print(y_pred)



from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize the classifier
model = RandomForestClassifier(random_state=42)

# Define a stratified k-fold cross-validator
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Output cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of Accuracy: {cv_scores.std():.4f}")

overall_cm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

# Detailed evaluation on one split with confusion matrix
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Fold Accuracy:", accuracy_score(y_test, y_pred))
    
    # Compute and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    overall_cm += cm

print("Overall Confusion Matrix:")
print(overall_cm)  


# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [1, 8, 16, 32, 64, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the Random Forest classifier
rf = RandomForestClassifier()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)


# Train the model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Evaluate the model
y_pred = best_rf.predict(X_test)

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




# # Define outer StratifiedKFold cross-validation
# outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Store cross-validation scores
# cv_scores = []
# overall_cm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

# # Perform nested cross-validation with hyperparameter tuning
# for train_index, test_index in outer_kf.split(X, y):
#     X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

#     # Define the inner StratifiedKFold cross-validation for hyperparameter tuning
#     inner_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#     # Perform GridSearchCV for hyperparameter tuning inside the inner loop
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_kf, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X_train_cv, y_train_cv)

#     # Best hyperparameters from GridSearchCV
#     print(f"Best parameters: {grid_search.best_params_}")

#     # Train the model with the best parameters
#     best_model = grid_search.best_estimator_

#     # Predict on the test set of the outer fold
#     y_pred_cv = best_model.predict(X_test_cv)

#     # Evaluate the performance
#     print("\nClassification Report (Outer fold):")
#     print(classification_report(y_test_cv, y_pred_cv))
#     print("Accuracy:", accuracy_score(y_test_cv, y_pred_cv))

#     # Compute and accumulate confusion matrix
#     cm = confusion_matrix(y_test_cv, y_pred_cv)
#     overall_cm += cm
#     cv_scores.append(accuracy_score(y_test_cv, y_pred_cv))


# # Best parameters: {'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}

# best_model = RandomForestClassifier(random_state=42, min_samples_leaf=1, min_samples_split=5, n_estimators=100)


# # Define a stratified k-fold cross-validator
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Perform cross-validation
# cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# # Output cross-validation results
# print(f"Cross-Validation Accuracy Scores: {cv_scores}")
# print(f"Mean Accuracy: {cv_scores.mean():.4f}")
# print(f"Standard Deviation of Accuracy: {cv_scores.std():.4f}")

# overall_cm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

# # Detailed evaluation on one split with confusion matrix
# for train_index, test_index in kf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#     best_model.fit(X_train, y_train)
#     y_pred = best_model.predict(X_test)
    
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
#     print("Fold Accuracy:", accuracy_score(y_test, y_pred))
    
#     # Compute and print confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     overall_cm += cm

# print("Overall Confusion Matrix:")
# print(overall_cm)  




# # Calculate precision, recall, and f1-score for each class using the overall confusion matrix
# precision = overall_cm.diagonal() / overall_cm.sum(axis=0)
# recall = overall_cm.diagonal() / overall_cm.sum(axis=1)
# f1 = 2 * (precision * recall) / (precision + recall)
# overall_accuracy = np.sum(overall_cm.diagonal()) / np.sum(overall_cm)

# # Print the results
# print("Precision for each class:", precision)
# print("Recall for each class:", recall)
# print("F1-Score for each class:", f1)

# print("Total precision:", np.mean(precision))
# print("Total recall:", np.mean(recall))
# print("Total F1 score:", np.mean(f1))
# print("Overall Accuracy:", overall_accuracy)



