import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'updated_transaction_data_withlabels3.csv'
data = pd.read_csv(file_path)

# Print columns to verify the dataset structure
print(data.columns)

label_column = 'Current pricing'  

# Preprocess the data (you can modify this as needed)
data['encoded__Miscellaneous Stores'] = data['encoded__Miscellaneous Stores'].astype(int)

X = data.drop(columns=[label_column, 'Transaction Fees per Unit Turnover_Scaled', 'Visa Debit Scaled', 'Total Annual Transaction Fees Scaled'])
y = data[label_column]  # Target column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the Naive Bayes classifier
nb_clf = GaussianNB()

# Define the parameter grid for GridSearchCV
param_grid = {
    'var_smoothing': np.logspace(0,-9, num=100)  # This is a typical hyperparameter for Naive Bayes
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(nb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# Best estimator
best_nb_clf = grid_search.best_estimator_

# Perform cross-validation with the best model
cv_scores = cross_val_score(best_nb_clf, X_train, y_train, cv=5, scoring='accuracy')

# Print the cross-validation results
print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")

# Train the model on the full training set with the best parameters
best_nb_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_nb_clf.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()