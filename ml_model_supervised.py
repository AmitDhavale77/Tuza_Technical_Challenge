import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_val_predict


pd.set_option('display.max_rows', None)  # No limit on rows
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Auto-detect width
pd.set_option('display.max_colwidth', None)  # No limit on column width


file_path = 'updated_transaction_data_withlabels4.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

#print(data["Current Provider"].value_counts())

print(data.columns)

label_column = 'Current pricing'  
#X = data.drop(columns=[label_column])  # Features
# data['encoded__Miscellaneous Stores'] = data['encoded__Miscellaneous Stores'].astype(int)

# 'Annual Card Turnover Scaled',
# 'Visa Debit Scaled',
# 'Visa Credit Scaled',
# 'Visa Business Debit Scaled',

X = data.drop(columns=[label_column,
'Transaction Fees per Unit Turnover Scaled',
])
  # Features
y = data[label_column]  # Target labels (already encoded)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

le = LabelEncoder()
y = le.fit_transform(y)

len(X.iloc[0])
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the lengths of the training and test sets
print(f"Length of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")
print(f"Length of y_train: {len(y_train)}")
print(f"Length of y_test: {len(y_test)}")

rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model using cross-validation
cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')  # 5-fold CV
# Define precision and recall scorers
precision_scorer = make_scorer(precision_score, average='weighted')  # 'weighted' averages precision for multi-class problems
recall_scorer = make_scorer(recall_score, average='weighted')  # 'weighted' averages recall for multi-class problems

# Perform cross-validation with precision and recall scores
cv_precision = cross_val_score(rf_clf, X, y, cv=5, scoring=precision_scorer)
cv_recall = cross_val_score(rf_clf, X, y, cv=5, scoring=recall_scorer)

# Train the model on the full training set
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")

print(f"Cross-Validation Precision Scores: {cv_precision}")
print(f"Mean CV Precision: {cv_precision.mean():.4f}")
print(f"Standard Deviation of CV Precision: {cv_precision.std():.4f}")

print(f"Cross-Validation Recall Scores: {cv_recall}")
print(f"Mean CV Recall: {cv_recall.mean():.4f}")
print(f"Standard Deviation of CV Recall: {cv_recall.std():.4f}")

# Perform cross-validation to get predictions on all folds
y_pred_cv = cross_val_predict(rf_clf, X, y, cv=5)

# Calculate the overall confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Assuming y and y_pred_cv are already defined
cv_conf_matrix = confusion_matrix(y, y_pred_cv)

# Define the labels
labels = ['competitive', 'neutral', 'non-competitive']

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cv_conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=labels, 
    yticklabels=labels
)
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.title('Cross-Validation Confusion Matrix', fontweight='bold')
plt.show()










# Evaluate the model on the test set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and display accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")















# Define individual classifiers
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
gb_clf = GradientBoostingClassifier(random_state=42)
ada_clf = AdaBoostClassifier(random_state=42)
logreg_clf = LogisticRegression(random_state=42, max_iter=1000)
svc_clf = SVC(probability=True, random_state=42)  # Enable probability for soft voting
dt_clf = DecisionTreeClassifier(random_state=42)
nb_clf = GaussianNB()

# Combine classifiers using VotingClassifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('logreg', logreg_clf),
        ('svc', svc_clf),
        ('nb', nb_clf)
    ],
    voting='soft'  # Use 'soft' voting for probability-based aggregation
)

# Train the VotingClassifier
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold CV

# Train the model on the full training set
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")


# Evaluate the model on the test set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display cross-validation results

# Calculate and display accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")