import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    make_scorer,
)
import seaborn as sns
import matplotlib.pyplot as plt

def random_forest_pipeline(file_path, label_column, excluded_columns=None, save_model=False, model_path="random_forest_model.pkl", save_encoder=False, encoder_path="label_encoder.pkl"):
    """
    Executes the Random Forest classification pipeline including data preprocessing,
    model training, evaluation, and optional saving of the model and label encoder.

    Parameters:
    - file_path (str): Path to the CSV dataset.
    - label_column (str): Name of the target column.
    - excluded_columns (list, optional): Columns to exclude from features.
    - save_model (bool, optional): Whether to save the trained model.
    - model_path (str, optional): Path to save the trained model.
    - save_encoder (bool, optional): Whether to save the label encoder.
    - encoder_path (str, optional): Path to save the label encoder.
    """

    # Step 1: Load the dataset
    data = pd.read_csv(file_path)

    # Step 2: Prepare the feature set (X) by dropping unwanted columns
    if excluded_columns is None:
        excluded_columns = []
    X = data.drop(columns=[label_column] + excluded_columns)

    # Step 3: Extract and encode the target variable (y)
    y = data[label_column]
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Optional: Save the fitted LabelEncoder
    if save_encoder:
        joblib.dump(le, encoder_path)

    # Step 4: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Initialize the Random Forest Classifier
    rf_clf = RandomForestClassifier(min_samples_leaf=4, random_state=42, n_estimators=100)

    # Step 6: Perform 5-fold cross-validation to evaluate the model
    cv_scores = cross_validate(rf_clf, X, y, cv=5, scoring='accuracy', return_train_score=True)
    precision_scorer = make_scorer(precision_score, average='weighted')
    recall_scorer = make_scorer(recall_score, average='weighted')
    cv_precision = cross_validate(rf_clf, X, y, cv=5, scoring=precision_scorer, return_train_score=True)
    cv_recall = cross_validate(rf_clf, X, y, cv=5, scoring=recall_scorer, return_train_score=True)

    # Extracting the train and test (validation) metrics
    train_accuracies = cv_scores['train_score']
    val_accuracies = cv_scores['test_score']

    train_precision = cv_precision['train_score']
    test_precision = cv_precision['test_score']

    train_recall = cv_recall['train_score']
    test_recall = cv_recall['test_score']
    # Step 7: Display cross-validation results
    print(f"\nTrain Accuracies: {train_accuracies}")
    print(f"Validation Accuracies: {val_accuracies}")
    print(f"Mean Train Accuracy: {np.mean(train_accuracies):.4f}")
    print(f"Mean Validation Accuracy: {np.mean(val_accuracies):.4f}")

    print(f"\nCross-Validation Train Precision Scores: {train_precision}")
    print(f"Cross-Validation Test Precision Scores: {test_precision}")
    print(f"Mean Train Precision: {np.mean(train_precision):.4f}")
    print(f"Mean Test Precision: {np.mean(test_precision):.4f}")

    print(f"\nCross-Validation Train Recall Scores: {train_recall}")
    print(f"Cross-Validation Test Recall Scores: {test_recall}")
    print(f"Mean Train Recall: {np.mean(train_recall):.4f}")
    print(f"Mean Test Recall: {np.mean(test_recall):.4f}")

    # Step 8: Generate predictions using cross-validation
    y_pred_cv = cross_val_predict(rf_clf, X, y, cv=5)

    # Step 9: Calculate and display the confusion matrix
    cv_conf_matrix = confusion_matrix(y, y_pred_cv)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cv_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')
    plt.title('Cross-Validation Confusion Matrix', fontweight='bold')
    plt.show()

    # Step 10: Train the model on the training set
    rf_clf.fit(X_train, y_train)

    # Step 11: Save the trained model
    if save_model:
        joblib.dump(rf_clf, model_path)

    # Step 12: Load the trained model and make predictions (optional)
    if save_model:
        loaded_rf_clf = joblib.load(model_path)
        predictions = loaded_rf_clf.predict(X_test)
        print("\nSample Predictions:", predictions)

if __name__ == "__main__":
    # Example Usage
    random_forest_pipeline(
        file_path='updated_transaction_data_with_labels.csv',
        label_column='Current pricing',
        excluded_columns=['Transaction Fees per Unit Turnover Scaled'],
        save_model=True,
        save_encoder=False
    )
