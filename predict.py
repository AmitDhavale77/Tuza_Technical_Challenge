import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from processed_input_data import input_data_preprocessing

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None
# Assuming rf_clf is your trained model and le is your label encoder

# Define the prediction function
def predict_fees_category(data, model, label_encoder):
    """
    Predicts the pricing category (e.g., 'Competitive', 'Neutral', 'Non-Competitive') 
    for the given input data based on a trained machine learning model.
    
    Parameters:
    - data (DataFrame or Series): The input data for which predictions are to be made. 
      This can either be a single row (Series) or a DataFrame with multiple rows.
    - model (sklearn.ensemble.RandomForestClassifier): The trained machine learning model used for prediction.
    - label_encoder (LabelEncoder): A fitted LabelEncoder used to decode the predicted labels back to their original categories.

    Returns:
    - predicted_labels (array): An array of predicted categories (e.g., 'Competitive', 'Neutral', 'Non-Competitive') 
      for each row in the input data.
    """
    # Check if the input is a single row or a DataFrame
    if isinstance(data, pd.Series):
        # If it's a single row (Series), convert it to a DataFrame to maintain consistency
        # data.to_frame().T
        data = data.to_frame().T   # Transpose the row to match DataFrame format
    
    # Preprocess the input data using the custom preprocessing function
    X = input_data_preprocessing(data, path_to_stored_scaling="pickle_files//scaler.pkl")

    # Drop any unnecessary columns that are not used for prediction
    columns_to_drop = ['Transaction Fees per Unit Turnover Scaled']
    X = X.drop(columns=columns_to_drop, errors='ignore')  # Ignore errors in case columns are missing

    # # Ensure that the row(s) have the same columns as the training set (align with training data)
    # X = X.reindex(columns=X.columns, fill_value=0)  # Add missing columns with 0 values (fill_value=0)
    
    # Make predictions using the model (RandomForestClassifier in this case)
    y_pred = model.predict(X)  # Predict for all rows in the DataFrame

    # Decode the predicted labels back to the original categories (e.g., 'competitive', 'neutral', 'non-competitive')
    predicted_labels = label_encoder.inverse_transform(y_pred)

    return predicted_labels

# Example input row (for demonstration purposes, this would need to match your data's structure)
# row = pd.Series({
#     'MCC Code': 'Some MCC', 
#     'Mastercard Debit': 0.1, 
#     'Mastercard Credit': 0.2,
#     'Mastercard Business Debit': 0.3,
#     'Visa Debit': 0.4, 
#     'Visa Credit': 0.5,
#     'Visa Business Debit': 0.6,
#     # Add all other features required by the model
# })
# data = pd.DataFrame([input_data_dict])

if __name__ == "__main__":
    file_path = 'data//data.csv'
    data = pd.read_csv(file_path)

    # Load the saved model
    rf_clf = joblib.load('pickle_files//random_forest_model.pkl')
    le = joblib.load('pickle_files//label_encoder.pkl')

    # Predict fee category for a single row (e.g., the first row of the dataset)
    predicted_category = predict_fees_category(data.iloc[1], rf_clf, le)  # Data must be 2D for prediction
    print(f"The predicted fee category for this business is : {predicted_category}")

    # Example usage for multiple rows (e.g., the first 5 rows of the dataset)
    predicted_categories = predict_fees_category(data.head(), rf_clf, le)
    print(f"The predicted fee categories for the businesses are : {predicted_categories}")