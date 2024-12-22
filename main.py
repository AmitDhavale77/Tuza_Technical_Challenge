import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from processed_input_data import input_data_preprocessing
from ml_model_supervised import random_forest_pipeline
from predict import predict_fees_category
from data_processing import data_preprocessing

def run_data_preprocessing(filename):
    """
    Preprocesses the raw input data and saves the processed data to a new CSV file.
    
    Parameters:
        filename (str): The path to the raw dataset file.
    """

    file_path = filename
    data = pd.read_csv(file_path)
    data = data_preprocessing(data)
    data.to_csv('updated_transaction_data_with_labels.csv', index=False)
    print("CSV file has been saved successfully!")

def run_model_training(path_to_processed_csv):
    """
    Trains a Random Forest model using the preprocessed dataset and evaluates it.
    
    Parameters:
        path_to_processed_csv (str): The path to the preprocessed dataset CSV.
    """

    file_path = path_to_processed_csv
    random_forest_pipeline(
        file_path='updated_transaction_data_with_labels.csv',
        label_column='Current pricing',
        excluded_columns=['Transaction Fees per Unit Turnover Scaled'],
        save_model=False,
        save_encoder=False
    )

def make_predictions(input_data, path_to_trained_model, path_to_encoded_labels):
    """
    Predicts the fee category for a given input using the trained Random Forest model.
    
    Parameters:
        input_data (pd.Series): A single row of input data for prediction.
        path_to_trained_model (str): Path to the saved trained model file.
        path_to_encoded_labels (str): Path to the saved label encoder file.
    """

    # Load the saved model and label encodings
    rf_clf = joblib.load(path_to_trained_model)
    le = joblib.load(path_to_encoded_labels)

    # Predict fee category for a single row (e.g., the first row of the dataset)
    predicted_category = predict_fees_category(input_data, rf_clf, le)  # Data must be 2D for prediction
    print(f"The predicted fee category for this business is 1: {predicted_category}")

if __name__ == "__main__":

    # Function to preprocess the input dataset and save it as a new CSV
    filename = 'data.csv'
    run_data_preprocessing(filename)

    # Function to train the Random Forest model using the preprocessed data obtained from run_data_preprocessing()
    path_to_processed_csv = 'updated_transaction_data_with_labels.csv'
    run_model_training(path_to_processed_csv)

    # Function to make predictions using the trained model
    path_to_trained_model = 'random_forest_model.pkl'
    path_to_encoded_labels = 'label_encoder.pkl'

    # Example of expected type of input row (for demonstration purposes)
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

    # Example input row for prediction 
    file_path = 'data.csv'
    data = pd.read_csv(file_path)

    input_data = data.iloc[5]

    # Note: The function can also handle multiple rows for prediction when provided as a pandas DataFrame.
    make_predictions(input_data, path_to_trained_model, path_to_encoded_labels)

