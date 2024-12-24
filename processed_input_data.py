import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import joblib 

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def assign_mcc_category(mcc):
    """
    Assigns an MCC (Merchant Category Code) to a predefined category based on its range.

    Parameters:
    - mcc (int): The Merchant Category Code to be classified.
    
    Returns:
    - str: The category name that the MCC falls into, or 'Unknown' if no match is found.
    """
    mcc_categories = {
        "Agricultural Services": (1, 1499),
        "Contracted Services": (1500, 2999),
        "Travel": (3000, 3999),
        "Transportation Services": (4000, 4799),
        "Utility Services": (4800, 4999),
        "Retail Outlet Services": (5000, 5599),
        "Clothing Stores": (5600, 5699),
        "Miscellaneous Stores": (5700, 7299),
        "Business Services": (7300, 7999),
        "Professional Services and Membership Organizations": (8000, 8999),
    }

    for category, (low, high) in mcc_categories.items():
        if low <= mcc <= high:
            return category
    return "Unknown"

def convert_mcc_into_two_groups(data):
    """
    Converts the 'MCC Category' into two groups: 'Miscellaneous Stores' and 'Other'.

    Parameters:
    - data (DataFrame): The DataFrame containing the 'MCC Category' column to be classified.
    
    Returns:
    - DataFrame: The input DataFrame with a new column 'MCC Category group' representing the two groups.
    """
    data['MCC Category group'] = data['MCC Category'].apply(
        lambda x: 'Miscellaneous Stores' if x == 'Miscellaneous Stores' else 'Other'
    )

    return data

def check_null_values(data):
    """
    Checks for null (missing) values in all columns of the dataset.

    Parameters:
    - data (DataFrame): The DataFrame to check for missing values.
    
    Returns:
    - None: This function prints the count of null values per column.
    """
    null_values_count = data.isnull().sum()


def fill_missing_values(data, column_name, fill_value='None'):
    """
    Fills missing values in the specified column with a given fill value.

    Parameters:
    - data (DataFrame): The DataFrame containing the column with missing values.
    - column_name (str): The name of the column to fill missing values in.
    - fill_value (str, optional): The value to replace missing values with. Default is 'None'.
    
    Returns:
    - DataFrame: The input DataFrame with the missing values in the specified column filled.
    """
    data[column_name] = data[column_name].fillna(fill_value)
    data[column_name] = data[column_name].astype(str)
    return data

def extract_p_value(fee):
    """
    Extracts the 'p' value from a fee string using a regular expression.

    Parameters:
    - fee (str): The fee string containing the 'p' value to extract.
    
    Returns:
    - str or None: The extracted number as a string if found, otherwise None.
    """
    match = re.search(r'(\d+)p', fee)
    return match.group(1) if match else None

def add_p_value_columns(data, columns_to_check):
    """
    Adds new columns for extracted 'p' values from specified columns.
    
    This function extracts the 'p' values from each of the given columns and creates new columns
    representing these extracted values.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the columns to process.
    - columns_to_check (list of str): A list of column names that contain fee values with 'p' values.
    
    Returns:
    - DataFrame: The input DataFrame with new columns added, where each new column contains the extracted 'p' values.
    """
    for col in columns_to_check:
        data[f'{col}_p_value'] = data[col].apply(extract_p_value)
    return data

def check_p_value_consistency(data, columns_to_check):
    """
    Checks if 'p' values are the same across specified columns for each row.

    Parameters:
    - data (DataFrame): The DataFrame containing the extracted 'p' value columns.
    - columns_to_check (list of str): A list of column names containing the extracted 'p' values to check for consistency.
    
    Returns:
    - DataFrame: The input DataFrame with a new column 'Same_p_value' indicating whether the 'p' values are consistent across columns.
    """
    data['Same_p_value'] = data.apply(
        lambda row: len(set(row[f'{col}_p_value'] for col in columns_to_check)) == 1, axis=1
    )
    return data


def get_inconsistent_rows(data):
    """
    Returns rows where 'p' values are not consistent across specified columns.

    Parameters:
    - data (DataFrame): The DataFrame containing the 'Same_p_value' column to check for consistency.
    
    Returns:
    - DataFrame: A subset of the input DataFrame containing only the rows where 'p' values are inconsistent.
    """
    return data[data['Same_p_value'] == False]

def extract_percentage_fixed_from_fees(data):
    """
    Extracts percentage values and fixed charges from fee columns and adds them as new columns.
    
    This function uses regular expressions to extract the percentage values (in decimal form) 
    from columns representing different fee types (Mastercard Debit, Mastercard Credit, etc.) 
    and the fixed charge (in 'p' value) from the fee columns. It then creates new columns with the extracted values.
    
    Parameters:
    - data (DataFrame): The DataFrame containing fee columns from which the percentage and fixed charge values 
                         are to be extracted.
    
    Returns:
    - DataFrame: The input DataFrame with new columns representing the extracted percentage and fixed charge values.
    """
    data['Mastercard Debit Percentage'] = data['Mastercard Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Mastercard Credit Percentage'] = data['Mastercard Credit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Mastercard Business Debit Percentage'] = data['Mastercard Business Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Visa Debit Percentage'] = data['Visa Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Visa Credit Percentage'] = data['Visa Credit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Visa Business Debit Percentage'] = data['Visa Business Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Fixed Charge (p)'] = data['Mastercard Debit'].str.extract(r'(\d+)p').astype(float).fillna(1)
    return data

def calculate_total_annual_transaction_fees(data):
    """
    Calculates the total annual transaction fees based on card turnover and fee percentages.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the necessary columns, such as 'Annual Card Turnover', 
                         the extracted fee percentages, and the fixed charges.
    
    Returns:
    - DataFrame: The input DataFrame with a new column 'Total Annual Transaction Fees', which contains 
                 the calculated total fees for each row.
    """
    # Calculate Total annual Transaction fees (excluding the fixed part initially)
    data['Total Annual Transaction Fees'] = (
        data['Annual Card Turnover'] * 0.4 * 0.9 * data['Mastercard Debit Percentage'] +
        data['Annual Card Turnover'] * 0.4 * 0.08 * data['Mastercard Credit Percentage'] +
        data['Annual Card Turnover'] * 0.4 * 0.02 * data['Mastercard Business Debit Percentage'] +
        data['Annual Card Turnover'] * 0.6 * 0.9 * data['Visa Debit Percentage'] +
        data['Annual Card Turnover'] * 0.6 * 0.08 * data['Visa Credit Percentage'] +
        data['Annual Card Turnover'] * 0.6 * 0.02 * data['Visa Business Debit Percentage']     
    )

    # Incorporate the fixed part of the fees
    data['Total Annual Transaction Fees'] += (
        data['Annual Card Turnover'] / data['Average Transaction Amount']
    ) * (data['Fixed Charge (p)'] / 100)

    return data

def update_payment_card_columns(data, fixed_charge_column, avg_transaction_amount_column, percentage_columns):
    """
    Calculate new columns based on payment network and card type weights.

    Parameters:
    - data (DataFrame): Input DataFrame containing required columns.
    - fixed_charge_column (str): Column name for fixed charges.
    - avg_transaction_amount_column (str): Column name for average transaction amount.
    - percentage_columns (dict): Mapping of output column names to their corresponding percentage columns.

    Returns:
    - DataFrame: The input DataFrame with new columns added.
    """
    # Define weights for payment networks and card types
    w_payment_network_mastercard = 0.4
    w_payment_network_visa = 0.6
    w_card_type_debit = 0.9
    w_card_type_credit = 0.08
    w_card_type_business_debit = 0.02

    # Define a dictionary to hold weights for card types
    card_type_weights = {
        'Mastercard Debit': w_payment_network_mastercard * w_card_type_debit,
        'Visa Debit': w_payment_network_visa * w_card_type_debit,
        'Mastercard Credit': w_payment_network_mastercard * w_card_type_credit,
        'Visa Credit': w_payment_network_visa * w_card_type_credit,
        'Mastercard Business Debit': w_payment_network_mastercard * w_card_type_business_debit,
        'Visa Business Debit': w_payment_network_visa * w_card_type_business_debit
    }

    # Iterate through percentage_columns and calculate new values
    for output_column, percentage_column in percentage_columns.items():
        data[output_column] = (
            card_type_weights[output_column] * data[percentage_column]
            + (data[fixed_charge_column] / (100 * data[avg_transaction_amount_column]))
        )

    return data

def process_current_providers(data):
    """
    Processes the 'Current Provider' column by:
    - Converting all entries to lowercase.
    - Removing non-alphanumeric characters.
    - Calculating the frequency of each provider.
    - Calculating the total number of unique providers.

    Parameters:
    - data (DataFrame): Input DataFrame containing the 'Current Provider' column.

    Returns:
    - DataFrame: Modified DataFrame with processed 'Current Provider' column.
    - int: Total number of unique providers.
    - Series: Frequency count of each provider.
    """
    # Make 'Current Provider' lowercase and remove non-alphanumeric characters
    data['Current Provider'] = data['Current Provider'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    # Calculate frequency of each provider
    provider_frequency = data['Current Provider'].value_counts()

    # Calculate total number of unique providers
    unique_providers_count = data['Current Provider'].nunique()

    return data, unique_providers_count, provider_frequency

def group_columns_into_two_categories(data):
    """
    Groups the 'Current Provider' column into two categories: 'Empty' and 'Other' 
    for the purpose of making the dataset more balanced by consolidating 'none' values 
    into a single category.

    Parameters:
    - data (DataFrame): The input DataFrame containing the 'Current Provider' column to be transformed.
    
    Returns:
    - DataFrame: The input DataFrame with an added column 'Current Provider Grouped' representing 
                 the grouped categories ('Empty' or 'Other').
    """
    # Group 'Current Provider' into two categories: 'Empty' and 'Other' to make the dataset balanced
    data['Current Provider Grouped'] = data['Current Provider'].apply(
        lambda x: 'Empty' if x == 'none' else 'Other'
    )

    return data

def encode_columns(data, column_name_list):
    """
    Encodes categorical columns into numerical values using Label Encoding.

    Parameters:
    - data (DataFrame): The input DataFrame containing the columns to be encoded.
    - column_name_list (list): A list of column names in the DataFrame that need to be encoded.
    
    Returns:
    - DataFrame: The input DataFrame with the specified columns encoded into numerical values.
    """
    # Create a label encoder
    label_encoder = LabelEncoder()
    for column in column_name_list:
        # Encode 'Registered' and 'Accepts Card' columns
        data[column] = label_encoder.fit_transform(data[column])
    
    return data

def save_fitted_scaler(data, column_name_list, scaler_path):
    """
    Fits a MinMaxScaler to the specified columns and saves the scaler to a file.

    Parameters:
    - data (DataFrame): The input DataFrame containing the columns to be scaled.
    - column_name_list (list): A list of column names in the DataFrame to fit the scaler on.
    - scaler_path (str): Path to save the fitted scaler.
    
    Returns:
    - MinMaxScaler: The fitted scaler object.
    """
    scaler = MinMaxScaler()
    scaler.fit(data[column_name_list])
    joblib.dump(scaler, scaler_path)  # Save the fitted scaler
    return scaler

def scale_columns(data, column_name_list, loaded_scaler):
    """
    Scales the specified columns using MinMax scaling to transform the data into a range between 0 and 1.

    Parameters:
    - data (DataFrame): The input DataFrame containing the columns to be scaled.
    - column_name_list (list): A list of column names in the DataFrame that need to be scaled.
    
    Returns:
    - DataFrame: The input DataFrame with new columns containing the scaled values, 
                 named as '{original_column_name} Scaled'.
    """
    # Initialize the MinMaxScaler

    scaled_values = loaded_scaler.transform(data[column_name_list])  # Transform all columns together
    for i, column in enumerate(column_name_list):
        data[column + ' Scaled'] = scaled_values[:, i]  # Assign scaled values to new columns

    return data


def input_data_preprocessing(data, path_to_stored_scaling):
    """
    Preprocesses the given input row/rows of data by performing multiple steps such as categorizing, cleaning, transforming, 
    encoding, and scaling. This function involves:
    - Assigning MCC categories based on MCC codes.
    - Handling missing values and inconsistencies.
    - Extracting and transforming percentage and fixed charge values.
    - Calculating annual transaction fees.
    - Encoding categorical columns.
    - Dropping redundant or highly correlated columns.

    Parameters:
    - data (DataFrame): The input DataFrame containing raw data.

    Returns:
    - DataFrame: The preprocessed DataFrame after applying all transformations.
    """
    # Step 1: Assign categories to 'MCC Category' based on MCC Code
    data['MCC Category'] = data['MCC Code'].apply(assign_mcc_category)

    # Step 2: Convert 'MCC Category' into two groups for better classification
    data = convert_mcc_into_two_groups(data)

    # Step 3: Check for missing (null) values across all columns
    check_null_values(data)

    # Step 4: Fill missing values in the 'Current Provider' column with default value
    data = fill_missing_values(data, 'Current Provider')

    # Step 5: Extract 'p' values from certain columns and add new columns for 'p' values
    columns_to_check = ['Mastercard Debit', 'Visa Debit', 'Mastercard Credit', 'Visa Credit', 'Mastercard Business Debit', 'Visa Business Debit']
    data = add_p_value_columns(data, columns_to_check)

    # Step 6: Check for consistency in 'p' values across the columns
    data = check_p_value_consistency(data, columns_to_check)

    # Step 7: Retrieve rows where 'p' values are inconsistent
    inconsistent_rows = get_inconsistent_rows(data)

    # Step 8: Extract percentage values from fee columns and calculate their fixed charge
    data = extract_percentage_fixed_from_fees(data)

    # Step 9: Calculate total annual transaction fees based on various percentages
    calculate_total_annual_transaction_fees(data)

    # Step 10: Drop columns related to 'p' values and consistency checks
    columns_to_drop = [
        'Mastercard Debit_p_value', 'Visa Debit_p_value',
        'Mastercard Credit_p_value', 'Visa Credit_p_value',
        'Mastercard Business Debit_p_value', 'Visa Business Debit_p_value',
        'Same_p_value',
    ]
    data = data.drop(columns=columns_to_drop)

    # Step 11: Map percentage columns to new names for clarity
    percentage_columns_mapping = {
        'Mastercard Debit': 'Mastercard Debit Percentage',
        'Visa Debit': 'Visa Debit Percentage',
        'Mastercard Credit': 'Mastercard Credit Percentage',
        'Visa Credit': 'Visa Credit Percentage',
        'Mastercard Business Debit': 'Mastercard Business Debit Percentage',
        'Visa Business Debit': 'Visa Business Debit Percentage'
    }

    # Step 12: Update payment card columns with fixed charge and average transaction amount
    data = update_payment_card_columns(
        data,
        fixed_charge_column='Fixed Charge (p)',
        avg_transaction_amount_column='Average Transaction Amount',
        percentage_columns=percentage_columns_mapping
    )

    # Step 13: Drop original columns after transformation
    col = ["MCC Code",'Mastercard Debit Percentage',
           'Mastercard Credit Percentage', 'Mastercard Business Debit Percentage',
           'Visa Debit Percentage', 'Visa Credit Percentage',
           'Visa Business Debit Percentage', 'Fixed Charge (p)']
    data = data.drop(columns=col)

    # Step 14: Process current providers to obtain unique count and frequencies
    data, unique_providers_count, provider_frequency = process_current_providers(data)

    # Step 15: Group 'Current Provider' into 'Empty' and 'Other' categories for balance
    data = group_columns_into_two_categories(data)

    # Step 16: Drop 'Current Provider' column after grouping
    data = data.drop(columns=['Current Provider'])

    # Step 17: Calculate transaction fees per unit turnover
    data['Transaction Fees per Unit Turnover'] = data['Total Annual Transaction Fees'] / data['Annual Card Turnover']

    # Step 18: Encode categorical columns
    column_name_list = ['Is Registered', 'Accepts Card', 'MCC Category group', 'Current Provider Grouped']
    data = encode_columns(data, column_name_list)

    # Drop the 'MCC Category' column after encoding
    data = data.drop(columns=['MCC Category'])

    # Step 19: Drop highly correlated columns
    data = data.drop(columns=['Current Provider Grouped', 'Mastercard Debit', 'Mastercard Credit', 'Mastercard Business Debit', 'Total Annual Transaction Fees', 'Visa Debit'])

    # Step 20: Categorize 'Transaction Fees per Unit Turnover' into pricing categories
    column_name_list = [
        'Transaction Fees per Unit Turnover',
        'Average Transaction Amount', 
        'Visa Credit', 
        'Annual Card Turnover',
        'Visa Business Debit'
    ]

    # Define the path where the fitted scaler will be saved or loaded
    scaler_path = path_to_stored_scaling # "scaler.pkl"

    # Uncomment the following line if you need to save a fitted scaler based on all rows in the dataset
    # This step is done only once to fit the scaler on the entire dataset and save it for future use
    # save_fitted_scaler(data, column_name_list, scaler_path)
    
    # Load the previously saved MinMaxScaler
    # This ensures consistency in scaling across different subsets of the data (e.g., train/test)
    loaded_scaler = joblib.load(scaler_path)

    # Step 21: Scale specified columns using MinMaxScaler
    data = scale_columns(data, column_name_list, loaded_scaler)

    # Step 22: Drop scaled columns and other specified columns
    data = data.drop(columns=['Average Transaction Amount', 'Annual Card Turnover', 
                              'Transaction Fees per Unit Turnover', 'Visa Credit', 'Visa Business Debit'])

    # Step 23: Return the final processed data
    return data

if __name__ == "__main__":
    file_path = 'data//data.csv'

    data = pd.read_csv(file_path)

    path_to_stored_scaling = 'pickle_files//scaler.pkl'
    data = input_data_preprocessing(data, path_to_stored_scaling)

    print(data.head())
