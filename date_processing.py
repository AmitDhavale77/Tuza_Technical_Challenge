import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', None)  # No limit on rows
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Auto-detect width
pd.set_option('display.max_colwidth', None)  # No limit on column width

def assign_mcc_category(mcc):
    
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

def print_mcc_groups(data):
    grouped_data = data.groupby('MCC Category').size()
    print("MCC Code Grouped by Categories:")
    print(grouped_data)

def convert_mcc_into_two_groups(data):
    data['MCC Category group'] = data['MCC Category'].apply(
        lambda x: 'Miscellaneous Stores' if x == 'Miscellaneous Stores' else 'Other'
    )

    return data


def check_null_values(data):
    null_values_count = data.isnull().sum()
    print("Count of Null Values in Each Column:")
    print(null_values_count)

def fill_missing_values(data, column_name, fill_value='None'):
    """Fills missing values in the specified column."""
    data[column_name] = data[column_name].fillna(fill_value)
    data[column_name] = data[column_name].astype(str)
    return data

def extract_p_value(fee):
    """Extracts the 'p' value from the fee string using regex."""
    match = re.search(r'(\d+)p', fee)
    return match.group(1) if match else None

def add_p_value_columns(data, columns_to_check):
    """Adds columns for extracted 'p' values."""
    for col in columns_to_check:
        data[f'{col}_p_value'] = data[col].apply(extract_p_value)
    return data

def check_p_value_consistency(data, columns_to_check):
    """Checks if 'p' values are the same across specified columns."""
    data['Same_p_value'] = data.apply(
        lambda row: len(set(row[f'{col}_p_value'] for col in columns_to_check)) == 1, axis=1
    )
    return data

def get_inconsistent_rows(data):
    """Returns rows where 'p' values are not consistent."""
    return data[data['Same_p_value'] == False]

def extract_percentage_fixed_from_fees(data):
    data['Mastercard Debit Percentage'] = data['Mastercard Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Mastercard Credit Percentage'] = data['Mastercard Credit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Mastercard Business Debit Percentage'] = data['Mastercard Business Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Visa Debit Percentage'] = data['Visa Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Visa Credit Percentage'] = data['Visa Credit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Visa Business Debit Percentage'] = data['Visa Business Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
    data['Fixed Charge (p)'] = data['Mastercard Debit'].str.extract(r'(\d+)p').astype(float).fillna(1)
    return data

def calculate_total_annual_transaction_fees(data):
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
    # Group 'Current Provider' into two categories: 'None' and 'Other' to make the dataset balanced
    data['Current Provider Grouped'] = data['Current Provider'].apply(
        lambda x: 'Empty' if x == 'none' else 'Other'
    )
    # Check the first few rows of the new column
    print(data[['Current Provider', 'Current Provider Grouped']].head())
    print(data["Current Provider Grouped"].value_counts())

    return data

def encode_columns(data, column_name_list):
# Create a label encoder
    label_encoder = LabelEncoder()
    for column in column_name_list:
    # Encode 'Registered' and 'Accepts Card' columns
        data[column] = label_encoder.fit_transform(data[column])
    
    return data

def plot_spearman_correlation(data, figsize=(13, 9), cmap='coolwarm'):
    """
    Plots the Spearman correlation matrix for the given DataFrame.

    Parameters:
    - data (DataFrame): Input DataFrame.
    - figsize (tuple): Figure size for the plot.
    - cmap (str): Colormap for the heatmap.

    Returns:
    - None: Displays the heatmap.
    """

    # Calculate the Spearman correlation matrix
    spearman_corr = data.corr(method='spearman')

    # Plot the Spearman correlation matrix using a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        spearman_corr, 
        annot=True, 
        cmap=cmap, 
        fmt='.2f', 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1, 
        annot_kws={'size': 8}
    )

    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
    plt.yticks(fontsize=10)   # Rotate y-axis labels if needed
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout() 
    plt.show()

def scale_columns(data, column_name_list):
# Initialize the RobustScaler
    scaler = MinMaxScaler()

    for column in column_name_list:       
        # Reshape the data to a 2D array (required for scikit-learn's scaler)
        transaction_values = data[column].values.reshape(-1, 1)
        # Fit and transform the data using RobustScaler
        data[column + ' Scaled'] = scaler.fit_transform(transaction_values)

    return data

def plot_histogram(data, column, bins, figsize, title, xlabel, ylabel, xlim=None):
    """
    Plots a histogram for the specified column in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset containing the column.
        column (str): The name of the column for which to plot the histogram.
        bins (int): The number of bins in the histogram.
        figsize (tuple): The size of the figure (width, height).
        title (str): The title of the histogram.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        xlim (tuple, optional): The limits for the x-axis.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.hist(data[column], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.show()

def plot_histogram_with_quartiles(data, column, bins=200, figsize=(10, 6), xlim=None, 
                                  title='Histogram', xlabel='Values', ylabel='Frequency'):
    """
    Plots a histogram of the specified column with quartile lines (Q1, Median/Q2, Q3).

    Parameters:
    - data: DataFrame containing the data.
    - column: Column name to plot the histogram for.
    - bins: Number of bins for the histogram (default: 200).
    - figsize: Tuple specifying the figure size (default: (10, 6)).
    - xlim: Tuple specifying the x-axis limits (default: None).
    - title: Title of the plot (default: 'Histogram').
    - xlabel: Label for the x-axis (default: 'Values').
    - ylabel: Label for the y-axis (default: 'Frequency').
    """
    # Calculate quartiles
    Q1 = data[column].quantile(0.33)
    Q2 = data[column].quantile(0.50)  # Median
    Q3 = data[column].quantile(0.66)

    # Plotting the histogram
    plt.figure(figsize=figsize)
    plt.hist(data[column], bins=bins, edgecolor='black', alpha=0.7)

    # Add lines for Q1, Q2, Q3
    plt.axvline(Q1, color='red', linestyle='--', label=f'Q1 ({Q1:.4f})')
    plt.axvline(Q2, color='green', linestyle='--', label=f'Q2 (Median) ({Q2:.4f})')
    plt.axvline(Q3, color='blue', linestyle='--', label=f'Q3 ({Q3:.4f})')

    # Title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    # Show legend
    plt.legend()

    # Optional grid
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

def categorize_pricing(data, column_name):
    # Calculate quartiles (Q1, Q2, Q3)
    Q1 = data[column_name].quantile(0.33)
    Q2 = data[column_name].quantile(0.50)  # Median
    Q3 = data[column_name].quantile(0.66)

    # Define conditions based on quartiles
    conditions = [
        (data[column_name] <= Q1),  # competitive
        (data[column_name] > Q1) & (data[column_name] <= Q3),  # neutral
        (data[column_name] > Q3)   # non-competitive
    ]

    # Define corresponding labels
    labels = ['Competitive', 'Neutral', 'Non-Competitive']

    # Apply conditions and assign labels to the 'Current pricing' column
    data['Current pricing'] = np.select(conditions, labels)

    return data

def data_preprocessing(data):

    data['MCC Category'] = data['MCC Code'].apply(assign_mcc_category)

    print_mcc_groups(data)

    data = convert_mcc_into_two_groups(data)

    check_null_values(data)

    columns_to_check = ['Mastercard Debit', 'Visa Debit', 'Mastercard Credit', 'Visa Credit', 'Mastercard Business Debit', 'Visa Business Debit']

    # Fill missing values in 'Current Provider' column
    data = fill_missing_values(data, 'Current Provider')

    # Extract 'p' values and add corresponding columns
    data = add_p_value_columns(data, columns_to_check)

    # Check for consistency in 'p' values across columns
    data = check_p_value_consistency(data, columns_to_check)

    # Get rows with inconsistent 'p' values
    inconsistent_rows = get_inconsistent_rows(data)

    print("Rows with inconsistent 'p' values:")
    print(inconsistent_rows)  # [] implies that fixed charge for each payment is same

    data = extract_percentage_fixed_from_fees(data)

    calculate_total_annual_transaction_fees(data)

    columns_to_drop = [
        'Mastercard Debit_p_value', 'Visa Debit_p_value',
        'Mastercard Credit_p_value', 'Visa Credit_p_value',
        'Mastercard Business Debit_p_value', 'Visa Business Debit_p_value',
        'Same_p_value', 
    ]

    data = data.drop(columns=columns_to_drop)

    percentage_columns_mapping = {
        'Mastercard Debit': 'Mastercard Debit Percentage',
        'Visa Debit': 'Visa Debit Percentage',
        'Mastercard Credit': 'Mastercard Credit Percentage',
        'Visa Credit': 'Visa Credit Percentage',
        'Mastercard Business Debit': 'Mastercard Business Debit Percentage',
        'Visa Business Debit': 'Visa Business Debit Percentage'
    }

    data = update_payment_card_columns(
        data,
        fixed_charge_column='Fixed Charge (p)',
        avg_transaction_amount_column='Average Transaction Amount',
        percentage_columns=percentage_columns_mapping
    )


    col = ["MCC Code",'Mastercard Debit Percentage',
    'Mastercard Credit Percentage', 'Mastercard Business Debit Percentage',
    'Visa Debit Percentage', 'Visa Credit Percentage',
    'Visa Business Debit Percentage', 'Fixed Charge (p)' ]

    data = data.drop(columns=col)

    data, unique_providers_count, provider_frequency = process_current_providers(data)
    print(f"Total number of unique providers: {unique_providers_count}")
    print(provider_frequency)

    data = group_columns_into_two_categories(data)

    data = data.drop(columns=['Current Provider'])

    data['Transaction Fees per Unit Turnover'] = data['Total Annual Transaction Fees'] / data['Annual Card Turnover']

    column_name_list = ['Is Registered', 'Accepts Card', 'MCC Category group', 'Current Provider Grouped']
    data = encode_columns(data, column_name_list)
    data = data.drop(columns=['MCC Category'])

    plot_spearman_correlation(data)

    # Remove the highly correlated feature 
    data = data.drop(columns=['Current Provider Grouped', 'Mastercard Debit',
     'Mastercard Credit', 'Mastercard Business Debit',
     'Total Annual Transaction Fees', 'Visa Debit'])

    plot_spearman_correlation(data)

    plot_histogram(
        data=data, 
        column='Transaction Fees per Unit Turnover', 
        bins=200, 
        figsize=(10, 6), 
        title="Histogram of Transaction Fees per Unit Turnover", 
        xlabel="Transaction Fees per Unit Turnover", 
        ylabel="Frequency", 
        xlim=(0, 0.02)
    )

    plot_histogram_with_quartiles(
        data=data, 
        column='Transaction Fees per Unit Turnover', 
        xlim=(0, 0.02), 
        title='Histogram of Transaction per Unit Turnover (Scaled)', 
        xlabel='Scaled Transaction per Unit Turnover', 
        ylabel='Frequency'
    )

    column_name_list = [
        'Transaction Fees per Unit Turnover',
        'Average Transaction Amount', 
        'Visa Credit', 
        'Annual Card Turnover',
        'Visa Business Debit']

    data = categorize_pricing(data, 'Transaction Fees per Unit Turnover')

    data = scale_columns(data, column_name_list)

    # Drop the specified columns
    data = data.drop(columns=['Average Transaction Amount',
                'Annual Card Turnover', 
                'Transaction Fees per Unit Turnover',
            'Visa Credit', 
            'Visa Business Debit'])

    print(data.head())

    return data

file_path = 'data.csv'

data = pd.read_csv(file_path)
data = data_preprocessing(data)
data.to_csv('updated_transaction_data_withlabels4.csv', index=False)

print("CSV file has been saved successfully!")