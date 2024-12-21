import pandas as pd
import numpy as np
import re

pd.set_option('display.max_rows', None)  # No limit on rows
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Auto-detect width
pd.set_option('display.max_colwidth', None)  # No limit on column width


file_path = 'data.csv'

data = pd.read_csv(file_path)

print("data_head", data.head())

print("datatype", data.dtypes)

mcc_frequency = data['MCC Code'].value_counts()

max_mcc = data['MCC Code'].max()
print("max_mcc", max_mcc)

min_mcc = data['MCC Code'].min()
print("min_mcc", min_mcc)

print("mcc_frequency", mcc_frequency)

print("total_rows", len(data))

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

# Function to assign MCC category based on its value
def assign_mcc_category(mcc):
    for category, (low, high) in mcc_categories.items():
        if low <= mcc <= high:
            return category
    return "Unknown"

data['MCC Category'] = data['MCC Code'].apply(assign_mcc_category)

grouped_data = data.groupby('MCC Category').size()
print("MCC Code Grouped by Categories:")
print(grouped_data)

data['MCC Category group'] = data['MCC Category'].apply(
    lambda x: 'Miscellaneous Stores' if x == 'Miscellaneous Stores' else 'Other'
)


numerical_summary = data.describe()
print("Summary of Numerical Features:")
print(numerical_summary)

null_values_count = data.isnull().sum()

# Print the result
print("Count of Null Values in Each Column:")
print(null_values_count)



data['Current Provider'] = data['Current Provider'].fillna('None')
data['Current Provider'] = data['Current Provider'].astype(str)
print("Updated 'Current Provider' Column with Nulls Filled and Converted to String:")
print(data['Current Provider'])

columns_to_check = ['Mastercard Debit', 'Visa Debit', 'Mastercard Credit', 'Visa Credit', 'Mastercard Business Debit', 'Visa Business Debit']


# Function to extract the value attached to 'p' (after the percentage)
def extract_p_value(fee):
    # Use regex to capture the part after the '%' and the 'p' value (e.g., '2p' or '3p')
    match = re.search(r'(\d+)p', fee)
    return match.group(1) if match else None

# Extract the 'p' values for each of the fee columns and create new columns for comparison
for col in columns_to_check:
    data[f'{col}_p_value'] = data[col].apply(extract_p_value)

print(data.columns)
# Check if the extracted 'p' values are the same for each row across the columns
data['Same_p_value'] = data.apply(
    lambda row: len(set(row[f'{col}_p_value'] for col in columns_to_check)) == 1, axis=1
)

# Print rows where 'p' values are not the same across the columns
inconsistent_rows = data[data['Same_p_value'] == False]
print("Rows with inconsistent 'p' values:")
print(inconsistent_rows) #[] implies that fixed charge for each payment is same

# Extract the percentage and fixed parts of fees for each type
# Extract percentages and handle non-matches
data['Mastercard Debit Percentage'] = data['Mastercard Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
data['Mastercard Credit Percentage'] = data['Mastercard Credit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
data['Mastercard Business Debit Percentage'] = data['Mastercard Business Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
data['Visa Debit Percentage'] = data['Visa Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
data['Visa Credit Percentage'] = data['Visa Credit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
data['Visa Business Debit Percentage'] = data['Visa Business Debit'].str.extract(r'(\d+\.\d+)%').astype(float).fillna(1) / 100
data['Fixed Charge (p)'] = data['Mastercard Debit'].str.extract(r'(\d+)p').astype(float).fillna(1)


# Calculate Total annual Transaction fees (excluding the fixed part initially)
data['Total Annual Transaction Fees'] = (
    data['Annual Card Turnover'] * 0.4 * 0.9 * data['Mastercard Debit Percentage'] +
    data['Annual Card Turnover'] * 0.4 * 0.08 * data['Mastercard Credit Percentage'] +
    data['Annual Card Turnover'] * 0.4 * 0.02 * data['Mastercard Business Debit Percentage'] +
    data['Annual Card Turnover'] * 0.6 * 0.9 * data['Visa Debit Percentage'] +
    data['Annual Card Turnover'] * 0.6 * 0.08 * data['Visa Credit Percentage'] +
    data['Annual Card Turnover'] * 0.6 * 0.02 * data['Visa Business Debit Percentage']     
)


non_matching_rows = data[data['Mastercard Debit'].str.contains(r'\d+\.\d+%', na=False) == False]
print("Rows where 'Mastercard Debit' does not match the expected pattern:")
print(non_matching_rows['Mastercard Debit'])

print(data['Mastercard Debit Percentage'][:5])

# Incorporate the fixed part of the fees
data['Total Annual Transaction Fees'] += (
    data['Annual Card Turnover'] / data['Average Transaction Amount']
) * (data['Fixed Charge (p)'] / 100)

# Display the updated DataFrame with the new feature
print(data[['Annual Card Turnover', 'Average Transaction Amount', 'Mastercard Debit', 'Mastercard Credit', 
            'Mastercard Business Debit', 'Total Annual Transaction Fees']])

print(data['Annual Card Turnover'][990])


print(data.columns)

print(data.head())

columns_to_drop = [
    'Mastercard Debit_p_value', 'Visa Debit_p_value',
    'Mastercard Credit_p_value', 'Visa Credit_p_value',
    'Mastercard Business Debit_p_value', 'Visa Business Debit_p_value',
    'Same_p_value', 
]

data = data.drop(columns=columns_to_drop)

# Define the weights for Payment Network and Card Type
w_payment_network_mastercard = 0.4  # 40% of card payments are from Mastercard
w_payment_network_visa = 0.6  # 60% of card payments are from Visa

w_card_type_debit = 0.9  # 90% of card payments are made using Debit cards
w_card_type_credit = 0.08  # 8% are made using Credit cards
w_card_type_business_debit = 0.02  # 2% are made using Business Debit cards



def calculate_new_columns(data, percentage_column, fixed_charge_column):
    # Calculate the individual columns for each card type and payment network
    mastercard_debit = w_payment_network_mastercard * w_card_type_debit * data[percentage_column] + (data[fixed_charge_column] / (100 * data['Average Transaction Amount']))
    visa_debit = w_payment_network_visa * w_card_type_debit * data[percentage_column] + (data[fixed_charge_column] / (100 * data['Average Transaction Amount']))
    mastercard_credit = w_payment_network_mastercard * w_card_type_credit * data[percentage_column] + (data[fixed_charge_column] / (100 * data['Average Transaction Amount']))
    visa_credit = w_payment_network_visa * w_card_type_credit * data[percentage_column] + (data[fixed_charge_column] / (100 * data['Average Transaction Amount']))
    mastercard_business_debit = w_payment_network_mastercard * w_card_type_business_debit * data[percentage_column] + (data[fixed_charge_column] / (100 * data['Average Transaction Amount']))
    visa_business_debit = w_payment_network_visa * w_card_type_business_debit * data[percentage_column] + (data[fixed_charge_column] / (100 * data['Average Transaction Amount']))

    # Return a dictionary with the calculated values for each combination
    return {
        'Mastercard Debit': mastercard_debit,
        'Visa Debit': visa_debit,
        'Mastercard Credit': mastercard_credit,
        'Visa Credit': visa_credit,
        'Mastercard Business Debit': mastercard_business_debit,
        'Visa Business Debit': visa_business_debit
    }

# Columns to apply
columns_to_apply = [
    'Mastercard Debit', 'Visa Debit', 'Mastercard Credit', 'Visa Credit',
    'Mastercard Business Debit', 'Visa Business Debit'
]

percentage_columns = {
    'Mastercard Debit': 'Mastercard Debit Percentage',
    'Visa Debit': 'Visa Debit Percentage',
    'Mastercard Credit': 'Mastercard Credit Percentage',
    'Visa Credit': 'Visa Credit Percentage',
    'Mastercard Business Debit': 'Mastercard Business Debit Percentage',
    'Visa Business Debit': 'Visa Business Debit Percentage'
}

# Apply the formula to each column
for column in columns_to_apply:
    data[column] = calculate_new_columns(
        data, percentage_columns[column], 'Fixed Charge (p)'
    )[column]

print(data.head())


# data['Mastercard Debit'] = (
#     data['Average Transaction Amount']*0.4*0.9*data['Mastercard Debit Percentage']
# ) + (data['Fixed Charge (p)'] / 100)


# Verify the columns have been dropped
print(data.columns)

print(data.iloc[0:5])

col = ["MCC Code",'Mastercard Debit Percentage',
'Mastercard Credit Percentage', 'Mastercard Business Debit Percentage',
'Visa Debit Percentage', 'Visa Credit Percentage',
'Visa Business Debit Percentage', 'Fixed Charge (p)' ]

data = data.drop(columns=col)

print(data.columns)
print(data.dtypes)

print(data["MCC Category"])

# data_encoded = pd.get_dummies(data, columns=['Accepts Card', 'Current Provider'])

print(len(data.columns))

data1 = data

#See number if unique Current Providers
unique_providers_count = data['Current Provider'].nunique()

# Print the result
print(f"Total number of unique providers: {unique_providers_count}")

#Make everything to small case as the data mix alphabets
data['Current Provider'] = data['Current Provider'].str.lower()
data1['Current Provider'] = data['Current Provider'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
mcc_frequency = data['Current Provider'].value_counts()
print(mcc_frequency)
unique_providers_count = data1['Current Provider'].nunique()

# Print the result
print(f"Total number of unique providers: {unique_providers_count}")


# Set display options
# Plot the histogram
import matplotlib.pyplot as plt
# Get the frequency of each provider
mcc_frequency = data['Current Provider'].value_counts()

 
# Group 'Current Provider' into two categories: 'None' and 'Other' to make the dataset balanced
data['Current Provider Grouped'] = data['Current Provider'].apply(
    lambda x: 'None' if x == 'none' else 'Other'
)

# Check the first few rows of the new column
print(data[['Current Provider', 'Current Provider Grouped']].head())
print(data1["Current Provider Grouped"].value_counts())


data = data.drop(columns=['Current Provider'])

print(len(data["Current Provider Grouped"]))

print(data.columns)

numerical_summary = data.describe()
print("Summary of Numerical Features:")
print(numerical_summary)

print(data.head())

mcc_frequency = data['MCC Category group'].value_counts()
print(mcc_frequency)

data.to_csv('data_processing1_trainv2.csv', index=False)

file_path = 'data_processing1_trainv2.csv'

# Read the CSV file into a DataFrame
data1 = pd.read_csv(file_path)

print(data1.head())

# Perform one-hot encoding for the 'MCC Category' column
mcc_category_encoded = pd.get_dummies(data1['MCC Category group'], prefix='encoded_')

# Merge the encoded columns back into the original dataframe
data1 = pd.concat([data1, mcc_category_encoded], axis=1)

# Check the updated dataframe
print(data1.head())
print(data1["Current Provider Grouped"].value_counts())

# print(data1["Current Provider Grouped"]==None)

data1 = data1.drop(columns=['MCC Category', 'MCC Category group', 'encoded__Other'])

# mcc_frequency = data['MCC_Category_Agricultural Services'].value_counts()
# print(mcc_frequency)

print(data1.head())

data1['Transaction Fees per Unit Turnover'] = data1['Total Annual Transaction Fees'] / data1['Annual Card Turnover']

# Display the first few rows to check the result
print(data1[['Annual Card Turnover', 'Average Transaction Amount', 'Transaction Fees per Unit Turnover']].head())

numerical_summary = data1.describe()
print("Summary of Numerical Features:")
print(numerical_summary)

data1['Is Registered'] = data1['Is Registered'].map({'Yes': 0, 'No': 1})

# mcc_frequency = data['Is Registered'].value_counts()
# print(mcc_frequency)




from sklearn.preprocessing import LabelEncoder

# Create a label encoder
label_encoder = LabelEncoder()

# Encode 'Registered' and 'Accepts Card' columns
data1['Registered'] = label_encoder.fit_transform(data1['Is Registered'])
data1['Accepts Card'] = label_encoder.fit_transform(data1['Accepts Card'])

# Display the updated data
print(data1[['Registered', 'Accepts Card']].head())

print(data1.head())

data1 = data1.drop(columns=['Registered'])

#data1['Current Provider'] = label_encoder.fit_transform(data1['Current Provider Grouped'])
data1['Current Provider'] = data1['Current Provider Grouped'].apply(
    lambda x: 0 if pd.isna(x) else 1)

data1 = data1.drop(columns=['Current Provider Grouped'])


data1.columns = [col.replace('Category_', '') for col in data1.columns]
print(data1.head())

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Calculate the Spearman correlation matrix
spearman_corr = data1.corr(method='spearman')

# Plot the Spearman correlation matrix using a heatmap
plt.figure(figsize=(13, 9))

sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 8})

plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)   # Rotate y-axis labels if needed

plt.title('Spearman Correlation Matrix')

plt.tight_layout() 
plt.show()


# Remove the highly correlated feature 
data1 = data1.drop(columns=['Current Provider'])


# Again observe the correlation matrix
spearman_corr = data1.corr(method='spearman')

# Plot the Spearman correlation matrix using a heatmap
plt.figure(figsize=(13, 9))

sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 8})

plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)   # Rotate y-axis labels if needed

plt.title('Spearman Correlation Matrix')

plt.tight_layout() 
plt.show()

data1 = data1.drop(columns=['Mastercard Debit', 'Mastercard Credit', 'Mastercard Business Debit'])

spearman_corr = data1.corr(method='spearman')

# Plot the Spearman correlation matrix using a heatmap
plt.figure(figsize=(13, 9))

sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 8})

plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)   # Rotate y-axis labels if needed

plt.title('Spearman Correlation Matrix')

plt.tight_layout() 
plt.show()


data1 = data1.drop(columns=['Total Annual Transaction Fees', 'Visa Debit'])

spearman_corr = data1.corr(method='spearman')

# Plot the Spearman correlation matrix using a heatmap
plt.figure(figsize=(13, 9))

sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 8})

plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)   # Rotate y-axis labels if needed

plt.title('Spearman Correlation Matrix')

plt.tight_layout() 
plt.show()

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

# Initialize the RobustScaler
scaler = MinMaxScaler()

# Reshape the data to a 2D array (required for scikit-learn's scaler)
transaction_values = data1['Transaction Fees per Unit Turnover'].values.reshape(-1, 1)

# Fit and transform the data using RobustScaler
data1['Transaction Fees per Unit Turnover_Scaled'] = scaler.fit_transform(transaction_values)

# Check the result
print(data1[['Transaction Fees per Unit Turnover', 'Transaction Fees per Unit Turnover_Scaled']].head())


print(data1.head())
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(data1['Transaction Fees per Unit Turnover'], bins=200, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Transaction Fees per Unit Turnover", fontsize=14)
plt.xlabel("Transaction Fees per Unit Turnover", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(0, 0.02)
# plt.xlim(0, 10000)
plt.tight_layout()
plt.show()

print(data1.head())

# Plot the histogram with a logarithmic x-axis
plt.figure(figsize=(10, 6))
plt.hist(data1['Transaction Fees per Unit Turnover'], bins=200, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Transaction Fees per Unit Turnover (Log Scale on X-Axis)", fontsize=14)
plt.xlabel("Transaction Fees per Unit Turnover (Log Scale)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xscale('log')  # Set x-axis to log scale
plt.yscale('linear')  # Keep y-axis linear
plt.tight_layout()
plt.show()

mean = data1['Transaction Fees per Unit Turnover'].mean()
std = data1['Transaction Fees per Unit Turnover'].std()

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(data1['Transaction Fees per Unit Turnover'], bins=200, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Transaction Fees per Unit Turnover", fontsize=14)
plt.xlabel("Transaction Fees per Unit Turnover", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Add mean line
plt.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean:.4f}')

# Add ±1σ lines
plt.axvline(mean - std/2, color='green', linestyle='--', linewidth=1.5, label=f'Mean - 1σ: {mean - std/2:.4f}')
plt.axvline(mean + std/2, color='green', linestyle='--', linewidth=1.5, label=f'Mean + 1σ: {mean + std/2:.4f}')

# Optional grid and layout adjustments
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(0, 0.02)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# Calculate quartiles (Q1, Q2, Q3)
Q1 = data1['Transaction Fees per Unit Turnover'].quantile(0.33)
Q2 = data1['Transaction Fees per Unit Turnover'].quantile(0.50)  # Median
Q3 = data1['Transaction Fees per Unit Turnover'].quantile(0.66)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(data1['Transaction Fees per Unit Turnover'], bins=200, edgecolor='black', alpha=0.7)

# Add lines for Q1, Q2, Q3
plt.axvline(Q1, color='red', linestyle='--', label=f'Q1 ({Q1:.4f})')
plt.axvline(Q2, color='green', linestyle='--', label=f'Q2 (Median) ({Q2:.4f})')
plt.axvline(Q3, color='blue', linestyle='--', label=f'Q3 ({Q3:.4f})')

# Title and labels
plt.title('Histogram of Transaction per Unit Turnover (Scaled)')
plt.xlabel('Scaled Transaction per Unit Turnover')
plt.ylabel('Frequency')
plt.xlim(0, 0.02)
# Show legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

# Calculate quartiles (Q1, Q2, Q3)
Q1 = data1['Transaction Fees per Unit Turnover'].quantile(0.33)
Q2 = data1['Transaction Fees per Unit Turnover'].quantile(0.50)  # Median
Q3 = data1['Transaction Fees per Unit Turnover'].quantile(0.66)

# Define conditions based on quartiles
conditions = [
    (data1['Transaction Fees per Unit Turnover'] <= Q1),  # competitive
    (data1['Transaction Fees per Unit Turnover'] > Q1) & 
    (data1['Transaction Fees per Unit Turnover'] <= Q3),  # neutral
    (data1['Transaction Fees per Unit Turnover'] > Q3)   # non-competitive
]

# Define corresponding labels
labels = ['Competitive', 'Neutral', 'Non-Competitive']

# Apply conditions and assign labels to the 'Current pay' column
data1['Current pricing'] = np.select(conditions, labels)

# Check the frequency of each label
print(data1['Current pricing'].value_counts())


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMax scaling to the selected columns
data1[['Average Transaction Amount Scaled', 
       'Total Annual Transaction Fees Scaled', 
       'Visa Debit Scaled', 
       'Visa Credit Scaled', 
       'Annual Card Turnover Scaled',
       'Visa Business Debit Scaled']] = scaler.fit_transform(
    data1[['Average Transaction Amount', 
           'Total Annual Transaction Fees', 
           'Visa Debit', 
           'Visa Credit', 
           'Annual Card Turnover',
           'Visa Business Debit']]
)

# Check the scaled data
print(data1[['Average Transaction Amount Scaled', 'Total Annual Transaction Fees Scaled']].head())

# Drop the specified columns
data1 = data1.drop(columns=['Average Transaction Amount',
            'Annual Card Turnover', 
            'Transaction Fees per Unit Turnover',
           'Total Annual Transaction Fees', 
           'Visa Debit', 
           'Visa Credit', 
           'Visa Business Debit'])

print(data1.head())

data1.to_csv('updated_transaction_data_withlabels3.csv', index=False)

print("CSV file has been saved successfully!")

data2 = pd.read_csv('updated_transaction_data_withlabels1.csv')

print(data2.head())
import matplotlib.pyplot as plt

# Assuming `data1` has a column with the class labels and the feature
class_column = 'Current pricing'  # Replace with the actual class label column name
feature_column = 'Transaction Fees per Unit Turnover_Scaled'  # Replace with the actual feature column name

# Get unique class labels
class_labels = data2[class_column].unique()

# Define color palette for classes
colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown']  # Add more if needed

plt.figure(figsize=(10, 6))

# Plot histograms for each class
for i, label in enumerate(class_labels):
    class_data = data2[data2[class_column] == label]    
    plt.hist(
        class_data[feature_column],
        bins=200,
        alpha=0.5,  # Transparency
        color=colors[i % len(colors)],  # Cycle through colors
        edgecolor='black',
        label=f"Class {label}"
    )

plt.title("Histogram of Transaction Fees per Unit Turnover by Class", fontsize=14)
plt.xlabel("Transaction Fees per Unit Turnover", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(0, 0.5)
plt.legend(title="Class Labels", fontsize=10)
plt.tight_layout()
plt.show()