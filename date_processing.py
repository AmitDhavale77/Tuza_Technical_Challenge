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


# List of columns to drop
columns_to_drop = [
    'Visa Debit', 'Mastercard Credit', 'Visa Credit',
    'Mastercard Business Debit', 'Visa Business Debit',
    'Mastercard Debit_p_value', 'Visa Debit_p_value',
    'Mastercard Credit_p_value', 'Visa Credit_p_value',
    'Mastercard Business Debit_p_value', 'Visa Business Debit_p_value',
    'Same_p_value', 'Mastercard Debit Percentage',
    'Mastercard Credit Percentage', 'Mastercard Business Debit Percentage',
    'Visa Debit Percentage', 'Visa Credit Percentage',
    'Visa Business Debit Percentage', 'Fixed Charge (p)',
]

# Drop the columns from the DataFrame
data = data.drop(columns=columns_to_drop)

# Verify the columns have been dropped
print(data.columns)

print(data.iloc[0:5])

col = ["MCC Code"]
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

mcc_frequency = data['MCC Category'].value_counts()
print(mcc_frequency)

data.to_csv('data_processing1.csv', index=False)

file_path = 'data_processing1.csv'

# Read the CSV file into a DataFrame
data1 = pd.read_csv(file_path)


# Perform one-hot encoding for the 'MCC Category' column
mcc_category_encoded = pd.get_dummies(data1['MCC Category'], prefix='MCC_Category')

# Merge the encoded columns back into the original dataframe
data1 = pd.concat([data1, mcc_category_encoded], axis=1)

# Check the updated dataframe
print(data1.head())
print(data1["Current Provider Grouped"].value_counts())

# print(data1["Current Provider Grouped"]==None)

data1 = data1.drop(columns=['MCC Category'])

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
data1 = data1.drop(columns=['Mastercard Debit'])



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
data1 = data1.drop(columns=['Annual Card Turnover', 'Current Provider'])


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

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

# Initialize the RobustScaler
scaler = MinMaxScaler()

# Reshape the data to a 2D array (required for scikit-learn's scaler)
transaction_values = data1['Transaction Fees per Unit Turnover'].values.reshape(-1, 1)

# Fit and transform the data using RobustScaler
data1['Transaction Fees per Unit Turnover_Scaled'] = scaler.fit_transform(transaction_values)

# Check the result
print(data1[['Transaction Fees per Unit Turnover', 'Transaction Fees per Unit Turnover_Scaled']].head())

print(data1[['Transaction Fees per Unit Turnover', 'Total Annual Transaction Fees']])

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(data['Average Transaction Amount Scaled'], bins=200, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of Transaction Fees per Unit Turnover", fontsize=14)
plt.xlabel("Transaction Fees per Unit Turnover", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xlim(0, 0.02)
# plt.xlim(0, 10000)
plt.tight_layout()
plt.show()

print(data.head())

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
Q1 = data1['Transaction Fees per Unit Turnover'].quantile(0.25)
Q2 = data1['Transaction Fees per Unit Turnover'].quantile(0.50)  # Median
Q3 = data1['Transaction Fees per Unit Turnover'].quantile(0.75)

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
Q1 = data1['Transaction Fees per Unit Turnover'].quantile(0.25)
Q2 = data1['Transaction Fees per Unit Turnover'].quantile(0.50)  # Median
Q3 = data1['Transaction Fees per Unit Turnover'].quantile(0.75)

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
data1[['Average Transaction Amount Scaled', 'Total Annual Transaction Fees Scaled']] = scaler.fit_transform(
    data1[['Average Transaction Amount', 'Total Annual Transaction Fees']]
)

# Check the scaled data
print(data1[['Average Transaction Amount Scaled', 'Total Annual Transaction Fees Scaled']].head())

# Drop the specified columns
data1 = data1.drop(columns=['Transaction Fees per Unit Turnover', 'Average Transaction Amount', 'Total Annual Transaction Fees'])



print(data1.head())

data1.to_csv('updated_transaction_data_withlabels.csv', index=False)

print("CSV file has been saved successfully!")











# mean = mean_filtered
# std_dev = std_dev_filtered/2
# # Define conditions for updating 'Current pay' column
# conditions = [
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] <= mean - std_dev),  # competitive
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] > mean - std_dev) & 
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] <= mean + std_dev),  # neutral
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] > mean + std_dev)   # non-competitive
# ]

# # Define the corresponding labels
# labels = ['competitive', 'neutral', 'non-competitive']

# # Update the 'Current pay' column based on the conditions
# data1['Current pay'] = np.select(conditions, labels, default='non-competitive')

# # Check the updated 'Current pay' column
# print(data1['Current pay'].value_counts())



# print(data1.columns)

# # Dropping the specified columns
# data1 = data1.drop(columns=['Transaction_per_Unit_Turnover'])
# # data1 = data1.drop(columns=['Transaction_per_Unit_Turnover', 'Transaction_per_Unit_Turnover_Scaled'])
# # Verify that the columns are dropped
# print(data1.head())




# from sklearn.preprocessing import RobustScaler

# # Initialize the RobustScaler
# robust_scaler = RobustScaler()

# # Apply scaling to 'Average Transaction Amount' and 'Total Annual Transaction Fees'
# data1['Average_Transaction_Amount_Scaled'] = robust_scaler.fit_transform(data1[['Average Transaction Amount']])
# data1['Total_Annual_Transaction_Fees_Scaled'] = robust_scaler.fit_transform(data1[['Total Annual Transaction Fees']])

# # Verify the scaled data
# print(data1[['Average_Transaction_Amount_Scaled', 'Total_Annual_Transaction_Fees_Scaled']].head())


# data1 = data1.drop(columns=['Average Transaction Amount', 'Total Annual Transaction Fees'])

# # Verify that the columns are dropped
# print(data1.head())


# data1.to_csv('data_processing_final.csv', index=False)

# file_path = 'data_processing_final.csv'

# # Read the CSV file into a DataFrame
# data1 = pd.read_csv(file_path)

# import numpy as np

# # Define the conditions based on 'Transaction_per_Unit_Turnover_RobustScaled'
# conditions = [
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] <= -2.370),
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] > -2.370) & (data1['Transaction_per_Unit_Turnover_RobustScaled'] <= 4.87),
#     (data1['Transaction_per_Unit_Turnover_RobustScaled'] > 4.87)
# ]

# # Define the corresponding values for each condition
# values = ['competitive', 'neutral', 'non-competitive']

# # Apply the conditions and assign the results to the new column 'Current pay'
# data1['Current pay'] = np.select(conditions, values, default='Unknown')

# # Verify the new column
# print(data1[['Transaction_per_Unit_Turnover_RobustScaled', 'Current pay']].head())

# frequency = data1['Current pay'].value_counts()

# # Print the frequency count
# print(frequency)

# print(data1['Transaction_per_Unit_Turnover_RobustScaled'].max())




# from sklearn.preprocessing import MinMaxScaler

# # Reshape the column to 2D for the scaler
# scaler = MinMaxScaler()

# # Scale the values between 0 and 1
# data1['Transaction_per_Unit_Turnover_Scaled'] = scaler.fit_transform(data1[['Transaction_per_Unit_Turnover']])

# # Check the updated data
# print(data1[['Transaction_per_Unit_Turnover', 'Transaction_per_Unit_Turnover_Scaled']].head())

# min_value = data1['Transaction_per_Unit_Turnover_Scaled'].max()

# sorted_list = data1['Transaction_per_Unit_Turnover_Scaled'].sort_values(ascending=True).tolist()


# print(len(sorted_list))
# import matplotlib.pyplot as plt

# # Plot the histogram for the scaled values
# plt.figure(figsize=(10, 6))
# plt.hist(data1['Transaction_per_Unit_Turnover_Scaled'], bins=1000, edgecolor='black')
# plt.title('Histogram of Scaled Transaction per Unit Turnover')
# plt.xlabel('Scaled Transaction per Unit Turnover')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# # Define the bin width
# bin_width = 5.0e-5

# # Get the minimum and maximum values from the data
# min_value = data1['Transaction_per_Unit_Turnover'].min()
# max_value = data1['Transaction_per_Unit_Turnover'].max()

# # Create the bins based on the bin width
# bins = np.arange(min_value, max_value + bin_width, bin_width)

# # Plot the histogram with the specified bins
# counts, bin_edges, _ = plt.hist(data1['Transaction_per_Unit_Turnover'], bins=bins, color='skyblue', edgecolor='black')

# # Find the index of the largest bin
# largest_bin_index = np.argmax(counts)

# # Get the range of the largest bin
# bin_range_start = bin_edges[largest_bin_index]
# bin_range_end = bin_edges[largest_bin_index + 1]

# # Print the range of the largest bin
# print(f"The range of values in the largest bin is: ({bin_range_start}, {bin_range_end})")

# # Show the plot
# plt.show()

# # Calculate the mean of the 'Transaction_per_Unit_Turnover' column
# mean_value = data1['Transaction_per_Unit_Turnover'].mean()

# # Calculate the standard deviation of the 'Transaction_per_Unit_Turnover' column
# std_dev = data1['Transaction_per_Unit_Turnover'].std()

# # Print the results
# print(f"The mean of 'Transaction_per_Unit_Turnover' is: {mean_value}")
# print(f"The standard deviation of 'Transaction_per_Unit_Turnover' is: {std_dev}")


# data1.to_csv('data_processing2.csv', index=False)

# file_path = 'data_processing2.csv'

# # Read the CSV file into a DataFrame
# data1 = pd.read_csv(file_path)





# # from sklearn.preprocessing import LabelBinarizer

# # # Initialize LabelBinarizer
# # lb_accepts_card = LabelBinarizer()
# # lb_current_provider = LabelBinarizer()
# # lb_mcc = LabelBinarizer()

# # # Apply LabelBinarizer to 'Accepts Card', 'Current Provider Grouped', and 'MCC Category'
# # accepts_card_encoded = lb_accepts_card.fit_transform(data['Accepts Card'])
# # current_provider_encoded = lb_current_provider.fit_transform(data['Current Provider Grouped'])
# # mcc_encoded = lb_mcc.fit_transform(data["MCC Category"])


# # print(mcc_encoded.shape)
# # # Convert the encoded arrays into DataFrames for easier integration with original data
# # accepts_card_df = pd.DataFrame(accepts_card_encoded, columns=[f'Accepts Card_{cls}' for cls in lb_accepts_card.classes_[:1]])
# # current_provider_df = pd.DataFrame(current_provider_encoded, columns=[f'Current Provider_{cls}' for cls in lb_current_provider.classes_[:1]])
# # mcc_df = pd.DataFrame(mcc_encoded, columns=[f'MCC_{cls}' for cls in lb_mcc.classes_])

# # print(lb_accepts_card.classes_)
# # # Concatenate the original data with the new one-hot encoded columns
# # data = pd.concat([data, accepts_card_df, current_provider_df, mcc_df], axis=1)
# # # Check the new DataFrame with One-Hot Encoded columns
# # print(data.head())

# # print(data.columns)

# # data1 = data.copy()
# # data['Is Registered'] = data['Is Registered'].map({'Yes': 0, 'No': 1})

# # # Check the transformation
# # print(data['Is Registered'].head())

# # data = data.drop(['Is Registered', 'Accepts Card', 'Mastercard Debit', 'MCC Category', 'Current Provider Grouped'], axis=1)

# # # Check the updated DataFrame
# # print(data.head())



# data = data.drop(['Annual Card Turnover'], axis=1)


# # Plot histogram for 'Total Annual Transaction Fees'
# plt.figure(figsize=(12, 10))
# counts, bin_edges, patches = plt.hist(data1['Transaction_per_Unit_Turnover'], bins=500, color='skyblue', edgecolor='black')
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# for center in bin_centers[:20]:  # Limiting to the first 20 bin centers for better visualization
#     plt.axvline(x=center, color='red', linestyle='--', linewidth=1)

# plt.title('Histogram of Total Annual Transaction Fees')
# plt.xlabel('Total Annual Transaction Fees')
# plt.ylabel('Frequency')
# # plt.xlim(0, 2500) # Set the x-axis limit up to 5000
# # plt.xticks(range(0, 6000, 500))
# # plt.xticks(bin_centers[::5], rotation=45) 
# plt.yticks(range(0, 1000, 50))
# plt.grid(True)
# plt.show()

# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# # Get the first 5 bin centers and their corresponding frequencies
# first_5_bin_centers = bin_centers[:5]
# first_5_counts = counts[:5]

# # Print the bin centers and their corresponding counts
# for i in range(5):
#     print(f"Bin center: {first_5_bin_centers[i]:.2f}, Frequency: {first_5_counts[i]}")


# frequency_count = data['Total Annual Transaction Fees'].value_counts().max()

# print(frequency_count)

# max_value = data['Total Annual Transaction Fees'].value_counts().idxmax()
# print(f"The value with the highest frequency is: {max_value}")


# # Calculate the standard deviation (sigma) of the 'Total Annual Transaction Fees'
# sigma = np.std(data['Total Annual Transaction Fees'])

# # Define the range within ± sigma of the max value
# lower_bound = max_value - sigma
# upper_bound = max_value + sigma

# # Filter the values within this range
# filtered_data = data[(data['Total Annual Transaction Fees'] >= lower_bound) & 
#                       (data['Total Annual Transaction Fees'] <= upper_bound)]

# # Show the filtered values
# filtered_data['Total Annual Transaction Fees'].value_counts()

# print(filtered_data['Total Annual Transaction Fees'].value_counts())



# min_value = data['Total Annual Transaction Fees'].min()
# print(min_value)

# min_row = data[data['Total Annual Transaction Fees'] == data['Total Annual Transaction Fees'].min()]
# print(min_row)

# import seaborn as sns


# # Calculate the peak (mode) and standard deviation
# peak_value = data['Total Annual Transaction Fees'].mode()[0]  # The mode of the data
# std_dev = data['Total Annual Transaction Fees'].std()
# std_dev = 2000

# # Calculate +/- 1, 2, and 3 sigma ranges
# sigma_1 = peak_value + std_dev
# sigma_2 = peak_value + 2 * std_dev
# sigma_3 = peak_value + 3 * std_dev

# # Plot the KDE
# plt.figure(figsize=(8, 6))
# sns.kdeplot(data['Total Annual Transaction Fees'], color='blue', shade=True)
# plt.title('Kernel Density Estimate of Total Annual Transaction Fees')
# plt.xlabel('Total Annual Transaction Fees')
# plt.ylabel('Density')

# # Display peak value and sigma ranges
# plt.axvline(peak_value, color='red', linestyle='--', label=f'Peak (Mode) = {peak_value:.2f}')
# plt.axvline(sigma_1, color='green', linestyle='--', label=f'+1 Sigma = {sigma_1:.2f}')
# plt.axvline(sigma_2, color='orange', linestyle='--', label=f'+2 Sigma = {sigma_2:.2f}')
# plt.axvline(sigma_3, color='purple', linestyle='--', label=f'+3 Sigma = {sigma_3:.2f}')
# plt.xlim(0, 10000)
# plt.legend()
# plt.grid(True)
# plt.show()