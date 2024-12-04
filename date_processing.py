import pandas as pd
import numpy as np
import re

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

print(data['Mastercard Debit Percentage'][999])

# Incorporate the fixed part of the fees
data['Total Annual Transaction Fees'] += (
    data['Total Annual Transaction Fees'] / data['Average Transaction Amount']
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

col = ["MCC Code"]
data = data.drop(columns=col)

print(data.columns)
print(data.dtypes)

print(data["MCC Category"])



data_encoded = pd.get_dummies(data, columns=['Accepts Card', 'Current Provider'])

print(len(data.columns))

mcc_frequency = data['Current Provider'].value_counts()
print(mcc_frequency)

# Group 'Current Provider' into two categories: 'None' and 'Other'
data['Current Provider Grouped'] = data['Current Provider'].apply(
    lambda x: 'None' if x == 'None' else 'Other'
)

# Check the first few rows of the new column
print(data[['Current Provider', 'Current Provider Grouped']].head())

data = data.drop(columns=['Current Provider'])



from sklearn.preprocessing import LabelBinarizer

# Initialize LabelBinarizer
lb_accepts_card = LabelBinarizer()
lb_current_provider = LabelBinarizer()
lb_mcc = LabelBinarizer()

# Apply LabelBinarizer to 'Accepts Card', 'Current Provider Grouped', and 'MCC Category'
accepts_card_encoded = lb_accepts_card.fit_transform(data['Accepts Card'])
current_provider_encoded = lb_current_provider.fit_transform(data['Current Provider Grouped'])
mcc_encoded = lb_mcc.fit_transform(data["MCC Category"])


print(mcc_encoded.shape)
# Convert the encoded arrays into DataFrames for easier integration with original data
accepts_card_df = pd.DataFrame(accepts_card_encoded, columns=[f'Accepts Card_{cls}' for cls in lb_accepts_card.classes_[:1]])
current_provider_df = pd.DataFrame(current_provider_encoded, columns=[f'Current Provider_{cls}' for cls in lb_current_provider.classes_[:1]])
mcc_df = pd.DataFrame(mcc_encoded, columns=[f'MCC_{cls}' for cls in lb_mcc.classes_])

print(lb_accepts_card.classes_)
# Concatenate the original data with the new one-hot encoded columns
data = pd.concat([data, accepts_card_df, current_provider_df, mcc_df], axis=1)
# Check the new DataFrame with One-Hot Encoded columns
print(data.head())

print(data.columns)

data1 = data.copy()
data['Is Registered'] = data['Is Registered'].map({'Yes': 0, 'No': 1})

# Check the transformation
print(data['Is Registered'].head())

data = data.drop(['Is Registered', 'Accepts Card', 'Mastercard Debit', 'MCC Category', 'Current Provider Grouped'], axis=1)

# Check the updated DataFrame
print(data.head())


import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Calculate the Spearman correlation matrix
spearman_corr = data.corr(method='spearman')

# Plot the Spearman correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.show()

data = data.drop(['Annual Card Turnover'], axis=1)


# Plot histogram for 'Total Annual Transaction Fees'
plt.figure(figsize=(12, 10))
counts, bin_edges, patches = plt.hist(data['Total Annual Transaction Fees'], bins=500, color='skyblue', edgecolor='black')
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

for center in bin_centers[:20]:  # Limiting to the first 20 bin centers for better visualization
    plt.axvline(x=center, color='red', linestyle='--', linewidth=1)

plt.title('Histogram of Total Annual Transaction Fees')
plt.xlabel('Total Annual Transaction Fees')
plt.ylabel('Frequency')
plt.xlim(0, 2500) # Set the x-axis limit up to 5000
# plt.xticks(range(0, 6000, 500))
# plt.xticks(bin_centers[::5], rotation=45) 
plt.yticks(range(0, 1000, 50))
plt.grid(True)
plt.show()

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Get the first 5 bin centers and their corresponding frequencies
first_5_bin_centers = bin_centers[:5]
first_5_counts = counts[:5]

# Print the bin centers and their corresponding counts
for i in range(5):
    print(f"Bin center: {first_5_bin_centers[i]:.2f}, Frequency: {first_5_counts[i]}")


frequency_count = data['Total Annual Transaction Fees'].value_counts().max()

print(frequency_count)

max_value = data['Total Annual Transaction Fees'].value_counts().idxmax()
print(f"The value with the highest frequency is: {max_value}")


# Calculate the standard deviation (sigma) of the 'Total Annual Transaction Fees'
sigma = np.std(data['Total Annual Transaction Fees'])

# Define the range within ± sigma of the max value
lower_bound = max_value - sigma
upper_bound = max_value + sigma

# Filter the values within this range
filtered_data = data[(data['Total Annual Transaction Fees'] >= lower_bound) & 
                      (data['Total Annual Transaction Fees'] <= upper_bound)]

# Show the filtered values
filtered_data['Total Annual Transaction Fees'].value_counts()

print(filtered_data['Total Annual Transaction Fees'].value_counts())



min_value = data['Total Annual Transaction Fees'].min()
print(min_value)

min_row = data[data['Total Annual Transaction Fees'] == data['Total Annual Transaction Fees'].min()]
print(min_row)

import seaborn as sns


# Calculate the peak (mode) and standard deviation
peak_value = data['Total Annual Transaction Fees'].mode()[0]  # The mode of the data
std_dev = data['Total Annual Transaction Fees'].std()
std_dev = 2000

# Calculate +/- 1, 2, and 3 sigma ranges
sigma_1 = peak_value + std_dev
sigma_2 = peak_value + 2 * std_dev
sigma_3 = peak_value + 3 * std_dev

# Plot the KDE
plt.figure(figsize=(8, 6))
sns.kdeplot(data['Total Annual Transaction Fees'], color='blue', shade=True)
plt.title('Kernel Density Estimate of Total Annual Transaction Fees')
plt.xlabel('Total Annual Transaction Fees')
plt.ylabel('Density')

# Display peak value and sigma ranges
plt.axvline(peak_value, color='red', linestyle='--', label=f'Peak (Mode) = {peak_value:.2f}')
plt.axvline(sigma_1, color='green', linestyle='--', label=f'+1 Sigma = {sigma_1:.2f}')
plt.axvline(sigma_2, color='orange', linestyle='--', label=f'+2 Sigma = {sigma_2:.2f}')
plt.axvline(sigma_3, color='purple', linestyle='--', label=f'+3 Sigma = {sigma_3:.2f}')
plt.xlim(0, 10000)
plt.legend()
plt.grid(True)
plt.show()