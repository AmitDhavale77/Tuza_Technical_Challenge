import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import seaborn as sns
from itertools import combinations
import math

pd.set_option('display.max_rows', None)  # No limit on rows
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Auto-detect width
pd.set_option('display.max_colwidth', None)  # No limit on column width


file_path = 'updated_transaction_data_withlabels.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

print(data.head())

print(data.columns)

label_column = 'Current pay'  
X = data.drop(columns=[label_column])  # Features
y = data[label_column]  # Target labels (already encoded)

print(X.columns)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

feature1 = X.columns[-1]
feature2 = X.columns[-5]

# Plotting pairwise combinations of features
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data[feature1], y=data[feature2], hue=data['Cluster'], palette='viridis', s=100, alpha=0.7, marker='o')

# Title and labels
plt.title(f'Cluster of {feature1} vs {feature2}')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend(title='Cluster')

# Show the plot
plt.show()