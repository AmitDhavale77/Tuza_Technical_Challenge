import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

# Load the dataset
file_path = 'updated_transaction_data_withlabels.csv'
data = pd.read_csv(file_path)

# Separate the target column and features
label_column = 'Current pay'  
X = data.drop(columns=[label_column])  # Features
y = data[label_column]  # Target labels

# Convert categorical labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into labeled and unlabeled data
n_labeled = int(0.1 * len(y))  # Using only 10% of the data as labeled
y_unlabeled = np.copy(y_encoded)
y_unlabeled[n_labeled:] = -1  # Masking the labels as unlabeled (-1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_unlabeled, test_size=0.2, random_state=42)

# Perform KMeans clustering on the entire data
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train)

# Visualize the clusters
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted")

# Scatter plot for two random features (for simplicity)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train.iloc[:, -1], y=X_train.iloc[:, -2], hue=kmeans.labels_, palette="Set1")
plt.title('K-Means Clustering on the Data (Semi-supervised)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Optionally: Compare clustering with actual labels if available
# Only consider the labeled data for comparison
y_train_labeled = y_train[y_train != -1]
y_pred_labeled = kmeans.labels_[:len(y_train_labeled)]

# Accuracy can be computed by matching clusters with known labels (optional)
accuracy = accuracy_score(y_train_labeled, y_pred_labeled)
print(f'Clustering Accuracy: {accuracy * 100:.2f}%')

# For visualization, we can also map the predicted clusters to actual labels and display it
y_train_pred = kmeans.predict(X_train)

# Display the predicted clusters alongside actual labels
clustered_data = X_train.copy()
clustered_data['Cluster'] = y_train_pred
clustered_data['Actual Label'] = y_train

print(clustered_data.head())