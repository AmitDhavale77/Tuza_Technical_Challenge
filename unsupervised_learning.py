import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('updated_transaction_data_withlabels.csv')

# Assuming 'Current pay' is the target label column
label_column = 'Current pricing'

# Encoding the labels
label_encoder = LabelEncoder()
data[label_column] = label_encoder.fit_transform(data[label_column])

# Select features (drop the target label column)
X = data.drop(columns=[label_column, 'Total Annual Transaction Fees Scaled'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, data[label_column], test_size=0.2, random_state=42)

# Step 1: Fit the GMM on the entire training set
gmm = GaussianMixture(n_components=3, random_state=42)  # Assuming 3 clusters
gmm.fit(X_train)

# Step 2: Predict the cluster labels for the entire training set
train_cluster_labels = gmm.predict(X_train)

X_train['cluster'] = train_cluster_labels

# Now, for each cluster, filter the data and get the values of 'Transaction Fees per Unit Turnover_Scaled'
cluster_0_values = X_train[X_train['cluster'] == 0]['Transaction Fees per Unit Turnover_Scaled']
cluster_1_values = X_train[X_train['cluster'] == 1]['Transaction Fees per Unit Turnover_Scaled']
cluster_2_values = X_train[X_train['cluster'] == 2]['Transaction Fees per Unit Turnover_Scaled']

# Display the values for each cluster
print("Cluster 0 values of 'Transaction Fees per Unit Turnover_Scaled':")
print(cluster_0_values)

print("\nCluster 1 values of 'Transaction Fees per Unit Turnover_Scaled':")
print(cluster_1_values)

print("\nCluster 2 values of 'Transaction Fees per Unit Turnover_Scaled':")
print(cluster_2_values)

import matplotlib.pyplot as plt

# Plot the histogram for 'Transaction Fees per Unit Turnover_Scaled' for each cluster
plt.figure(figsize=(10, 6))

# Plot for cluster 0
plt.hist(cluster_0_values, bins=30, alpha=0.5, label='Cluster 0', color='blue', edgecolor='black')

# Plot for cluster 1
plt.hist(cluster_1_values, bins=30, alpha=0.5, label='Cluster 1', color='green', edgecolor='black')

# Plot for cluster 2
plt.hist(cluster_2_values, bins=30, alpha=0.5, label='Cluster 2', color='red', edgecolor='black')

# Add title and labels
plt.title('Histogram of Transaction Fees per Unit Turnover_Scaled by Cluster', fontsize=14)
plt.xlabel('Transaction Fees per Unit Turnover_Scaled', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the legend
plt.legend()

# Display grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()











# Step 3: Map the cluster labels to the majority class in each cluster
# Create a DataFrame with cluster labels and true labels
clustered_data = pd.DataFrame({'cluster': train_cluster_labels, 'true_label': y_train})

# For each cluster, assign the majority label
cluster_majority_labels = clustered_data.groupby('cluster')['true_label'].agg(lambda x: x.mode()[0])

cluster_0_labels = clustered_data[clustered_data['cluster'] == 2]['true_label']
label_counts_cluster_0 = cluster_0_labels.value_counts()

# Display the count of each label in cluster 0
print(label_counts_cluster_0)

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
# Use t-SNE to reduce the dimensionality of the feature space for visualization (3D)
# Use PCA to reduce the dimensionality of the feature space for visualization (3D)
pca = PCA(n_components=3)  # Reduce to 3 dimensions
pca = TSNE(n_components=3) 
X_train_pca = pca.fit_transform(X_train)

# Plot the clusters in 3D space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
ax.scatter(X_train_pca[train_cluster_labels == 0, 0], X_train_pca[train_cluster_labels == 0, 1], X_train_pca[train_cluster_labels == 0, 2], 
            s=30, label='Cluster 1', alpha=0.6)
ax.scatter(X_train_pca[train_cluster_labels == 1, 0], X_train_pca[train_cluster_labels == 1, 1], X_train_pca[train_cluster_labels == 1, 2], 
            s=30, label='Cluster 2', alpha=0.6)
ax.scatter(X_train_pca[train_cluster_labels == 2, 0], X_train_pca[train_cluster_labels == 2, 1], X_train_pca[train_cluster_labels == 2, 2], 
            s=30, label='Cluster 3', alpha=0.6)

# Adding labels and title
ax.set_title("GMM Clusters (PCA Reduced to 3D)", fontsize=14)
ax.set_xlabel("PCA Component 1", fontsize=12)
ax.set_ylabel("PCA Component 2", fontsize=12)
ax.set_zlabel("PCA Component 3", fontsize=12)

# Show the legend and grid
ax.legend()
ax.grid(True)

# Show the plot
plt.show()

X_train_with_clusters = np.hstack((X_train, train_cluster_labels.reshape(-1, 1)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
# Step 4: Train a classifier (Random Forest) on the clustered data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_with_clusters, y_train)

# Step 5: Predict cluster labels for the test set
test_cluster_labels = gmm.predict(X_test)

# Add the test cluster labels as a feature to the test data
X_test_with_clusters = np.hstack((X_test, test_cluster_labels.reshape(-1, 1)))

# Step 6: Predict the output labels for the test set
y_pred = rf_classifier.predict(X_test_with_clusters)

# Step 7: Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))