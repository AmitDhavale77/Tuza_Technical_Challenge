import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = 'updated_transaction_data_withlabels1.csv'
data = pd.read_csv(file_path)

# Define the label column and features
label_column = 'Current pricing'
X = data.drop(columns=[label_column, 'Transaction Fees per Unit Turnover_Scaled'])  # Features
print(X.dtypes)
X['encoded__Miscellaneous Stores'] = X['encoded__Miscellaneous Stores'].astype(int)
print(X.dtypes)

y = data[label_column]  # Target labels

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = X.to_numpy()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Create a DataLoader for batching
batch_size = 32  # You can adjust this depending on your memory capacity and model complexity
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define a deeper neural network
class DeepNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)  # Softmax for multiclass classification
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize the model
input_size = X_train.shape[1]
num_classes = len(np.unique(y))  # Number of classes in the target
model = DeepNN(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
# Training loop with batching
epochs = 1000
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    y_pred_train = torch.argmax(model(X_train_tensor), axis=1)
    y_pred_test = torch.argmax(model(X_test_tensor), axis=1)
    
    train_accuracy = (y_pred_train == y_train_tensor).float().mean().item() * 100
    test_accuracy = (y_pred_test == y_test_tensor).float().mean().item() * 100
    
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

