import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
def load_data(real_path, fake_path):
    try:
        # Load the data
        def get_stats(arr):
            return {
                "mean":np.mean(arr),         # Mean
                "median":np.median(arr),       # Median
                "std":np.std(arr),          # Standard Deviation
                "min":np.min(arr),          # Min
                "max":np.max(arr),          # Max
                "percentile25":np.percentile(arr, 25),  # 25th percentile
                "percentile75":np.percentile(arr, 75)   # 75th percentile
            }
        real_data = pd.read_csv(real_path)
        fake_data = pd.read_csv(fake_path)
        data = {
                "mean":[] ,        # Mean
                "median":[],       # Median
                "std":[],          # Standard Deviation
                "min":[],          # Min
                "max":[],          # Max
                "percentile25":[],  # 25th percentile
                "percentile75":[],   # 75th percentile
                "label":[]
            }
        for _, row in real_data.iterrows():
            data["label"].append(1)
            stats = get_stats(row)
            for key , val in stats.items():
                data[key].append(val)

        for _, row in fake_data.iterrows():
            data["label"].append(0)
            stats = get_stats(row)
            for key , val in stats.items():
                data[key].append(val)

        return pd.DataFrame(data)
    except Exception as e:
        # If there are any errors, print the error message
        print(f"Error loading data: {e}")
        return None
train_data = load_data('train_data_2leg.csv', 'train_data_4leg.csv')

X = train_data.drop('label', axis=1).values
y = train_data['label'].values

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# Model definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.output_layer(x)
        return x

model = NeuralNetwork()

# Initialize weights with He initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_model(num_epochs):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("NaN loss detected at epoch:", epoch)
                return  # Stop training if NaN loss is detected

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the model
train_model(10)

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the model on the test data: %d %%' % (100 * correct / total))

# Prediction example
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    print(predicted)
