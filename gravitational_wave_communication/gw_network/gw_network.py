import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.metrics import mean_tweedie_deviance
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import pinball_loss
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import d2_pinball_score
from sklearn.metrics import mean_d2_pinball_score

class GWNetwork(nn.Module):
    def __init__(self):
        super(GWNetwork, self).__init__()
        self.fc1 = nn.Linear(1024, 512)  # Input layer (1024) -> Hidden layer (512)
        self.fc2 = nn.Linear(512, 256)  # Hidden layer (512) -> Hidden layer (256)
        self.fc3 = nn.Linear(256, 128)  # Hidden layer (256) -> Hidden layer (128)
        self.fc4 = nn.Linear(128, 64)  # Hidden layer (128) -> Hidden layer (64)
        self.fc5 = nn.Linear(64, 32)  # Hidden layer (64) -> Hidden layer (32)
        self.fc6 = nn.Linear(32, 16)  # Hidden layer (32) -> Hidden layer (16)
        self.fc7 = nn.Linear(16, 8)  # Hidden layer (16) -> Hidden layer (8)
        self.fc8 = nn.Linear(8, 4)  # Hidden layer (8) -> Hidden layer (4)
        self.fc9 = nn.Linear(4, 2)  # Hidden layer (4) -> Output layer (2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = self.fc9(x)
        return x

class GWData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label = self.labels[idx]
        return {
            'data': torch.tensor(data_point, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_gw_network(data, labels, batch_size=32, epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GWNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = GWData(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in data_loader:
            data_point = batch['data'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            output = model(data_point)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    return model

def evaluate_gw_network(model, data, labels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = GWData(data, labels)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            data_point = batch['data'].to(device)
            output = model(data_point)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    print(f'Accuracy: {accuracy:.3f}')
    return accuracy

# Example usage:
data = np.random.rand(100, 1024)  # Replace with your data
labels = np.random.randint(0, 2, 100)  # Replace with your labels
model = train_gw_network(data, labels)
evaluate_gw_network(model, data, labels)
