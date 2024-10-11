import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

class GalacticDataset(Dataset):
  def __init__(self, data, labels, transform=None):
    self.data = data
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample = self.data[idx]
    label = self.labels[idx]

    if self.transform:
      sample = self.transform(sample)

    return sample, label

class GalacticNeuralNetwork(nn.Module):
  def __init__(self):
    super(GalacticNeuralNetwork, self).__init__()
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

def train(model, device, loader, optimizer, criterion):
  model.train()
  for batch_idx, (data, target) in enumerate(loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

def test(model, device, loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += criterion(output, target).item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(loader.dataset)
  print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)')

def explain(model, device, data, target):
  ig = IntegratedGradients(model)
  attributions = ig.attribute(data, target=target, n_steps=50)
  return attributions

def visualize(attributions, data):
  _ = viz.visualize_image_attr(attributions, data, method="blended_heatmap", show_colorbar=True, sign="positive")

# Example usage:
if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load dataset
  data = np.random.rand(1000, 784)
  labels = np.random.randint(0, 10, 1000)
  train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

  # Create dataset and data loader
  dataset = GalacticDataset(train_data, train_labels, transform=transforms.Compose([transforms.ToTensor()]))
  loader = DataLoader(dataset, batch_size=64, shuffle=True)

  # Create model and optimizer
  model = GalacticNeuralNetwork()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  # Train model
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion)
    test(model, device, loader)

  # Explain model
  data, target = next(iter(loader))
  attributions = explain(model, device, data, target)
  visualize(attributions, data)
