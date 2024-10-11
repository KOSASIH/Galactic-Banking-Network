import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim import Adam, SGD, RMSprop
from torch.optim import AdamW, Nadam, RAdam
from torch.optim import LBFGS, Adagrad
from torch.optim import Adadelta, Adamax

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

def train(model, device, loader, optimizer, criterion, scheduler=None):
  model.train()
  for batch_idx, (data, target) in enumerate(loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()

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
  criterion = nn.CrossEntropyLoss()

  # Adam optimizer
  optimizer = Adam(model.parameters(), lr=0.001)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # SGD optimizer
  optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
  scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # RMSprop optimizer
  optimizer = RMSprop(model.parameters(), lr=0.01, alpha=0.99)
  scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-2, step_size_up=5, step_size_down=5)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # AdamW optimizer
  optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0. 5, patience=5, min_lr=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # Nadam optimizer
  optimizer = Nadam(model.parameters(), lr=0.01)
  scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # RAdam optimizer
  optimizer = RAdam(model.parameters(), lr=0.01)
  scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-2, step_size_up=5, step_size_down=5)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # LBFGS optimizer
  optimizer = LBFGS(model.parameters(), lr=0.01)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # Adagrad optimizer
  optimizer = Adagrad(model.parameters(), lr=0.01)
  scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # Adadelta optimizer
  optimizer = Adadelta(model.parameters(), lr=0.01)
  scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-2, step_size_up=5, step_size_down=5)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)

  # Adamax optimizer
  optimizer = Adamax(model.parameters(), lr=0.01)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
  for epoch in range(10):
    train(model, device, loader, optimizer, criterion, scheduler)
    test(model, device, loader)
