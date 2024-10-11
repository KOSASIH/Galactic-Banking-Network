import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

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

def prune_model(model, amount=0.2):
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
      prune.l1_unstructured(module, name='weight', amount=amount)

def evaluate_pruned_model(model, device, loader):
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
  print(f'Pruned model: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)')

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

  # Prune model
  prune_model(model, amount=0.2)

  # Evaluate pruned model
  evaluate_pruned_model(model, device, loader)
