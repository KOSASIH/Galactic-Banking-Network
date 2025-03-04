import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Node:
  def __init__(self, node_rank, world_size, master_addr, master_port):
    self.node_rank = node_rank
    self.world_size = world_size
    self.master_addr = master_addr
    self.master_port = master_port
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def init_process_group(self):
    dist.init_process_group(backend='nccl', init_method=f'tcp://{self.master_addr}:{self.master_port}',
                            world_size=self.world_size, rank=self.node_rank)

  def create_data_loader(self, data, labels, batch_size):
    dataset = GalacticDataset(data, labels, transform=transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

  def train_model(self, model, loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
      data, target = data.to(self.device), target.to(self.device)
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

  def test_model(self, model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in loader:
        data, target = data.to(self.device), target.to(self.device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)')

  def run(self, model, data, labels, batch_size):
    self.init_process_group()
    loader = self.create_data_loader(data, labels, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
      self.train_model(model, loader, optimizer, criterion)
      self.test_model(model, loader)

  def send_gradients(self, gradients, destination_rank):
    dist.send(tensor=gradients, dst=destination_rank)

  def receive_gradients(self, source_rank):
    gradients = torch.zeros(1)
    dist.recv(tensor=gradients, src=source_rank)
    return gradients

  def aggregate_gradients(self, gradients):
    dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
    gradients /= self.world_size

# Example usage:
if __name__ == '__main__':
  node_rank = 0
  world_size = 4
  master_addr = '127.0.0.1'
  master_port = 12345

  data = np.random.rand(1000, 784)
  labels = np.random.randint(0, 10, 1000)

  node = Node(node_rank, world_size, master_addr, master_port)
  model = GalacticNeuralNetwork()
  node.run(model, data, labels, batch_size=64)
