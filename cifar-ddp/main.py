from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))


def main(args):
    use_cuda =  args.cuda and torch.cuda.is_available()
    print(f"use_cuda: {use_cuda}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if WORLD_SIZE > 1:
        dist.init_process_group(backend=dist.Backend.GLOO)
    print("process group initialized")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, 
            shuffle=True, 
            **kwargs
        )
    print("train loader created")
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, download=True, transform=transforms.ToTensor()),
            batch_size=args.test_batch_size, 
            shuffle=False, 
            **kwargs
        )
    print("test loader created")
    model = Net().to(device)

    if dist.is_initialized():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)
    print("distributed model initialized")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("start training")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    torch.save(model.state_dict(),"model.pt")

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Example')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input_file batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='input_file batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1,                    
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status (default: 10)')
                    
if __name__ == '__main__':
    MASTER_PORT = os.environ.get("MASTER_PORT", "{}")
    print(f"MASTER_PORT: {MASTER_PORT}")

    MASTER_ADDR = os.environ.get("MASTER_ADDR", "{}")
    print(f"MASTER_ADDR: {MASTER_ADDR}")
    
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    
    RANK = int(os.environ.get("RANK", 0))
    print(f"RANK: {RANK}")
    
    args = parser.parse_args()
    print(f"args: {args}")
    main(args)
