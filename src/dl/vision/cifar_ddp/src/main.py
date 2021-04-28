import os
import argparse
import cProfile
import pstats

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import utils
import resnet


def train(args):
    CUDA = args.cuda and torch.cuda.is_available()
    print(f"CUDA: {CUDA}")
    
    DEVICE = torch.device("cuda" if CUDA else "cpu")
    print(f"DEVICE: {DEVICE}")
    
    SEED = args.seed
    torch.manual_seed(SEED)
    print(f"SEED: {SEED}")

    if WORLD_SIZE > 1:
        print("> Initializing process group")
        dist.init_process_group(backend=args.backend)
    
    print("> Reading dataset from HDFS")
    utils.from_hdfs('data/cifar10/cifar-10-python.tar.gz', './data/cifar-10-python.tar.gz')
    
    print("> Creating train_loader")
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    if dist.is_initialized():
        train_sampler = DistributedSampler(trainset, shuffle=True)
    else:
        train_sampler = None
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None, 
            **kwargs
        )
    print("> Creating test_loader")
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    if dist.is_initialized():
        test_sampler = DistributedSampler(testset, shuffle=False)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.test_batch_size,
            sampler=test_sampler,
            shuffle=False, 
            **kwargs
        )
    
    print("> Initializing model")
    model = resnet.get_resnet_model(args.model, version=args.version).to(DEVICE)

    if dist.is_initialized():
        print("> Initializing distributed model")
        Distributor = nn.parallel.DistributedDataParallel if CUDA \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)
        
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=.9, 
                          dampening=0, 
                          weight_decay=1e-4, 
                          nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    print("> Training: ")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                niter = epoch * len(train_loader) + batch_idx
        
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader.dataset)
        scheduler.step(test_loss)
        
        accuracy = float(correct) / len(test_loader.dataset)
        print('\naccuracy={:.4f}\n'.format(accuracy))
        
        if accuracy >= 0.8:
            break
    
    if RANK == 0:
        print("> Saving model")
        torch.save(model.state_dict(),"./model.pt")
        
        print("> Writing model to HDFS")
        utils.to_hdfs('./model.pt', 'models/pytorch-cifar10.pt')


def predict(args):
    CUDA = args.cuda and torch.cuda.is_available()
    print(f"CUDA: {CUDA}")
    
    DEVICE = torch.device("cuda" if CUDA else "cpu")
    print(f"DEVICE: {DEVICE}")
    
    print("> Reading data from HDFS")
    utils.from_hdfs('data/cifar10/cifar10-test.npy', './data/cifar10-test.npy')
    data = np.load('./data/cifar10-test.npy')
    data = torch.from_numpy(data).to(DEVICE)
    
    print("> Initializing model")
    model = Net().to(DEVICE)
    
    print("> Loading model from HDFS")
    utils.from_hdfs('models/pytorch-cifar10.pt', './model.pt')
    model.load_state_dict(torch.load('./model.pt', map_location=DEVICE))
    model.eval()
    
    print("> Making predictions")
    with torch.no_grad():
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
    
    print("> Saving predictions to HDFS") 
    np.save('./prediction.npy', pred.cpu().numpy())
    utils.to_hdfs('./prediction.npy', 'data/cifar10/cifar10-prediction.npy')

parser = argparse.ArgumentParser(description='PyTorch Distributed Data-Parallel Cifar10 Benchmark')
parser.add_argument('--profile', action='store_true', default=False, 
                    help='Profile execution (default: False)')
subparsers = parser.add_subparsers()
train_parser = subparsers.add_parser('train')
train_parser.add_argument('--model', type=str, default='resnet20',
                          help='Model (default: resnet20)')
train_parser.add_argument('--version', type=int, default=1,
                          help='Model version (default: 1)')
train_parser.add_argument('--batch-size', type=int, default=128,
                          help='Training batch size (default: 128)')
train_parser.add_argument('--test-batch-size', type=int, default=1000,
                          help='Testing batch size (default: 1000)')
train_parser.add_argument('--epochs', type=int, default=10,
                          help='Number of training epochs (default: 10)')
train_parser.add_argument('--lr', type=float, default=0.02,
                          help='Learning rate (default: 0.02)')
train_parser.add_argument('--cuda', action='store_true', default=False,
                          help='Enable CUDA')
train_parser.add_argument('--backend', type=str, default=dist.Backend.GLOO,
                          choices=[dist.Backend.GLOO, dist.Backend.NCCL],
                          help='Distributed backend')
train_parser.add_argument('--seed', type=int, default=0,                    
                          help='Random seed (default: 0)')
train_parser.add_argument('--log-interval', type=int, default=10,
                          help='Logging interval (default: 10)')
train_parser.set_defaults(func=train)

predict_parser = subparsers.add_parser('predict')
predict_parser.add_argument('--cuda', action='store_true', default=False,
                            help='enable CUDA')
predict_parser.set_defaults(func=predict)


if __name__ == '__main__':
    MASTER_ADDR = os.environ['MASTER_ADDR']
    print(f"MASTER_ADDR: {MASTER_ADDR}")
    
    MASTER_PORT = os.environ['MASTER_PORT']
    print(f"MASTER_PORT: {MASTER_PORT}")
    
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    print(f"WORLD_SIZE: {WORLD_SIZE}")
    
    RANK = int(os.environ['RANK'])
    print(f"RANK: {RANK}")
    
    args = parser.parse_args()
    if args.profile:
        cProfile.run('args.func(args)', filename='profile.txt')
        stats = pstats.Stats('profile.txt')
        stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
    else:
        args.func(args)
