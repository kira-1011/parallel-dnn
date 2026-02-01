#!/usr/bin/env python3
"""
Parallel Training Script for CNN on CIFAR-10
Uses PyTorch DistributedDataParallel (DDP) for data-parallel training.

Usage:
    # Using torchrun (recommended):
    torchrun --nproc_per_node=4 parallel_training.py --epochs 20 --batch-size 128

    # Using mp.spawn (alternative):
    python parallel_training.py --epochs 20 --batch-size 128 --world-size 4
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms


# ============================================================================
# Configuration
# ============================================================================
RANDOM_SEED = 42
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ============================================================================
# Model Definition
# ============================================================================
class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================================================
# Data Loading
# ============================================================================
def get_datasets():
    """Load CIFAR-10 datasets with appropriate transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    return train_dataset, test_dataset


def create_distributed_dataloaders(train_dataset, test_dataset, batch_size, 
                                   rank, world_size, num_workers=2):
    """Create DataLoaders with DistributedSampler."""
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, train_sampler


# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # DDP handles AllReduce here
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Aggregate across workers
    if dist.is_initialized():
        metrics = torch.tensor([correct, total, running_loss], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        correct, total, running_loss = metrics.tolist()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return running_loss / len(test_loader) / world_size, 100. * correct / total


# ============================================================================
# Main Training Function
# ============================================================================
def train_worker(rank, world_size, args):
    """Main training function for each worker."""
    # Setup distributed
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    os.environ['USE_LIBUV'] = "0"
    init_method="env://"

    dist.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=world_size)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Create model
    torch.manual_seed(RANDOM_SEED)
    model = SimpleCNN().to(device)
    model = DDP(model)

    # Data
    train_dataset, test_dataset = get_datasets()
    train_loader, test_loader, train_sampler = create_distributed_dataloaders(
        train_dataset, test_dataset, args.batch_size, rank, world_size
    )

    # Training setup
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    results = {'epoch_times': [], 'train_losses': [], 'test_accuracies': []}
    total_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.perf_counter() - epoch_start
        results['epoch_times'].append(epoch_time)
        results['train_losses'].append(train_loss)
        results['test_accuracies'].append(test_acc)

        if rank == 0:
            print(f"Epoch [{epoch+1:2d}/{args.epochs}] | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

    total_time = time.perf_counter() - total_start

    if rank == 0:
        print(f"\nTraining completed in {total_time:.2f}s")
        results['total_time'] = total_time
        results['world_size'] = world_size

        # Save results
        with open(f'parallel_results_{world_size}workers.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to parallel_results_{world_size}workers.json")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Parallel CNN Training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--world-size', type=int, default=4, 
                        help='Number of workers (only used with mp.spawn)')
    args = parser.parse_args()

    # Check if launched via torchrun
    if 'RANK' in os.environ:
        # Launched via torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        train_worker(rank, world_size, args)
    else:
        # Launch via mp.spawn
        print(f"Launching {args.world_size} workers via mp.spawn...")
        mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size)


if __name__ == '__main__':
    main()
