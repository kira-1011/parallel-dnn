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
from datetime import datetime

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


# Configuration
RANDOM_SEED = 42
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
VERBOSE = True  # Enable verbose logging to see parallel execution


def log(rank, message, force=False):
    """Print timestamped log message with worker rank."""
    if VERBOSE or force:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [Worker {rank}] {message}", flush=True)


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


def get_datasets():
    """Load CIFAR-10 datasets with appropriate transforms."""
    train_transform = transforms.Compose([
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
                                   rank, world_size, num_workers=0):
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


def train_one_epoch(model, train_loader, optimizer, criterion, device, rank, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)
    
    log(rank, f"Epoch {epoch+1}: Starting training on {num_batches} batches")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        
        # Log progress every 25% of batches
        if (batch_idx + 1) % max(1, num_batches // 4) == 0 or batch_idx == 0:
            log(rank, f"Epoch {epoch+1}: Batch {batch_idx+1}/{num_batches} "
                      f"({100*(batch_idx+1)/num_batches:.0f}%) - Loss: {loss.item():.4f}")

    log(rank, f"Epoch {epoch+1}: Training complete - processed {total} samples")
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device, rank, epoch):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    log(rank, f"Epoch {epoch+1}: Starting evaluation")
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    log(rank, f"Epoch {epoch+1}: Local evaluation done - {correct}/{total} correct")
    
    # Aggregate across workers
    if dist.is_initialized():
        log(rank, f"Epoch {epoch+1}: Starting AllReduce to aggregate metrics...")
        metrics = torch.tensor([correct, total, running_loss], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        correct, total, running_loss = metrics.tolist()
        log(rank, f"Epoch {epoch+1}: AllReduce complete - global {int(correct)}/{int(total)} correct")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return running_loss / len(test_loader) / world_size, 100. * correct / total


def train_worker(rank, world_size, args):
    """Main training function for each worker."""
    log(rank, f"Worker starting - PID: {os.getpid()}", force=True)
    
    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    log(rank, "Initializing process group...")
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    log(rank, f"Process group initialized - {world_size} workers connected", force=True)

    device = torch.device('cpu')
    log(rank, f"Using device: {device}")

    # Create model
    torch.manual_seed(RANDOM_SEED)
    model = SimpleCNN().to(device)
    log(rank, "Model created, wrapping with DDP...")
    model = DDP(model)
    log(rank, "DDP wrapper applied - model parameters will be synchronized")

    # Data
    log(rank, "Loading datasets...")
    train_dataset, test_dataset = get_datasets()
    train_loader, test_loader, train_sampler = create_distributed_dataloaders(
        train_dataset, test_dataset, args.batch_size, rank, world_size
    )
    samples_per_worker = len(train_loader) * args.batch_size
    log(rank, f"Data loaded - {samples_per_worker} training samples assigned to this worker", force=True)

    # Training setup
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Synchronization barrier before training
    log(rank, "Waiting at barrier for all workers to be ready...")
    dist.barrier()
    log(rank, "All workers synchronized - starting training!", force=True)

    # Training loop
    results = {'epoch_times': [], 'train_losses': [], 'test_accuracies': []}
    total_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, rank, epoch)
        
        # Synchronization point after training (implicit in DDP, explicit barrier for logging)
        log(rank, f"Epoch {epoch+1}: Waiting for all workers to finish training...")
        dist.barrier()
        log(rank, f"Epoch {epoch+1}: All workers finished training - starting evaluation")
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank, epoch)

        epoch_time = time.perf_counter() - epoch_start
        results['epoch_times'].append(epoch_time)
        results['train_losses'].append(train_loss)
        results['test_accuracies'].append(test_acc)

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1:2d}/{args.epochs}] SUMMARY | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")
            print(f"{'='*70}\n")

    total_time = time.perf_counter() - total_start

    log(rank, f"Training complete - total time: {total_time:.2f}s", force=True)
    
    if rank == 0:
        print(f"\nTraining completed in {total_time:.2f}s")
        results['total_time'] = total_time
        results['world_size'] = world_size

        # Save results
        with open(f'parallel_results_{world_size}workers.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to parallel_results_{world_size}workers.json")

    log(rank, "Cleaning up process group...")
    dist.destroy_process_group()
    log(rank, "Worker finished", force=True)


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description='Parallel CNN Training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--world-size', type=int, default=4, 
                        help='Number of workers (only used with mp.spawn)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Enable verbose logging to see parallel execution')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable verbose logging')
    args = parser.parse_args()
    
    # Set verbose mode
    VERBOSE = args.verbose and not args.quiet

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
