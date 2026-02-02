# Parallel Deep Learning: CNN Training on CIFAR-10

This project implements and compares serial and parallel deep learning training using PyTorch's DistributedDataParallel (DDP).

## Project Overview

**Assignment:** Parallelization of Deep Learning Models

| Component | Description |
|-----------|-------------|
| Model | SimpleCNN (~1.1M parameters) |
| Dataset | CIFAR-10 (50,000 training images) |
| Parallelization | Data Parallelism with DDP |
| Communication | Ring-AllReduce via Gloo backend |

## File Structure

```
parallel-dnn/
├── main.ipynb              # Notebook: serial training + analysis
├── parallel_training.py    # Script: parallel training (run from terminal)
├── REPORT.md               # Technical report
├── README.md               # This file
├── parallel_results_*.json # Experiment results (generated)
└── performance_analysis.png # Performance plots (generated)
```

## Prerequisites

- Python 3.10+
- PyTorch 2.7.x (tested with 2.7.1)
- Windows/Linux/macOS

## Step-by-Step Setup

### Step 1: Clone or Download the Project

```bash
cd E:\parallel-dnn   # or your project directory
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using uv (recommended)
uv venv
uv pip install torch==2.7.1 torchvision==0.22.1 numpy matplotlib ipykernel

# OR using pip
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install torch==2.7.1 torchvision==0.22.1 numpy matplotlib ipykernel
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

Expected output: `PyTorch 2.7.1` (or similar)

## Running the Experiments

### Experiment 1: Serial Baseline (in Notebook)

1. Open `main.ipynb` in Jupyter or VS Code
2. Run all cells in **Part 1** (Setup and Serial Implementation)
3. The serial training will run and save results

**Expected output:**
```
Epoch [1/5] | Loss: 1.55 | Train Acc: 43.2% | Test Acc: 52.1% | Time: 45.2s
Epoch [2/5] | Loss: 1.12 | Train Acc: 59.8% | Test Acc: 62.3% | Time: 44.8s
...
```

### Experiment 2: Parallel Training (in Terminal)

**Important:** Parallel training must be run from the command line, not in Jupyter.

Open a terminal and run:

```bash
# Step 1: Navigate to project directory
cd E:\parallel-dnn

# Step 2: Activate virtual environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Step 3: Run parallel training with different worker counts
python parallel_training.py --world-size 1 --epochs 5
python parallel_training.py --world-size 2 --epochs 5
python parallel_training.py --world-size 4 --epochs 5
python parallel_training.py --world-size 6 --epochs 5
python parallel_training.py --world-size 8 --epochs 5
```

**Expected output for 4 workers:**
```
Launching 4 workers via mp.spawn...
[01:23:45.123] [Worker 0] Worker starting - PID: 12345
[01:23:45.125] [Worker 1] Worker starting - PID: 12346
[01:23:45.127] [Worker 2] Worker starting - PID: 12347
[01:23:45.129] [Worker 3] Worker starting - PID: 12348
...
Epoch [ 1/5] | Loss: 1.76 | Train Acc: 35.1% | Test Acc: 50.7% | Time: 26.5s
...
Training completed in 134.92s
Results saved to parallel_results_4workers.json
```

### Experiment 3: Analyze Results (in Notebook)

1. After running parallel experiments, return to `main.ipynb`
2. Run cells in **Part 4** (Performance Analysis)
3. This will generate:
   - Performance comparison table
   - `performance_analysis.png` with speedup/efficiency plots

## Command Reference

| Command | Description |
|---------|-------------|
| `python parallel_training.py --world-size 4` | Run with 4 workers |
| `python parallel_training.py --epochs 10` | Run for 10 epochs |
| `python parallel_training.py --batch-size 64` | Use batch size 64 |
| `python parallel_training.py --verbose` | Enable detailed logging |
| `python parallel_training.py --quiet` | Disable verbose logging |

## Full Experiment Reproduction

To reproduce all results from scratch:

```bash
# 1. Activate environment
cd E:\parallel-dnn
.venv\Scripts\activate

# 2. Run serial baseline (in notebook or use --world-size 1)
python parallel_training.py --world-size 1 --epochs 5

# 3. Run parallel experiments
python parallel_training.py --world-size 2 --epochs 5
python parallel_training.py --world-size 4 --epochs 5
python parallel_training.py --world-size 6 --epochs 5
python parallel_training.py --world-size 8 --epochs 5
python parallel_training.py --world-size 10 --epochs 5
python parallel_training.py --world-size 12 --epochs 5

# 4. Open main.ipynb and run Part 4 to generate analysis
```

## Expected Results

Based on Intel i5-12450H (8 cores, 12 threads):

| Workers | Total Time | Speedup | Efficiency | Test Accuracy |
|---------|------------|---------|------------|---------------|
| 1 | ~252s | 1.00× | 100% | ~71% |
| 2 | ~147s | 1.72× | 86% | ~71% |
| 4 | ~135s | 1.87× | 47% | ~71% |
| 6 | ~130s | 1.94× | 32% | ~69% |
| 8 | ~130s | 1.94× | 24% | ~67% |

**Key Observations:**
- Maximum speedup: ~1.94× at 6-8 workers
- Performance degrades beyond 8 workers (communication overhead)
- Accuracy slightly decreases with more workers (larger effective batch)

## Troubleshooting

### Error: `unsupported gloo device`
**Solution:** Downgrade to PyTorch 2.7.x
```bash
uv pip install torch==2.7.1 torchvision==0.22.1
```

### Error: `use_libuv was requested but PyTorch was built without libuv support`
**Solution:** Use `python parallel_training.py` instead of `torchrun`

### Error: `Access is denied` when installing packages
**Solution:** Close all Python processes and Jupyter kernels, then retry

### Parallel training hangs
**Possible causes:**
- Windows Firewall blocking inter-process communication
- Another process using port 12355
**Solution:** Try changing `MASTER_PORT` in `parallel_training.py`

## Configuration

Default training parameters (can be changed via command line):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 128 | Batch size per worker |
| `--lr` | 0.01 | Learning rate |
| `--world-size` | 4 | Number of parallel workers |

## References

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
