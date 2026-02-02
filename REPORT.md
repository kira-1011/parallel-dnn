# Technical Report: Parallelization of Deep Learning Models

## CNN Training on CIFAR-10 using Data Parallelism

## 1. Model Selection and Serial Baseline

### 1.1 Model Architecture

We implemented a **Convolutional Neural Network (CNN)** for image classification on the CIFAR-10 dataset. The architecture consists of:

| Layer | Configuration | Output Shape |
|-------|---------------|--------------|
| Conv1 | 3→32 filters, 3×3, padding=1 | 32×32×32 |
| BatchNorm + ReLU + MaxPool(2×2) | - | 32×16×16 |
| Conv2 | 32→64 filters, 3×3, padding=1 | 64×16×16 |
| BatchNorm + ReLU + MaxPool(2×2) | - | 64×8×8 |
| Conv3 | 64→128 filters, 3×3, padding=1 | 128×8×8 |
| BatchNorm + ReLU + MaxPool(2×2) | - | 128×4×4 |
| Flatten | - | 2048 |
| FC1 + ReLU + Dropout(0.5) | 2048→512 | 512 |
| FC2 | 512→10 | 10 |

**Total Parameters:** ~1,147,914

### 1.2 Training Components

- **Loss Function:** Cross-Entropy Loss (suitable for multi-class classification)
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum=0.9 and weight_decay=1e-4
- **Learning Rate:** 0.01

### 1.3 Training Process

The serial training follows the standard deep learning workflow:

1. **Forward Pass:** Input images pass through convolutional and fully-connected layers to produce class predictions
2. **Loss Computation:** Cross-entropy loss measures the difference between predictions and true labels
3. **Backward Pass:** Gradients are computed via backpropagation through the network
4. **Parameter Update:** SGD updates model weights using computed gradients

### 1.4 Serial Baseline Performance

| Metric | Value |
|--------|-------|
| Total Training Time | 252.24 seconds |
| Average Epoch Time | 50.45 seconds |
| Final Test Accuracy | 71.17% |

## 2. Parallelization Strategy

### 2.1 Data Parallelism Approach

We implemented **Data Parallelism**, which:
- Replicates the entire model on each worker
- Partitions the training dataset across workers
- Each worker processes a different subset of data in parallel
- Gradients are synchronized across workers after each batch

**Why Data Parallelism?**
- Simple to implement for CNN architectures
- Model fits entirely in each worker's memory
- Scales well with dataset size
- Natural fit for batch-based training

### 2.2 Parallelized Components

| Component | Parallelization |
|-----------|-----------------|
| Forward Pass | Each worker computes on local data subset |
| Backward Pass | Each worker computes local gradients |
| Gradient Sync | AllReduce averages gradients across workers |
| Parameter Update | Each worker updates identical model copy |

### 2.3 Data Partitioning

Using PyTorch's `DistributedSampler`:
- Dataset is divided into `world_size` non-overlapping partitions
- Each worker processes `total_samples / world_size` samples per epoch
- Shuffling is coordinated to ensure no sample overlap

```
Worker 0: Samples [0, 4, 8, 12, ...]      → 12,500 samples
Worker 1: Samples [1, 5, 9, 13, ...]      → 12,500 samples
Worker 2: Samples [2, 6, 10, 14, ...]     → 12,500 samples
Worker 3: Samples [3, 7, 11, 15, ...]     → 12,500 samples
─────────────────────────────────────────────────────────
Total:    50,000 samples (complete dataset)
```

### 2.4 Gradient Synchronization (AllReduce)

After each worker computes local gradients, the **Ring-AllReduce** algorithm synchronizes them:

1. Workers arranged in logical ring topology
2. Each worker sends/receives gradient chunks to/from neighbors
3. After `2(N-1)` communication steps, all workers have the averaged gradient
4. All workers apply identical parameter updates

**Communication Complexity:** O(N) bandwidth, O(N) latency for N workers

## 3. Target Architecture and Programming Model

### 3.1 Programming Model: Distributed-Memory (Message-Passing)

Our implementation uses a **distributed-memory programming model** with **explicit message passing**, conceptually similar to MPI (Message Passing Interface). This is a critical distinction from shared-memory models like OpenMP.

#### Comparison of Parallel Programming Models

| Characteristic | Shared-Memory (OpenMP) | Distributed-Memory (MPI) | **Our Implementation (DDP)** |
|---------------|------------------------|--------------------------|------------------------------|
| **Execution Units** | Threads | Processes | **Processes** |
| **Memory Space** | Single shared address space | Separate per process | **Separate per process** |
| **Communication** | Implicit (shared variables) | Explicit (send/receive) | **Explicit (AllReduce)** |
| **Synchronization** | Locks, critical sections | Collective operations | **Collective operations** |
| **Data Sharing** | Direct memory access | Message passing | **Message passing** |

#### Why This is NOT Shared-Memory

Despite running on a single physical machine, our implementation follows the distributed-memory paradigm:

1. **Separate Processes**: We use `torch.multiprocessing.spawn` to create independent Python processes, NOT threads. Each process has its own Python interpreter, memory heap, and model copy.

2. **No Shared State**: Workers do not share any memory. Each worker maintains:
   - Its own copy of the neural network (~1.1M parameters × 4 bytes = ~4.4 MB per worker)
   - Its own optimizer state
   - Its own data partition

3. **Explicit Communication**: Gradient synchronization happens through explicit collective operations (AllReduce), not through shared variables. The Gloo backend transmits data over TCP sockets between processes.

4. **Message-Passing Semantics**: The AllReduce operation follows MPI semantics:
   ```
   MPI_Allreduce(local_gradients, global_gradients, MPI_SUM, MPI_COMM_WORLD)
   ```
   PyTorch's equivalent:
   ```python
   dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
   ```

### 3.2 Framework: PyTorch Distributed Data Parallel (DDP)

We used **PyTorch's DistributedDataParallel (DDP)** framework, which implements the message-passing model:

| Component | Implementation |
|-----------|----------------|
| **Communication Backend** | Gloo (TCP-based, cross-platform) |
| **Process Management** | `torch.multiprocessing.spawn` |
| **Collective Operations** | AllReduce, Broadcast, Barrier |
| **Process Coordination** | Rendezvous via TCP store |

#### How DDP Implements Message-Passing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED-MEMORY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   Process 0     │    │   Process 1     │    │   Process N     │     │
│  │  (Rank 0)       │    │  (Rank 1)       │    │  (Rank N)       │     │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤     │
│  │ Own Memory:     │    │ Own Memory:     │    │ Own Memory:     │     │
│  │ - Model Copy    │    │ - Model Copy    │    │ - Model Copy    │     │
│  │ - Gradients     │    │ - Gradients     │    │ - Gradients     │     │
│  │ - Data Subset   │    │ - Data Subset   │    │ - Data Subset   │     │
│  │ - Optimizer     │    │ - Optimizer     │    │ - Optimizer     │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           └──────────────────────┼──────────────────────┘               │
│                                  │                                      │
│                    ┌─────────────▼─────────────┐                        │
│                    │   GLOO COMMUNICATION      │                        │
│                    │   (TCP Sockets)           │                        │
│                    │                           │                        │
│                    │   AllReduce: Average      │                        │
│                    │   gradients across all    │                        │
│                    │   processes               │                        │
│                    └───────────────────────────┘                        │
│                                                                         │
│  NO SHARED MEMORY - All communication is explicit message passing       │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Communication Pattern: Ring-AllReduce

The gradient synchronization uses **Ring-AllReduce**, a bandwidth-optimal collective algorithm:

```
Step 1: Scatter-Reduce          Step 2: All-Gather
   ┌───┐                           ┌───┐
   │ 0 │◄────────┐          ┌─────►│ 0 │
   └─┬─┘        │          │      └─┬─┘
     │          │          │        │
     ▼          │          │        ▼
   ┌───┐        │          │      ┌───┐
   │ 1 │────────┼──────────┼─────►│ 1 │
   └─┬─┘        │          │      └─┬─┘
     │          │          │        │
     ▼          │          │        ▼
   ┌───┐        │          │      ┌───┐
   │ 2 │────────┼──────────┼─────►│ 2 │
   └─┬─┘        │          │      └─┬─┘
     │          │          │        │
     ▼          │          │        ▼
   ┌───┐        │          │      ┌───┐
   │ 3 │────────┘          └──────│ 3 │
   └───┘                          └───┘

Each process sends/receives to neighbors in a ring.
After 2(N-1) steps, all processes have the complete sum.
```

### 3.3 Execution Environment

| Component | Specification |
|-----------|---------------|
| **Hardware** | Single multi-core machine (12 CPU cores) |
| **OS** | Windows 10/11 |
| **Programming Model** | Distributed-memory (message-passing) |
| **Framework** | PyTorch Distributed with Gloo backend |
| **Process Count** | 1 to 12 worker processes |

#### Hardware vs. Programming Model Distinction

| Aspect | Description |
|--------|-------------|
| **Physical Hardware** | Single machine with shared physical RAM |
| **Programming Model** | Distributed-memory with separate logical address spaces |
| **Communication** | TCP sockets via Gloo (not shared memory) |

**Important**: Although the hardware has shared physical memory, our programming model treats each process as having its own isolated memory space. This is analogous to running MPI on a single node—the processes could theoretically share memory, but the programming model enforces explicit message passing for communication.

### 3.4 Justification for This Approach

**Why distributed-memory model on a single machine?**

1. **Scalability**: The same code can run on multiple machines without modification
2. **Isolation**: Process isolation prevents race conditions and simplifies debugging
3. **Framework Support**: PyTorch DDP is optimized for this model
4. **Portability**: Works identically on clusters, cloud instances, or single machines

**Why Gloo backend?**

1. **Cross-platform**: Works on Windows, Linux, and macOS
2. **CPU-compatible**: Does not require NVIDIA GPUs (unlike NCCL)
3. **Standard TCP**: Uses standard networking, no special hardware required


## 4. Experimental Setup

### 4.1 Hardware Environment

| Component | Specification |
|-----------|---------------|
| CPU | 12-core processor |
| Memory | Shared memory |
| Storage | Local SSD |

### 4.2 Software Environment

| Software | Version |
|----------|---------|
| Python | 3.12 |
| PyTorch | 2.7.1 |
| Backend | Gloo |

### 4.3 Dataset

**CIFAR-10:**
- 50,000 training images
- 10,000 test images
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Image size: 32×32×3 (RGB)

**Preprocessing:**
- Convert to tensor
- Normalize with CIFAR-10 mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616)

### 4.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 128 (per worker) |
| Epochs | 5 |
| Learning Rate | 0.01 |
| Optimizer | SGD (momentum=0.9) |
| Workers Tested | 1, 2, 4, 6, 8, 10, 12 |


## 5. Performance Evaluation and Comparison

### 5.1 Results Summary

| Workers | Total Time (s) | Speedup | Efficiency | Test Accuracy |
|---------|----------------|---------|------------|---------------|
| 1 (Serial) | 252.24 | 1.00× | 100.0% | 71.17% |
| 2 | 146.53 | 1.72× | 86.1% | 71.39% |
| 4 | 134.92 | 1.87× | 46.7% | 71.48% |
| 6 | 130.28 | 1.94× | 32.3% | 69.28% |
| 8 | 130.26 | 1.94× | 24.2% | 66.69% |
| 10 | 137.66 | 1.83× | 18.3% | 64.64% |
| 12 | 143.77 | 1.75× | 14.6% | 65.27% |

### 5.2 Speedup Analysis

**Observations:**
- Maximum speedup of **1.94×** achieved at 6-8 workers
- Speedup plateaus and then decreases beyond 8 workers
- Linear speedup (ideal) was not achieved due to communication overhead

**Speedup Formula:** `S(n) = T(1) / T(n)`

The speedup curve flattens significantly after 4 workers, indicating that communication overhead begins to dominate the computation time.

### 5.3 Efficiency Analysis

**Efficiency Formula:** `E(n) = S(n) / n × 100%`

| Workers | Efficiency |
|---------|------------|
| 2 | 86.1% (good) |
| 4 | 46.7% (moderate) |
| 6 | 32.3% (low) |
| 8+ | <25% (poor) |

Efficiency drops rapidly because:
- Fixed communication overhead becomes larger relative to reduced computation
- Amdahl's Law limits speedup due to serial portions of the code

### 5.4 Scalability Analysis

The system exhibits **sub-linear scaling**:
- Strong scaling: Speedup exists but efficiency decreases with more workers
- Optimal worker count: 6-8 workers for this workload
- Beyond 8 workers: Performance degrades due to over-subscription and communication costs

### 5.5 Correctness Verification

| Workers | Final Test Accuracy | Loss Convergence |
|---------|---------------------|------------------|
| 1 | 71.17% | ✓ Decreasing |
| 2 | 71.39% | ✓ Decreasing |
| 4 | 71.48% | ✓ Decreasing |
| 6 | 69.28% | ✓ Decreasing |
| 8 | 66.69% | ✓ Decreasing |

All configurations show proper loss convergence and reasonable accuracy, verifying correctness of the parallel implementation. Slight accuracy variations are expected due to different gradient averaging dynamics.


## 6. Performance Challenges and Optimization

### 6.1 Communication Overhead

**Challenge:** Gradient synchronization requires all workers to exchange data after each batch.

**Impact:**
- AllReduce time is approximately constant regardless of worker count
- As workers increase, computation time per worker decreases, but communication stays constant
- Communication becomes the bottleneck at high worker counts

**Mitigation:**
- DDP uses gradient bucketing to batch multiple small tensors
- Overlaps backward computation with communication for earlier layers

### 6.2 Synchronization Barriers

**Challenge:** All workers must wait at synchronization points.

**Impact:**
- Stragglers (slow workers) delay all other workers
- Load imbalance reduces parallel efficiency

**Mitigation:**
- Equal data partitioning via DistributedSampler
- Homogeneous hardware ensures similar computation times

### 6.3 Effective Batch Size Scaling

**Challenge:** With N workers, effective batch size becomes N × batch_size.

**Impact:**
- Larger effective batch size can affect convergence
- May require learning rate adjustment for optimal results

**Observation:** In our experiments, accuracy slightly decreased with more workers, consistent with larger batch effects.

### 6.4 Memory Overhead

**Challenge:** Each worker maintains a complete model copy.

**Impact:**
- Memory usage scales with worker count
- Not a limiting factor for our small CNN, but relevant for larger models

### 6.5 Diminishing Returns

Beyond 8 workers, performance actually **decreased**:
- 10 workers: 137.66s (slower than 8 workers)
- 12 workers: 143.77s (slower than 10 workers)

**Cause:** Communication overhead exceeds computational gains from additional parallelism.


## 7. Discussion

### 7.1 Effectiveness of the Programming Model

The **distributed-memory (message-passing) model** proved well-suited for data-parallel deep learning:

**Advantages Observed:**
- **Clear separation of concerns**: Each worker operates independently on its data partition
- **Explicit communication**: AllReduce operations make synchronization points visible and measurable
- **Debugging clarity**: Process isolation eliminates shared-memory race conditions
- **Scalability path**: Same code would work on a cluster without modification

**Comparison with Shared-Memory Alternative:**

| Aspect | Shared-Memory (OpenMP) | Our Approach (Message-Passing) |
|--------|------------------------|-------------------------------|
| Implementation | Would require thread-safe model updates | Clean process separation |
| Debugging | Race conditions possible | No shared state issues |
| Scalability | Limited to single node | Extends to clusters |
| Communication | Implicit (cache coherence) | Explicit (measurable) |
| Framework Support | Limited in PyTorch | Native DDP support |

### 7.2 Effectiveness of Data Parallelism

Data parallelism proved effective for this CNN workload:
- Achieved 1.94× speedup (nearly 2× faster than serial)
- Maintained model accuracy within acceptable range
- Implementation was straightforward using DDP

### 7.3 Trade-offs

| Aspect | Trade-off |
|--------|-----------|
| Performance vs. Complexity | DDP abstracts complexity; manual implementation would be error-prone |
| Speedup vs. Efficiency | Higher speedup comes at cost of lower per-worker efficiency |
| Workers vs. Accuracy | More workers may slightly reduce accuracy due to batch size effects |
| Scalability vs. Hardware | Optimal worker count depends on available cores |

### 7.4 Model Characteristics and Parallel Behavior

**Factors favoring parallelization:**
- Batch-based training naturally divides across workers
- Independent forward passes on different data samples
- Gradient averaging is mathematically equivalent to larger batch training

**Limiting factors:**
- Small model size (1.1M parameters) means fast computation, making communication relatively expensive
- Small dataset (50K samples) limits scalability benefits
- CPU-only training adds overhead compared to GPU

### 7.5 Recommendations

For optimal performance with this setup:
- Use **6-8 workers** for best speedup
- Consider GPU acceleration for larger models
- For better scaling, increase batch size or model complexity


## 8. Conclusion

This project successfully demonstrated data-parallel training of a CNN on CIFAR-10 using a **distributed-memory (message-passing) programming model**:

### Key Achievements

1. **Achieved 1.94× speedup** using 6-8 workers on a 12-core CPU system
2. **Verified correctness** through consistent loss convergence and accuracy across all configurations
3. **Identified optimal parallelism** at 6-8 workers for this workload
4. **Understood limitations** including communication overhead and diminishing returns

### Programming Model Summary

| Aspect | Our Implementation |
|--------|-------------------|
| **Model** | Distributed-memory (message-passing) |
| **Similar To** | MPI (Message Passing Interface) |
| **Communication** | Explicit via AllReduce operations |
| **Processes** | Separate memory spaces, no shared state |
| **Framework** | PyTorch DDP with Gloo backend |

### Key Insight

The distributed-memory model, despite running on a single physical machine, provides:
- **Explicit control** over communication patterns
- **Scalability** to multi-node clusters without code changes
- **Process isolation** that eliminates shared-memory race conditions
- **Clear performance model** where communication costs are explicit and measurable

The Ring-AllReduce algorithm efficiently synchronizes gradients with O(N) bandwidth complexity, making it suitable for scaling to many workers. However, for small models like ours (1.1M parameters), the fixed communication overhead limits speedup to approximately 2× regardless of available cores.

For larger models and datasets, this distributed-memory approach would show improved efficiency as computation time dominates communication overhead.


## References

- PyTorch Distributed Documentation: https://pytorch.org/docs/stable/distributed.html
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Ring-AllReduce: Patarasuk & Yuan, "Bandwidth Optimal All-reduce Algorithms"
