# Vic-Torch Architecture Documentation

> **Version:** v1.0.0  
> **Last Updated:** 2026-01-24  
> **Status:** Phase 0 (Core Tensor + Autograd)

This document provides a detailed breakdown of all modules, components, and design patterns in the Vic-Torch framework.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Module Breakdown](#module-breakdown)
4. [Data Flow & Execution Model](#data-flow--execution-model)
5. [Memory Architecture](#memory-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Extension Points](#extension-points)

---

## System Overview

Vic-Torch is a modular, self-evolving AI framework built around three core principles:

1. **Fractal Memory Architecture** - Memory shards that evolve and compress over time
2. **Recursive Self-Improvement** - Models that can modify their own structure during training
3. **Lightweight Execution** - 30K params, 240KB footprint, runs on minimal hardware

### Design Philosophy

- **Eager-first execution** with optional graph compilation path (future)
- **Modular composition** over monolithic architectures
- **Memory-persistent training** with automatic checkpoint management
- **Introspection-native** - models understand their own structure

---

## Core Architecture

```
Vic-Torch/
├── victorch/
│   ├── core/              # Tensor primitives & autograd engine
│   ├── models/            # Model architectures (Victor Transformer)
│   ├── modules/           # Reusable layers (Dense, Transformer Block, etc.)
│   ├── optim/             # Optimizers (Adam, SGD)
│   ├── memory/            # Memory bank systems (standard & Godcore)
│   ├── inspector/         # Model introspection & analysis tools
│   └── runner/            # Training & evaluation scripts
├── docs/                  # Documentation & roadmap
├── configs/               # Configuration files
├── scripts/               # CLI utilities
└── tests/                 # Unit & integration tests
```

---

## Module Breakdown

### 1. Core Tensor System (`victorch/core/`)

#### `tensor.py` - VictorTensor
**Purpose:** Dynamic tensor with automatic differentiation  
**Key Features:**
- Reverse-mode autograd via computation graph
- Lazy gradient accumulation
- NumPy backend with planned CUDA support

**API:**
```python
class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None)
    def backward(self, grad=None)  # Backpropagate gradients
    def zero_grad()                # Reset gradients
    def shape()                    # Return tensor shape
    
    # Operations (autograd-aware)
    def __add__, __sub__, __mul__, __truediv__
    def matmul(other)
    def mean(), sum(), min(), max()
    def reshape(), transpose(), squeeze(), unsqueeze()
```

**Autograd Operations Supported:**
- `add`, `sub`, `mul`, `div` (element-wise)
- `matmul` (matrix multiplication with correct gradient shapes)
- Reduction ops (mean, sum) - partial support

**Current Limitations:**
- Broadcasting gradients not fully implemented
- No in-place operations
- CPU-only (CUDA planned for Phase 1)

---

### 2. Model Architecture (`victorch/models/`)

#### `victor_model.py` - VictorTransformerModel
**Purpose:** Full transformer-based model with modular blocks  
**Architecture:**

```
Input (vocab indices or embeddings)
    ↓
Embedding Layer (Dense: vocab_size → embed_dim)
    ↓
Positional Encoding (sinusoidal)
    ↓
Transformer Block 1 (Self-Attention + FFN)
    ↓
Transformer Block 2
    ↓
    ...
    ↓
Transformer Block N
    ↓
Output Projection (Dense: embed_dim → num_classes)
    ↓
Logits
```

**Configuration:**
```python
model = VictorTransformerModel(
    vocab_size=32,      # Input vocabulary size
    embed_dim=32,       # Embedding dimension
    num_layers=4,       # Number of transformer blocks
    hidden_dim=64,      # FFN hidden dimension
    num_classes=10      # Output classes
)
```

**Default Configuration Stats:**
- Total Parameters: **30,826**
- Memory Footprint: **240.83 KB**
- Layers: 4 transformer blocks

---

### 3. Reusable Modules (`victorch/modules/`)

#### `layers.py` - Dense Layer
**Purpose:** Fully connected layer with weight + bias  
**Forward:**  
```python
output = input @ W + b  # (batch, in_features) → (batch, out_features)
```

#### `transformer_block.py` - TransformerBlock
**Components:**
1. **Self-Attention** (multi-head or single-head)
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Attention weights: `softmax(Q @ K.T / sqrt(d_k))`
2. **Feed-Forward Network (FFN)**
   - Two-layer MLP: `ReLU(x @ W1 + b1) @ W2 + b2`
3. **Layer Normalization** (planned)
4. **Residual Connections** (planned)

#### `losses.py` - Loss Functions
- **MSELoss**: Mean Squared Error for regression
- **CrossEntropyLoss**: Numerically stable cross-entropy for classification
  - Uses log-sum-exp trick to prevent overflow

---

### 4. Optimizers (`victorch/optim/`)

#### `optimizers.py` - Adam Optimizer
**Update Rule:**
```python
m_t = β1 * m_{t-1} + (1 - β1) * grad        # First moment
v_t = β2 * v_{t-1} + (1 - β2) * grad^2      # Second moment
m_hat = m_t / (1 - β1^t)                     # Bias correction
v_hat = v_t / (1 - β2^t)
param -= lr * m_hat / (sqrt(v_hat) + ε)
```

**Hyperparameters:**
- `lr`: Learning rate (default: 0.001)
- `β1`: First moment decay (default: 0.9)
- `β2`: Second moment decay (default: 0.999)
- `ε`: Numerical stability constant (default: 1e-8)

**Planned:** SGD, AdamW, RMSprop

---

### 5. Memory Systems (`victorch/memory/`)

#### `memory_bank.py` - MemoryBank
**Purpose:** Experience replay buffer for episodic training  
**Features:**
- Capacity-limited circular buffer
- Save/load checkpoints
- Random sampling for batch training

**API:**
```python
memory = MemoryBank(capacity=10000)
memory.store({"input": x, "output": y, "loss": loss_val})
batch = memory.sample(batch_size=32)
memory.save("checkpoint.npz")
memory.load("checkpoint.npz")
```

#### `godcore_memory_bank.py` - GodcoreMemoryBank
**Purpose:** Advanced memory system with auto-archiving and versioning  
**Features:**
- Automatic periodic saves (configurable interval)
- Checkpoint versioning (v1.0, v1.1, ...)
- ZIP archiving for long-term storage
- Prime Directive Shard verification (integrity check)
- Background save scheduler (daemon thread)

**API:**
```python
memory = GodcoreMemoryBank(
    project_name="VictorCortex",
    model_name="BaseModel",
    autosave_root="checkpoints",
    capacity=50000,
    autosave_interval=50,    # Save every 50 episodes
    archive_interval=5        # Archive every 5 saves
)
```

**Directory Structure:**
```
checkpoints/
└── VictorCortex/
    └── BaseModel/
        ├── v1.0/
        │   ├── memory_20260124_143022.pkl
        │   └── memory_20260124_143022.pkl.sig
        ├── v1.1/
        └── v1.0.zip  # Archived after 5 saves
```

---

### 6. Introspection Tools (`victorch/inspector/`)

#### `model_inspector.py` - summarize_model()
**Purpose:** Analyze model structure and resource usage  
**Output:**
```
=== Victor Model Inspector ===
VictorTransformerModel: 30826 params | 240.83 KB
  ├─ embedding: 1024 params | 8.00 KB
  ├─ transformer_blocks[0]: 8448 params | 66.00 KB
  ├─ transformer_blocks[1]: 8448 params | 66.00 KB
  ├─ transformer_blocks[2]: 8448 params | 66.00 KB
  ├─ transformer_blocks[3]: 8448 params | 66.00 KB
  └─ output_layer: 320 params | 2.50 KB
------------------------------
Total Parameters: 30826
Estimated Memory: 240.83 MB
==============================
```

---

## Data Flow & Execution Model

### Forward Pass
```
Input Tensor (requires_grad=True)
    ↓
Embedding Lookup
    ↓
Positional Encoding (additive)
    ↓
[For each Transformer Block]
    ↓
  Self-Attention
    ├─ Q = input @ W_q
    ├─ K = input @ W_k
    ├─ V = input @ W_v
    ├─ Attention = softmax(Q @ K.T / sqrt(d_k))
    └─ Output = Attention @ V
    ↓
  Feed-Forward Network
    └─ ReLU(x @ W1) @ W2
    ↓
Output Projection
    ↓
Logits (batch, num_classes)
```

### Backward Pass
```
Loss.backward()  # Scalar loss
    ↓
Gradient flows backward through computation graph
    ↓
[For each operation with creators]
    ↓
  Compute local gradients (chain rule)
    ↓
  Accumulate gradients in .grad attribute
    ↓
Optimizer.step()  # Update weights using accumulated gradients
    ↓
Optimizer.zero_grad()  # Clear gradients for next iteration
```

---

## Memory Architecture

### Memory Bank Lifecycle

1. **Training Episode**
   ```python
   x, y = generate_data()
   pred = model(x)
   loss = loss_fn(pred, y)
   loss.backward()
   optimizer.step()
   ```

2. **Store Experience**
   ```python
   memory_bank.store({
       "input": x.data,
       "output": y.data,
       "loss": loss.data,
       "epoch": epoch_num
   })
   ```

3. **Auto-Save Trigger**
   - Every `autosave_interval` episodes
   - Checkpoint saved with timestamp
   - SHA256 signature generated

4. **Archive Trigger**
   - Every `archive_interval` saves
   - ZIP compression
   - Old checkpoints preserved

---

## Training Pipeline

### Standard Training Loop (`train_victor.py`)

```python
# 1. Initialize
model = VictorTransformerModel(...)
optimizer = Adam(model.parameters(), lr=0.001)
memory_bank = MemoryBank(capacity=5000)

# 2. Multi-task training
tasks = ['regression', 'classification', 'autoencode', 'fractal_compression']

for epoch in range(epochs):
    task = random.choice(tasks)
    x, y = generate_batch(task)
    
    # 3. Forward
    preds = model(x)
    
    # 4. Compute loss
    if task == 'classification':
        loss = ce_loss(preds, y)
    else:
        loss = mse_loss(preds, y)
    
    # 5. Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 6. Store experience
    memory_bank.store({"input": x, "output": y, "loss": loss})
    
    # 7. Periodic save
    if epoch % save_interval == 0:
        memory_bank.save(f"checkpoint_{epoch}.npz")
```

### Godcore Training Loop (`train_victor_godcore_v2.py`)

**Key Differences:**
- Uses `GodcoreMemoryBank` with auto-archiving
- Automatic background saves (daemon thread)
- Version-controlled checkpoints
- Prime Directive Shard integrity verification

---

## Extension Points

### 1. Custom Layers
```python
class CustomLayer:
    def __init__(self, ...):
        self.weights = Tensor(...)
    
    def __call__(self, x):
        return x @ self.weights
    
    def parameters(self):
        return [self.weights]
```

### 2. Custom Loss Functions
```python
class CustomLoss:
    def __call__(self, pred, target):
        # Compute loss
        return Tensor(loss_value, requires_grad=True)
```

### 3. Custom Optimizers
```python
class CustomOptimizer:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad.data
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
```

---

## Future Architecture Plans

### Phase 1 (Model Zoo + Training)
- Add ResNet, BERT-small implementations
- Data loaders for CIFAR-10, ImageNet-lite
- Mixed precision training
- Advanced checkpointing

### Phase 2 (Compiler + Graph/Eager Hybrid)
- Graph capture API (`vic.compile(model)`)
- AOT autograd
- Op fusion (LayerNorm + activation)

### Phase 3 (Distributed Training)
- Multi-GPU support (NCCL/Gloo)
- Data parallelism
- Gradient bucketing

### Phase 4 (Runtime/Ops + Kernel DSL)
- Custom kernel authoring DSL
- MLIR/LLVM lowering
- Hardware-specific optimizations

---

## References

- [Roadmap](roadmap.md) - Phased development plan
- [Contributing](../CONTRIBUTING.md) - How to contribute
- [Benchmarks](roadmap.md#benchmark-suites--minimal-baselines) - Performance baselines

---

**Maintained by:** Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)  
**License:** GODCORE Fractal Sovereign License  
**Project:** Massive Magnetics / Ethica AI / BHeard Network