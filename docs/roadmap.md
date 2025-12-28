# Vic-Torch Roadmap & Design Doc

This document scopes Vic-Torch into phased deliverables with measurable goals, benchmarking baselines, and technical bets to prioritize. It is intentionally pragmatic: each phase should land a usable artifact, a minimal benchmark suite, and clear performance/compatibility/UX targets.

## Scope Phases

### Phase 0 — Core Tensor + Autograd (MVP)
**Goal:** Provide a minimal PyTorch-like tensor API with eager execution and reverse-mode autograd.

**Deliverables**
- Tensor core (CPU first, optional CUDA behind feature flag).
- Autograd tape, basic ops (matmul, conv, activations, reductions).
- Serialization format for tensors and model states.
- Minimal optimizer set (SGD, Adam).

**Out-of-scope**
- Graph compilation, distributed training, model zoo.

### Phase 1 — Model Zoo + Training Loops
**Goal:** Make it easy to train and evaluate canonical models with reproducible recipes.

**Deliverables**
- Reference implementations: ResNet, Transformer, BERT-small.
- Data loaders for CIFAR-10, ImageNet-lite, WikiText-2.
- Training CLI with config files.
- Checkpointing + mixed precision.

### Phase 2 — Compiler + Graph/Eager Hybrid
**Goal:** Introduce a graph capture + compiler path without sacrificing eager UX.

**Deliverables**
- Graph capture API (`torch.compile`-like), fallback to eager for unsupported ops.
- AOT autograd for fused backward passes.
- Basic op fusion and kernel scheduling.

### Phase 3 — Distributed Training
**Goal:** Add single-node multi-GPU and multi-node training support.

**Deliverables**
- NCCL/Gloo backends.
- DDP-style data parallelism, gradient bucketing.
- Checkpoint sharding.

### Phase 4 — Runtime/Ops + Kernel DSL
**Goal:** Mature runtime with custom kernel authoring and hardware portability.

**Deliverables**
- Kernel DSL for custom ops (CPU + CUDA).
- MLIR or LLVM lowering path.
- Profiler and runtime tracer.

## Measurable Goals

### Performance Targets
- **Phase 0:** Matmul throughput within **2–3x PyTorch** on CPU for square matrices up to 4096.
- **Phase 1:**
  - **ResNet-18**: ≥ **400 img/s** on single A100 (mixed precision).
  - **BERT-small**: ≥ **1,200 seq/s** on single A100 (seq len 128).
- **Phase 2:** ≥ **1.3x speedup** vs eager on ResNet-50 inference through graph compilation.
- **Phase 3:** **≥ 0.85 scaling efficiency** going from 1 → 8 GPUs on ResNet-50.
- **Phase 4:** Custom fused kernel matches **≥ 90%** of vendor-optimized kernel on representative ops (GEMM/LayerNorm).

### Compatibility Targets
- **Phase 1:** Import **ONNX opset 13** for a minimal subset (Conv, MatMul, Gemm, Relu, LayerNorm, Softmax).
- **Phase 2:** Export ONNX opset 13 for core models.
- **Phase 3:** Interop with PyTorch checkpoints for model weights (key remapping).

### Developer UX Goals
- **Phase 0:** Eager API parity for basic tensor ops; docs + examples for autograd.
- **Phase 1:** One-command training: `victrain --model resnet18 --dataset cifar10`.
- **Phase 2:** `vic.compile(model)` drops into graph compile with a clear fallback report.
- **Phase 3:** Launch distributed job with a single config file and a `vicrun` wrapper.
- **Phase 4:** Kernel DSL tutorial with a working vectorized op in < 50 lines.

## Revolutionary Approaches To Prioritize

1. **MLIR-first compilation**: lower tensor IR into MLIR dialects early for portability.
2. **Graph + eager hybrid**: capture graphs opportunistically while keeping eager defaults.
3. **Memory-aware scheduling**: schedule ops based on memory pressure to reduce OOMs and swap.
4. **Custom kernel DSL**: safe, ergonomic, JIT-able DSL for authoring kernels.
5. **Autograd-aware fusion**: fuse forward + backward for common patterns (e.g., LayerNorm + activation).

## Benchmark Suites & Minimal Baselines

### Phase 0 (Core Tensor + Autograd)
**Benchmark suites**
- **Microbench**: matmul, elementwise, reductions.
- **Autograd**: linear regression with MSE, 2-layer MLP.
- **Memory**: peak allocation tracking during MLP training.

**Minimal baseline**
- Matmul 1024x1024 runs without error and autograd gradients match finite-difference checks.

### Phase 1 (Model Zoo + Training)
**Benchmark suites**
- **CV**: ResNet-18 on CIFAR-10.
- **NLP**: BERT-small on WikiText-2.
- **Throughput**: single-GPU training throughput for ResNet-18.

**Minimal baseline**
- ResNet-18 reaches **≥ 80%** CIFAR-10 accuracy within 90 epochs (standard augmentations).

### Phase 2 (Compiler + Hybrid)
**Benchmark suites**
- **Graph compile**: ResNet-50 inference throughput.
- **Fusion**: LayerNorm + GELU fusion benchmark.
- **AOT backward**: compare graph vs eager backward time for Transformer blocks.

**Minimal baseline**
- Graph compile achieves **≥ 1.1x** speedup on ResNet-50 inference vs eager.

### Phase 3 (Distributed Training)
**Benchmark suites**
- **Scaling**: ResNet-50 throughput from 1 → 8 GPUs.
- **Stability**: all-reduce correctness under gradient accumulation.
- **Fault tolerance**: resume training from sharded checkpoints.

**Minimal baseline**
- 8-GPU training delivers **≥ 0.8 scaling efficiency** vs single-GPU baseline.

### Phase 4 (Runtime/Ops + Kernel DSL)
**Benchmark suites**
- **Kernel DSL**: custom LayerNorm kernel vs vendor baseline.
- **Operator coverage**: top-50 ops used by ResNet/BERT.
- **Latency**: operator launch overhead vs PyTorch.

**Minimal baseline**
- Custom LayerNorm kernel achieves **≥ 85%** of vendor-optimized performance.

## Risks & Mitigations
- **Scope creep:** enforce phase gates; avoid compiler work before Phase 1 is stable.
- **Performance debt:** set benchmarks per phase and fail CI if regression > 10%.
- **UX erosion:** keep eager API as default and document every compile fallback.

## Milestone Tracking
Each phase is considered complete when:
- All deliverables are implemented.
- Baseline benchmarks pass.
- Performance/compatibility/UX goals are met or explicitly deferred with rationale.
