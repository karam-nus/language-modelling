---
title: "Chapter 21 — CUDA & Kernel Development"
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — CUDA & Kernel Development

> *"If you want to understand why Flash Attention is fast, or why fused kernels matter, you need to understand how GPU programs work at the thread level."*

## The CUDA Programming Model

CUDA organizes computation into a hierarchy:

<div class="diagram">
<div class="diagram-title">CUDA Thread Hierarchy</div>
<div class="layer-stack">
  <div class="layer accent">Grid — All threads launched by a single kernel call</div>
  <div class="layer purple">Block — Up to 1024 threads. Shared memory within a block. Maps to one SM.</div>
  <div class="layer green">Warp — 32 threads executing in lockstep (SIMT). Hardware scheduling unit.</div>
  <div class="layer orange">Thread — Individual execution unit. Has its own registers.</div>
</div>
</div>

Each thread knows its position via built-in variables:
- `threadIdx.x` — thread ID within the block
- `blockIdx.x` — block ID within the grid
- `blockDim.x` — number of threads per block

## A First Kernel: Vector Addition

```c
// vector_add.cu — simplest possible CUDA kernel
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch: ceil(n/256) blocks, 256 threads each
// vector_add<<<(n+255)/256, 256>>>(a, b, c, n);
```

This kernel is **memory-bound**: each thread does 1 add but reads 2 floats and writes 1 float (12 bytes for 1 FLOP).

## Why Fused Kernels Matter

Every separate kernel launch reads from and writes to HBM. Fusing operations into a single kernel keeps intermediate values in fast SRAM:

<div class="diagram">
<div class="diagram-title">Unfused vs Fused Operations</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Unfused (3 kernels)</div>
    <ul>
      <li>Kernel 1: matmul → write to HBM</li>
      <li>Kernel 2: read from HBM → add bias → write to HBM</li>
      <li>Kernel 3: read from HBM → GELU → write to HBM</li>
      <li>6 HBM reads/writes total</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Fused (1 kernel)</div>
    <ul>
      <li>Kernel: matmul → bias (in SRAM) → GELU (in SRAM) → write to HBM</li>
      <li>Only the final result hits HBM</li>
      <li>2 reads + 1 write total</li>
      <li>3× less memory traffic</li>
    </ul>
  </div>
</div>
</div>

This is why `torch.compile` and frameworks like Triton exist — they automatically fuse element-wise operations.

## Tiled Matrix Multiplication

The naive matrix multiply reads each element from HBM many times. Tiling loads blocks into shared memory, reusing each element multiple times:

```c
// Simplified tiled matmul (C = A × B)
__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < K / TILE_SIZE; t++) {
        // Load tiles from HBM to shared memory (collaborative)
        tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

Each element of A and B is loaded from HBM once per tile but reused TILE_SIZE times in shared memory. This reduces HBM traffic by TILE_SIZE×.

## Triton: GPU Kernels in Python

**Triton** (OpenAI) lets you write GPU kernels in Python, with automatic tiling and shared memory management:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Fused online softmax in Triton."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row from HBM
    row = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))

    # Compute softmax in SRAM (numerically stable)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write back to HBM
    tl.store(output_ptr + row_idx * n_cols + col_offsets, softmax_output, mask=mask)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

## Flash Attention Kernel Design

Flash Attention ([Chapter 4](./04_attention_sdpa_and_mha.md)) is the most impactful fused kernel in LLM inference. Its key innovations:

<div class="diagram">
<div class="diagram-title">Flash Attention Kernel Techniques</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Tiling</div>
    <div class="card-desc">Divide Q, K, V into blocks that fit in SRAM. Process attention block by block. Never materialize the full T×T score matrix in HBM.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Online Softmax</div>
    <div class="card-desc">Compute softmax incrementally across blocks using running max and sum statistics. Numerically equivalent to standard softmax.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Kernel Fusion</div>
    <div class="card-desc">Fuse Q×K^T, scaling, masking, softmax, and ×V into a single kernel. One read of Q, K, V from HBM.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Recomputation</div>
    <div class="card-desc">In the backward pass, recompute attention instead of storing it. Trades compute for memory (IO is the bottleneck anyway).</div>
  </div>
</div>
</div>

Flash Attention reduces memory from O(T²) to O(T) and is 2–4× faster than unfused attention on long sequences.

## torch.compile

PyTorch's compiler automatically fuses operations and generates optimized kernels:

```python
import torch

model = MyTransformerBlock(d_model=4096, n_heads=32)
model = torch.compile(model, mode="reduce-overhead")

# First call triggers compilation (slow)
# Subsequent calls use compiled, fused kernels
output = model(input_tensor)
```

`torch.compile` with Inductor backend:
- Fuses element-wise operations (LayerNorm components, activations, residuals)
- Generates Triton kernels for fused operations
- Doesn't replace Flash Attention (that's already hand-optimized)
- 10–30% speedup on typical transformer workloads

## What's Next

Single-GPU performance has limits. The next chapter covers **distributed training** — how to split work across hundreds or thousands of GPUs using data, tensor, and pipeline parallelism.

[← Previous: Chapter 20 — GPU Architecture](./20_gpu_architecture.md) · **Next: [Chapter 22 — Distributed Training →](./22_distributed_training.md)**

---

*Last updated: April 2026*
