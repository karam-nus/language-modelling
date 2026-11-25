---
title: "Chapter 25 — CUDA & Kernel Development"
description: "The CUDA model from threads to grids, memory coalescing and occupancy, why fusion matters, tiled matmul, Triton, the online-softmax trick behind FlashAttention verified in NumPy, torch.compile, and the 2025–26 kernel ecosystem: CUTLASS/CuTe DSL, ThunderKittens, TileLang, DeepSeek's FlashMLA/DeepGEMM/DeepEP, Liger, and LLM-written kernels."
---

[← Back to Table of Contents](./README.md)

# Chapter 25 — CUDA & Kernel Development

> **The one-sentence version:** a fast kernel moves each byte between HBM and the chip as few times as possible and keeps the tensor cores fed — everything else (tiling, fusion, online softmax, warp specialisation) is a technique for doing that.

## The CUDA Programming Model

CUDA organizes computation into a hierarchy:

<div class="diagram">
<div class="diagram-title">CUDA Thread Hierarchy</div>
<div class="layer-stack">
  <div class="layer accent">Grid — All threads launched by a single kernel call</div>
  <div class="layer purple">Block — Up to 1,024 threads that share shared memory and can synchronise. Runs entirely on one SM (an SM can host several blocks).</div>
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

This kernel is **memory-bound**: each thread does 1 add but reads 2 floats and writes 1 float (12 bytes for 1 FLOP — an intensity of 0.08, thousands of times below an H100's ridge point). Two rules make memory-bound kernels as fast as they can be:

- **Coalescing**: consecutive threads of a warp should touch consecutive addresses, so the 32 loads merge into a few wide memory transactions. `a[idx]` with `idx = blockIdx.x * blockDim.x + threadIdx.x` does exactly that; a strided pattern such as `a[idx * 32]` can be many times slower.
- **Occupancy and latency hiding**: an HBM load takes hundreds of cycles. Each SM keeps many warps resident and switches to another warp while one waits, so enough independent work must be in flight. Registers and shared memory per block limit how many warps fit.

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

This is why `torch.compile` and Triton exist — they fuse element-wise operations automatically — and why hand-fused kernels for the hottest paths (attention, RMSNorm + residual, SwiGLU, rotary embeddings, cross-entropy over a large vocabulary) are standard in training and serving stacks.

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

Each element of A and B is loaded from HBM once per tile but reused TILE_SIZE times from shared memory, cutting HBM traffic by about TILE_SIZE×. (This teaching version assumes M, N and K are multiples of TILE_SIZE and uses CUDA cores; production GEMMs use tensor-core instructions, much larger tiles, multi-stage asynchronous pipelines and swizzled shared-memory layouts — which is why you call cuBLAS or CUTLASS rather than writing your own.)

## Triton: GPU Kernels in Python

**Triton** (OpenAI) lets you write GPU kernels in Python, with automatic tiling and shared memory management:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Fused row softmax: one program per row; the whole row fits in one block."""
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

FlashAttention ([Chapter 4](./04_attention_sdpa_and_mha.md)) is the most important fused kernel in LLM training and inference. The enabling trick is **online softmax**: you can compute softmax(scores)·V while streaming over blocks of keys, keeping only a running maximum, a running sum and a running output, rescaling them whenever a new block raises the maximum:

```python
import numpy as np

def attention_reference(q, K, V):
    s = K @ q                                   # [T] scores
    p = np.exp(s - s.max()); p /= p.sum()
    return p @ V                                # [d]

def attention_online(q, K, V, block=4):
    """One query, keys/values streamed in blocks: never holds all T scores at once."""
    m, l, acc = -np.inf, 0.0, np.zeros(V.shape[1])
    for i in range(0, len(K), block):
        s = K[i:i + block] @ q                  # scores for this block only
        m_new = max(m, s.max())
        scale = np.exp(m - m_new)               # rescale what we accumulated so far
        p = np.exp(s - m_new)
        l = l * scale + p.sum()
        acc = acc * scale + p @ V[i:i + block]
        m = m_new
    return acc / l

rng = np.random.default_rng(0)
q, K, V = rng.standard_normal(8), rng.standard_normal((32, 8)), rng.standard_normal((32, 8))
print(np.allclose(attention_reference(q, K, V), attention_online(q, K, V)))   # True
```

The streamed result matches the reference to machine precision (max difference ~2 × 10⁻¹⁶). A FlashAttention kernel does the same for a whole tile of queries at once, on tensor cores, with K/V tiles double-buffered in shared memory. Its ingredients:

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

FlashAttention reduces attention's memory from O(T²) to O(T) and is several times faster than unfused attention on long sequences. FlashAttention-3 adds Hopper-specific **warp specialisation** — producer warps issue TMA copies while consumer warpgroups run `wgmma` matrix instructions, overlapping data movement, matmuls and softmax — and FlashAttention-4 retargets the design to Blackwell's tensor memory.

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

`torch.compile` with the Inductor backend:
- captures the graph and fuses element-wise and reduction operations (norms, activations, residual adds);
- generates Triton kernels for them, and can use CUDA graphs (`mode="reduce-overhead"`) to remove per-kernel launch overhead — important for small-batch decode, where launches can dominate;
- calls into existing hand-written kernels (cuBLAS GEMMs, FlashAttention via SDPA) rather than replacing them;
- typically gives 10–30% on transformer training and more on overhead-bound inference.

## The 2025–26 Kernel Ecosystem

| Layer | Tools | What they are for |
|---|---|---|
| Vendor libraries | cuBLAS, cuDNN, NCCL; ROCm's hipBLASLt, RCCL | GEMMs, attention, collectives — call these first |
| Templates and DSLs | **CUTLASS 3/4 + CuTe**, **CuTe DSL** (Python, 2025), **ThunderKittens**, **TileLang** | Write near-peak tensor-core kernels for Hopper/Blackwell without raw PTX |
| Python kernel languages | **Triton** (and its lower-level Gluon dialect), Pallas (TPU/JAX), MLX custom kernels (Apple) | Fast iteration; fused element-wise and attention variants |
| Serving kernels | FlashAttention 2/3/4, **FlashInfer**, **FlashMLA**, Marlin/Machete | Paged attention, MLA decode, low-bit GEMMs |
| Open lab releases | DeepSeek (Feb 2025): **FlashMLA**, **DeepGEMM** (FP8 GEMMs with fine-grained scaling), **DeepEP** (expert-parallel all-to-all) | Production kernels behind V3/R1 serving |
| Training kernels | **Liger Kernel**, fused cross-entropy (Cut Cross-Entropy), TransformerEngine | Lower activation memory and higher MFU |

<div class="callout trend">
<span class="callout-title">2026 trend — kernels written by models</span>
<p>Benchmarks such as KernelBench (2025) measure whether LLMs can write correct, fast GPU kernels, and labs use agentic search loops to generate and tune kernels. Results are real but uneven, and early claims of huge speed-ups were sometimes traced to kernels that exploited flaws in the evaluation harness — a reward-hacking story (Chapter 15) in miniature. Always verify a generated kernel's correctness on adversarial inputs.</p>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>FlashAttention required no new mathematics — only a reordering of an existing computation so that it fits the memory hierarchy. Which other operation in an LLM (cross-entropy over a 256K vocabulary? MoE routing and all-to-all? sampling?) still materialises something large it doesn't need to?</p>
</div>

## Key Takeaways

- Grid → block (one SM, shared memory) → warp (32 threads, SIMT) → thread (registers).
- Memory-bound kernels live or die by coalescing and occupancy; compute-bound ones by tiling and tensor-core utilisation.
- Fusion keeps intermediates on chip; online softmax lets attention stream over keys without materialising T×T scores.
- Use vendor libraries and existing kernels first; reach for Triton, CuTe DSL or ThunderKittens when you need something new.
- The open kernel ecosystem (FlashInfer, FlashMLA, DeepGEMM, DeepEP, Liger) is where serving and training speed-ups ship in 2025–26.

## Check Yourself

<details class="check">
<summary>A kernel reads a 4096 × 4096 BF16 matrix and writes one of the same size, doing one FLOP per element. What is its minimum time on an H100?</summary>
<p>It moves 2 × 33.5 MB = 67 MB; at 3.35 TB/s that is ~20 µs. The 16.8 MFLOP of arithmetic would take ~0.02 µs on tensor cores — the kernel is entirely bandwidth-bound.</p>
</details>

<details class="check">
<summary>In online softmax, why must the running sum and output be multiplied by exp(m_old − m_new) when a new block raises the maximum?</summary>
<p>Earlier terms were exponentiated relative to the old maximum, e^(s − m_old). To express them relative to the new maximum, multiply by e^(m_old − m_new). Without the rescale, earlier blocks would be over-weighted.</p>
</details>

<details class="check">
<summary>Why does CUDA-graph capture help small-batch decode much more than training?</summary>
<p>A decode step launches hundreds of small kernels that each finish in microseconds, so CPU launch overhead (a few µs per launch) can rival the GPU work. Replaying a captured graph launches them all at once. Training kernels are large enough that launch overhead is negligible.</p>
</details>

<details class="check">
<summary>Why are threads in a warp ideally given consecutive memory addresses?</summary>
<p>The memory system serves a warp's loads in wide aligned transactions (e.g., 32–128 bytes). Consecutive addresses coalesce into a few transactions; scattered addresses need one transaction each, multiplying traffic and latency.</p>
</details>

<details class="check">
<summary>What did DeepEP add that standard collectives libraries lacked for MoE models?</summary>
<p>All-to-all dispatch and combine kernels specialised for expert parallelism — routing each token's hidden state to the GPUs holding its selected experts and back — with low-latency modes for decode and overlap with computation, across NVLink and RDMA.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| CUDA C++ Programming Guide | NVIDIA | [docs.nvidia.com/cuda/cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Execution model, coalescing, occupancy |
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | Tillet et al., 2019 | [doi.org/10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973) | Triton |
| Online normalizer calculation for softmax | Milakov & Gimelshein, 2018 | [arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867) | Online softmax |
| FlashAttention / FlashAttention-3 | Dao et al., 2022; Shah et al., 2024 | [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) · [arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608) | Tiling, recomputation, warp specialisation |
| ThunderKittens | Spector et al., 2024 | [arxiv.org/abs/2410.20399](https://arxiv.org/abs/2410.20399) | Tile-based kernel DSL |
| FlashInfer | Ye et al., 2025 | [arxiv.org/abs/2501.01005](https://arxiv.org/abs/2501.01005) | Serving attention kernels |
| DeepSeek open-source week (FlashMLA, DeepEP, DeepGEMM) | DeepSeek, 2025 | [github.com/deepseek-ai](https://github.com/deepseek-ai) | Production MoE/MLA kernels |
| CUTLASS and CuTe DSL | NVIDIA, 2025 | [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | Templates and Python DSL for tensor cores |
| KernelBench | Ouyang et al., 2025 | [arxiv.org/abs/2502.10517](https://arxiv.org/abs/2502.10517) | Evaluating LLM-written kernels |
| torch.compile tutorial | PyTorch | [docs.pytorch.org/tutorials](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | Inductor fusion, CUDA graphs |

---

**Next:** [Chapter 26 — Distributed Training](./26_distributed_training.md) — when one GPU is not enough, which is always.

[← Back to Table of Contents](./README.md)

## What's Next

Single-GPU performance has limits. The next chapter covers **distributed training** — how to split work across hundreds or thousands of GPUs using data, tensor, and pipeline parallelism.

[← Previous: Chapter 24 — GPU Architecture for ML](./24_gpu_architecture.md) · **Next: [Chapter 26 — Distributed Training →](./26_distributed_training.md)**

---

*Last updated: April 2026*
