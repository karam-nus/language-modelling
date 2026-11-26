---
title: "Chapter 24 — GPU Architecture for ML"
description: "What an LLM practitioner needs to know about GPUs: SMs, warps and tensor cores, the memory hierarchy with real bandwidths, the roofline model and ridge points, why decode is memory-bound and prefill compute-bound, worked FLOP and byte counts for Llama 3.1 8B, and dense (not sparse) specs from A100 to Rubin, MI355X and TPU Ironwood."
related:
  - { id: chips, why: "The complete guide to the silicon: GPUs, memory and interconnects." }
  - { id: nvidia, why: "The NVIDIA hardware and software stack." }
---

[← Back to Table of Contents](./README.md)

# Chapter 24 — GPU Architecture for ML

Every performance number in this guide — tokens per second, MFU, the speed-up from FP8 or speculative decoding — is a consequence of a few hardware facts: how many multiply-adds a chip can do per second, how fast it can move bytes, and how big each level of its memory is. This chapter gives you those facts and the one model (the roofline) that ties them together. The companion [Chips guide](https://karam-nus.github.io/chips/) goes all the way down to the transistor.

> **The one-sentence version:** a GPU is a huge array of matrix-multiply units fed by a memory system that is ~300× slower than the units can consume, so the question for every LLM operation is whether it does enough arithmetic per byte to keep them busy.

## Why GPUs for ML?

Deep learning is dominated by **matrix multiplications**, which split into millions of independent multiply-adds. GPUs spend their transistors on throughput — thousands of simple lanes and dedicated matrix units — where CPUs spend them on latency for a few complex threads.

| Feature | CPU (AMD EPYC 9654) | GPU (NVIDIA H100 SXM) |
|---------|:---:|:---:|
| Cores | 96 | 132 SMs: 16,896 FP32 lanes + 528 tensor cores |
| Clock | 2.4 GHz base | up to 1.98 GHz |
| 16-bit matrix throughput (dense) | ~15 TFLOP/s (AVX-512 BF16) | 989 TFLOP/s |
| Memory | up to 6 TB DDR5 | 80 GB HBM3 |
| Bandwidth | 461 GB/s | 3,350 GB/s |
| Power | 360 W | 700 W |

## NVIDIA GPU Anatomy

<div class="diagram">
<div class="diagram-title">GPU Architecture — Hierarchical View</div>
<div class="layer-stack">
  <div class="layer accent">GPU — 132 SMs (H100 SXM), a 50 MB L2 cache, HBM stacks, NVLink</div>
  <div class="layer purple">GPC — a cluster of SMs; Hopper lets a thread-block cluster share SMs' shared memory</div>
  <div class="layer green">SM — the compute unit: 4 sub-partitions, each with a warp scheduler, 32 FP32 lanes and 1 tensor core</div>
  <div class="layer orange">Warp — 32 threads issued together (SIMT); the unit of scheduling</div>
  <div class="layer cyan">Thread — one lane's view: its own registers</div>
</div>
</div>

### The Streaming Multiprocessor (H100)

- **128 FP32 lanes** ("CUDA cores") for general arithmetic.
- **4 fourth-generation tensor cores**, each performing 512 dense FP16/BF16 multiply-adds per clock (1,024 for FP8). The tensor cores supply about 15× the FP32 lanes' throughput — essentially all of an LLM's FLOPs.
- **256 KB register file** (64K 32-bit registers).
- **256 KB of SRAM** split between L1 cache and programmer-managed **shared memory** (up to 228 KB).
- **TMA** (tensor memory accelerator) for asynchronous bulk copies from HBM into shared memory, and warp-group MMA instructions (`wgmma`) — the features FlashAttention-3 exploits ([Chapter 25](./25_cuda_and_kernel_development.md)).

Blackwell (B200) adds **tensor memory** — a dedicated store for matrix-unit operands — native FP4/FP6 and microscaling formats, and two dies acting as one GPU.

## Memory Hierarchy

<div class="img-caption">
  <img src="{{ '/assets/images/gpu_memory_hierarchy.svg' | relative_url }}" alt="GPU memory hierarchy pyramid showing registers, L1/Shared, L2 cache, and HBM with bandwidth and latency numbers for H100">
  <figcaption>H100 memory hierarchy — from per-thread registers to off-chip HBM3</figcaption>
</div>

| Level (H100 SXM) | Capacity | Aggregate bandwidth | Scope |
|---|---|---|---|
| Registers | 256 KB per SM (33 MB total) | far above shared memory | One thread |
| Shared memory / L1 | 256 KB per SM (33 MB total) | ~30 TB/s | One thread block (or cluster) |
| L2 cache | 50 MB | ~10 TB/s (order of) | Whole GPU |
| HBM3 | 80 GB | 3.35 TB/s | Whole GPU — weights, KV cache, activations |
| Host DRAM via PCIe 5.0 ×16 | hundreds of GB–TB | ~64 GB/s per direction | Offloading |
| Peer GPUs via NVLink 4 | — | 900 GB/s per GPU (both directions) | Tensor/expert parallelism |

Two gaps matter. On-chip SRAM is only about **10× faster** than HBM but **~2,500× smaller** (33 MB vs 80 GB): you cannot keep a model on chip, you can only keep the *current tile* there. And HBM is ~50× faster than PCIe: anything offloaded to the CPU is very slow to bring back. **Kernel fusion** and **tiling** — doing as much work as possible on a tile while it sits in SRAM — exist because of the first gap; offloading strategies are bounded by the second.

## Compute-Bound vs Memory-Bound: The Roofline

**Arithmetic intensity** = FLOPs performed / bytes moved from memory. A chip's **ridge point** = peak FLOP/s / peak bytes/s. Below the ridge, an operation is limited by bandwidth; above it, by compute:

$$\text{attainable FLOP/s} = \min(\text{peak FLOP/s},\; \text{intensity} \times \text{bandwidth})$$

| Chip | Dense BF16 | Bandwidth | Ridge (BF16) | Ridge (FP8) |
|---|---|---|---|---|
| A100 80GB | 312 TFLOP/s | 2.04 TB/s | 153 FLOP/B | — |
| H100 SXM | 989 | 3.35 | 295 | 591 |
| B200 | 2,250 | 8.0 | 281 | 563 (FP4: 1,125) |
| MI355X | 2,500 | 8.0 | 313 | 625 |

<div class="diagram">
<div class="diagram-title">Compute-Bound vs Memory-Bound Operations</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Compute-bound (above the ridge)</div>
    <ul>
      <li>Large matmuls: training, prefill, large-batch decode</li>
      <li>Intensity ≈ number of tokens processed together</li>
      <li>Faster with more FLOPs: newer tensor cores, FP8/FP4</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Memory-bound (below the ridge)</div>
    <ul>
      <li>Small-batch decode, attention over a long KV cache, norms, activations</li>
      <li>Intensity ≈ 1–2 FLOP per byte</li>
      <li>Faster with fewer bytes: quantization, fusion, more bandwidth</li>
    </ul>
  </div>
</div>
</div>

**Decode at small batch is deeply memory-bound.** A linear layer applied to B tokens reads each 2-byte weight once and uses it for B multiply-adds (2B FLOPs), so its intensity is about **B FLOPs per byte**. At batch 1 that is ~1 — three hundred times below an H100's ridge. You need a batch of roughly 300 tokens in flight per weight read before BF16 decode becomes compute-bound, which is why serving systems batch aggressively (Chapters 17, 18, 31).

Move the operations around the roofline yourself:

<div class="lab" data-lab="roofline"><noscript>Interactive lab: a roofline plot with LLM operations (decode GEMV, batched decode, prefill GEMM, attention) on selectable GPUs and precisions. Enable JavaScript to use it.</noscript></div>

## FLOPs and Bytes for a Matrix Multiply

For $C = A \times B$ with $A: [M, K]$ and $B: [K, N]$: **FLOPs = 2MNK** (a multiply and an add per term).

Llama 3.1 8B, one layer, BF16 weights, on an H100:

| Operation | M (tokens) | K | N | FLOPs | Weight bytes | Time if compute-bound | Time if bandwidth-bound |
|-----------|:-:|:-:|:-:|:-----:|:---:|:---:|:---:|
| QKV projection, prefill 4,096 tokens | 4,096 | 4,096 | 6,144 (32 Q + 8 K + 8 V heads × 128) | 206 GFLOP | 50 MB | 0.21 ms | 0.015 ms |
| SwiGLU FFN, prefill 4,096 tokens | 4,096 | 4,096 | 2 × 14,336 up, then 14,336 → 4,096 down | 1,443 GFLOP | 352 MB | 1.46 ms | 0.11 ms |
| QKV projection, decode 1 token | 1 | 4,096 | 6,144 | 50 MFLOP | 50 MB | 0.00005 ms | 0.015 ms |

For prefill the compute time dominates — the operation is compute-bound. For decode, reading 50 MB of weights takes 300× longer than the arithmetic: the tensor cores idle ~99.7% of the time unless many sequences share the read.

## Key Accelerator Comparison

Dense peak throughput (no 2:4-sparsity bonus — vendors often quote sparse figures, which are exactly 2× these), per chip, from the Chips guide's sourced data file:

| Accelerator | Year | Memory | Bandwidth | BF16 | FP8 | FP4 | Scale-up link |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RTX 4090 | 2022 | 24 GB GDDR6X | 1.0 TB/s | 330 | 661 | — | PCIe 4.0 |
| RTX 5090 | 2025 | 32 GB GDDR7 | 1.8 TB/s | 419 | 838 | 1,676 | PCIe 5.0 |
| A100 SXM 80GB | 2020 | 80 GB HBM2e | 2.04 TB/s | 312 | — | — | NVLink 600 GB/s |
| H100 SXM | 2022 | 80 GB HBM3 | 3.35 TB/s | 989 | 1,979 | — | NVLink 900 GB/s |
| H200 SXM | 2024 | 141 GB HBM3e | 4.8 TB/s | 989 | 1,979 | — | NVLink 900 GB/s |
| B200 (HGX) | 2024 | 180 GB HBM3e | 8.0 TB/s | 2,250 | 4,500 | 9,000 | NVLink 1.8 TB/s |
| B300 (Blackwell Ultra) | 2025 | 288 GB HBM3e | 8.0 TB/s | 2,250 | 4,500 | 15,000 | NVLink 1.8 TB/s |
| Rubin (VR200) | 2026, ramping | 288 GB HBM4 | 22 TB/s | — | — | 35,000 | NVLink 3.6 TB/s |
| AMD MI300X | 2023 | 192 GB HBM3 | 5.3 TB/s | 1,307 | 2,615 | — | Infinity Fabric 896 GB/s |
| AMD MI355X | 2025 | 288 GB HBM3e | 8.0 TB/s | 2,500 | 5,000 | 10,000 | 1,075 GB/s |
| Google TPU v7 Ironwood | 2025 | 192 GB HBM3e | 7.37 TB/s | 2,307 | 4,614 | — | ICI 1.2 TB/s |

(TFLOP/s. "—" = not supported or not published. Rubin figures are NVIDIA's announced numbers.)

The trend is the opposite of what intuition suggests: **compute grows faster than bandwidth** — peak low-precision throughput rose ~7.6× from H100 (FP8) to B300 (FP4) while bandwidth rose 2.4× — so ridge points rise and more workloads become memory-bound each generation. That "memory wall" is why every chapter from 17 to 22 is about bytes, and why HBM capacity and bandwidth are the headline features of each new part.

## Tensor Cores

Tensor cores execute small matrix multiply-accumulates (warp- or warpgroup-sized tiles) in hardware. On an H100 they provide:

| Format | H100 SXM dense throughput | vs FP32 lanes (67 TFLOP/s) |
|-----------|:---:|:---:|
| TF32 | 495 TFLOP/s | 7× |
| BF16 / FP16 | 989 TFLOP/s | 15× |
| FP8 | 1,979 TFLOP/s | 30× |
| INT8 | 1,979 TOP/s | 30× |

To use them well, matrix dimensions should be multiples of the tile sizes (16, 64, 128) — which is why model widths are multiples of 128 and vocabularies are padded (Qwen3's 151,936 = 1,187 × 128).

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Ridge points keep rising: an H100 needs ~300 FLOPs per byte, a B300 at FP4 nearly 2,000. Decode at batch 1 offers ~1. If chips keep adding FLOPs faster than bandwidth, which of today's inference tricks — batching, speculative decoding, MLA's absorbed attention, multi-token prediction — become <em>more</em> valuable, and why?</p>
</div>

## Key Takeaways

- An SM = warp schedulers + FP32 lanes + tensor cores + registers + shared memory; tensor cores provide essentially all LLM FLOPs.
- On-chip SRAM is ~10× faster but ~2,500× smaller than HBM — hence tiling and fusion.
- Roofline: attainable = min(peak, intensity × bandwidth); H100's BF16 ridge is ~295 FLOP/byte.
- Linear layers have intensity ≈ tokens per weight read: decode at small batch is memory-bound, prefill compute-bound.
- Quote dense, per-chip numbers (H100 BF16 = 989, not 1,979); compute is outgrowing bandwidth every generation.

## Check Yourself

<details class="check">
<summary>What batch size makes BF16 decode compute-bound on an H100, ignoring the KV cache?</summary>
<p>Intensity ≈ B FLOP/byte for 2-byte weights; the ridge is 989 / 3.35 ≈ 295, so B ≈ 300 tokens per step. With FP8 weights and FP8 math the ridge doubles to ~590 but each byte carries twice as many weights, so B stays ≈ 300.</p>
</details>

<details class="check">
<summary>A vendor slide says "H100: 3,958 TFLOPS FP8". What is the dense figure, and why does the difference matter?</summary>
<p>1,979 TFLOP/s — the slide quotes 2:4 structured-sparsity throughput, which needs half the weights to be zero in a specific pattern. Dense models get the dense number; using the sparse one in MFU or time estimates is off by 2×.</p>
</details>

<details class="check">
<summary>How long does one decode step of Llama 3.1 70B (BF16) take at minimum on 4 × H100 with tensor parallelism?</summary>
<p>Each GPU reads a quarter of 141 GB ≈ 35 GB at 3.35 TB/s ≈ 10.5 ms, so at most ~95 tokens/s per sequence (before communication and kernel overheads, and ignoring the KV cache).</p>
</details>

<details class="check">
<summary>Why are prefill FLOPs per layer proportional to the prompt length while decode FLOPs per step are constant (for the linear layers)?</summary>
<p>Prefill multiplies an [M, d] activation matrix by each weight, M = prompt tokens. Decode multiplies a [1, d] (or [B, d]) row. The weight bytes read are the same in both cases, which is exactly why their arithmetic intensities differ by a factor of M.</p>
</details>

<details class="check">
<summary>Why are model dimensions and vocabularies padded to multiples of 64 or 128?</summary>
<p>Tensor-core instructions and kernel tiles operate on fixed tile sizes; a dimension that is not a multiple wastes part of a tile (or forces a slow edge path), so padding costs a few unused parameters and gains full-speed kernels.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| NVIDIA H100 Tensor Core GPU Architecture (whitepaper) | NVIDIA, 2022 | [resources.nvidia.com](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) | SM structure, TMA, dense throughput |
| NVIDIA Blackwell architecture | NVIDIA, 2024–25 | [nvidia.com/en-us/data-center/technologies/blackwell-architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) | Tensor memory, FP4 |
| Roofline: An Insightful Visual Performance Model | Williams, Waterman & Patterson, 2009 | [doi.org/10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) | The roofline model |
| Making Deep Learning Go Brrrr From First Principles | He, 2022 | [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html) | Compute vs bandwidth vs overhead |
| Chips guide — accelerator data and Chapters 11, 15 | karam-nus, 2026 | [karam-nus.github.io/chips](https://karam-nus.github.io/chips/) | Sourced dense specs; memory wall |
| AMD Instinct MI355X / Google Ironwood specifications | AMD, Google, 2025 | [amd.com](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) | Non-NVIDIA comparison |

---

**Next:** [Chapter 25 — CUDA & Kernel Development](./25_cuda_and_kernel_development.md) — writing the code that runs on all of this.

[← Back to Table of Contents](./README.md)
