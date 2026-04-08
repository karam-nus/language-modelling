[← Back to Table of Contents](./README.md)

# Chapter 23 — ASICs & Accelerators

> *"NVIDIA dominates, but the AI hardware landscape is diversifying. Specialized silicon — TPUs, LPUs, and custom accelerators — are reshaping what's possible."*

## The AI Accelerator Landscape

<div class="diagram">
<div class="diagram-title">AI Hardware Ecosystem</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">GPU (General Purpose)</div>
    <div class="card-desc">NVIDIA (H100, B200), AMD (MI300X). Flexible, massive ecosystem. Dominant for both training and inference.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">TPU (Google)</div>
    <div class="card-desc">Systolic arrays optimized for matmuls. Tightly integrated with JAX/XLA. Powers Gemini and PaLM training.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Custom ASICs</div>
    <div class="card-desc">Groq LPU, Intel Gaudi, AWS Trainium/Inferentia, Cerebras WSE. Each designed for specific workload patterns.</div>
  </div>
</div>
</div>

## Google TPUs

TPUs (Tensor Processing Units) are Google's in-house accelerators, designed from the ground up for neural network workloads.

### Architecture

TPUs use **systolic arrays** — a grid of multiply-accumulate units that data flows through rhythmically, performing matmuls with maximal data reuse and minimal memory access:

<div class="diagram">
<div class="diagram-title">TPU Systolic Array (Simplified)</div>
<div class="flow">
  <div class="flow-step accent">Weight matrix loaded into the 128×128 systolic array</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">Input activations stream in from the left, row by row</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">Each cell: multiply input × weight, add to accumulated sum, pass both values to neighbors</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step orange">Output activations stream out from the bottom after N cycles</div>
</div>
</div>

### TPU Generations

| TPU Version | Year | BF16 TFLOPS | HBM | HBM BW | Interconnect |
|-------------|------|------------|-----|---------|-------------|
| TPU v2 | 2017 | 46 | 8 GB | 700 GB/s | — |
| TPU v3 | 2018 | 123 | 16 GB | 900 GB/s | — |
| TPU v4 | 2022 | 275 | 32 GB | 1,200 GB/s | ICI (optical) |
| TPU v5e | 2023 | 197 | 16 GB | 819 GB/s | ICI |
| TPU v5p | 2023 | 459 | 95 GB | 2,765 GB/s | ICI |
| TPU v6e (Trillium) | 2024 | 918 | 32 GB | 1,640 GB/s | ICI |

TPUs are deployed in **pods** — tightly interconnected clusters connected via Inter-Chip Interconnect (ICI). A TPU v5p pod has 8,960 chips. Google used 16,384 TPU v5p chips to train Gemini.

### TPU Programming Model

```python
# JAX: NumPy-like API that compiles to XLA for TPUs
import jax
import jax.numpy as jnp
from jax import pmap  # parallel map across devices

# Automatically shard computation across TPU cores
@pmap
def train_step(params, batch):
    def loss_fn(p):
        logits = model_forward(p, batch['input_ids'])
        return cross_entropy(logits, batch['labels'])
    
    grads = jax.grad(loss_fn)(params)
    grads = jax.lax.pmean(grads, axis_name='batch')  # all-reduce
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params

# With jax.sharding for FSDP-style training
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
mesh = Mesh(jax.devices(), ('dp', 'tp'))
```

## AMD GPUs

AMD's MI300X directly competes with NVIDIA's H100/H200:

| Feature | NVIDIA H100 SXM | AMD MI300X |
|---------|-----------------|------------|
| BF16 TFLOPS | 990 | 1,307 |
| HBM | 80 GB (HBM3) | 192 GB (HBM3) |
| HBM Bandwidth | 3.35 TB/s | 5.3 TB/s |
| TDP | 700W | 750W |
| Software | CUDA (dominant) | ROCm (growing) |

AMD's advantage: **2.4× more HBM** at competitive bandwidth. This matters for inference where memory capacity determines the largest model you can serve. The software gap (ROCm vs CUDA) remains the primary barrier to adoption, but PyTorch and JAX now support ROCm.

## Intel Gaudi

Intel Gaudi accelerators target training and inference with a focus on cost efficiency:

<div class="diagram">
<div class="diagram-title">Gaudi Architecture</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Matrix Math Engine (MME)</div>
    <div class="card-desc">Dedicated engines for matmuls (GEMMs). Optimized for BF16 and FP8.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Tensor Processing Cores (TPC)</div>
    <div class="card-desc">Programmable VLIW cores for element-wise operations, activations, and custom ops. Flexible like GPU CUDA cores.</div>
  </div>
</div>
</div>

| Feature | Gaudi 2 | Gaudi 3 |
|---------|---------|---------|
| BF16 TFLOPS | 432 | 1,835 |
| HBM | 96 GB (HBM2e) | 128 GB (HBM2e) |
| FP8 TFLOPS | 865 | 3,670 |
| Form factor | OAM | OAM |

Gaudi uses standard Ethernet (RoCE) for inter-chip communication rather than proprietary interconnects, reducing networking costs for training clusters.

## Groq LPU

The **Language Processing Unit (LPU)** by Groq takes a radically different approach:

<div class="diagram">
<div class="diagram-title">Groq LPU — Deterministic Architecture</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">GPU Approach</div>
    <ul>
      <li>Dynamic scheduling at runtime</li>
      <li>Cache hierarchy (L1→L2→HBM)</li>
      <li>Memory-bound during autoregressive decode</li>
      <li>Variable latency per token</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">LPU Approach</div>
    <ul>
      <li>Schedule computed at compile time (TSP architecture)</li>
      <li>230 MB SRAM, no HBM — no memory bandwidth bottleneck</li>
      <li>Deterministic execution — no cache misses</li>
      <li>Consistent, ultra-low latency per token</li>
    </ul>
  </div>
</div>
</div>

Groq achieves ~500 tokens/sec on LLaMA-2 70B (across 8 LPUs), with single-digit ms latency per token. The tradeoff: limited SRAM means model weights must be distributed across many chips. Inference only — not designed for training.

## AWS Trainium & Inferentia

Amazon's custom chips, integrated directly with AWS services:

| Chip | Purpose | Compute (BF16) | HBM | Notes |
|------|---------|----------------|-----|-------|
| Inferentia 1 | Inference | 128 TOPS (INT8) | — | Uses on-chip memory |
| Inferentia 2 | Inference | 380 TFLOPS | 32 GB | NeuronLink interconnect |
| Trainium 1 | Training | 210 TFLOPS | 32 GB | Used in Trn1 instances |
| Trainium 2 | Training | 756 TFLOPS | 96 GB | 4× Trainium 1 perf |

Programmed via AWS Neuron SDK, which compiles PyTorch models to run on Trainium/Inferentia. Used by Anthropic to train Claude models.

## Cerebras WSE

The **Wafer-Scale Engine (WSE-3)** is a single massive chip occupying an entire silicon wafer:

- 4 trillion transistors (56× more than H100)
- 900,000 AI cores
- 44 GB on-chip SRAM
- 21 PB/s memory bandwidth (on-chip)

The entire model fits on-chip, eliminating the memory wall entirely. Cerebras targets training large models without the complexity of distributed parallelism.

## Apple Silicon

Increasingly relevant for local inference and research:

| Chip | GPU TFLOPS (FP16) | Unified Memory | Memory BW |
|------|-------------------|----------------|-----------|
| M1 Ultra | 21 | 128 GB | 800 GB/s |
| M2 Ultra | 27 | 192 GB | 800 GB/s |
| M3 Max | 14.2 | 128 GB | 400 GB/s |
| M4 Max | 18 | 128 GB | 546 GB/s |
| M4 Ultra | ~36 | 256 GB | 819 GB/s |

**Unified memory** means the CPU and GPU share the same memory — no PCIe transfer overhead. The M4 Ultra with 256 GB can run LLaMA-3 70B at 4-bit quantization entirely in memory. Tools like `llama.cpp` with Metal backend and MLX are optimized for Apple Silicon.

## Choosing Hardware

<div class="diagram">
<div class="diagram-title">Hardware Decision Matrix</div>

| Use Case | Best Options | Why |
|----------|-------------|-----|
| **Frontier training** | H100/B200, TPU v5p | Proven at scale, mature software |
| **Cost-efficient training** | Gaudi 3, Trainium 2 | Lower $/TFLOP, cloud-integrated |
| **High-throughput inference** | H100/H200, MI300X | Large HBM, ecosystem support |
| **Low-latency inference** | Groq LPU, Inferentia 2 | Deterministic, high tokens/sec |
| **Local development** | Apple M-series, RTX 4090 | Accessible, unified memory (Apple) |
| **Research prototyping** | TPU (free via Colab/Kaggle) | Free compute for experiments |
</div>

## What's Next

With hardware and systems covered, we move to the **software ecosystem**. The next chapter introduces the **Hugging Face ecosystem** — the libraries, hubs, and tools that connect models to practitioners.

[← Previous: Chapter 22 — Distributed Training](./22_distributed_training.md) · **Next: [Chapter 24 — The Hugging Face Ecosystem →](./24_huggingface_ecosystem.md)**

---

*Last updated: April 2026*
