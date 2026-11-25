---
title: "Chapter 27 — ASICs & Specialized Accelerators"
description: "The non-NVIDIA-GPU world as of late 2026: Google TPUs through Ironwood and TPU 8, AMD Instinct, AWS Trainium, Groq's SRAM-only LPU (now licensed to NVIDIA), Cerebras wafer-scale, Microsoft Maia and other in-house chips, Apple Silicon for local inference, the software stacks that decide what is usable, and how to choose."
---

[← Back to Table of Contents](./README.md)

# Chapter 27 — ASICs & Specialized Accelerators

NVIDIA GPUs train and serve most LLMs, but not all of them: Gemini is trained on TPUs, Claude on TPUs and Trainium as well as GPUs, and the fastest public inference endpoints in 2024–25 ran on SRAM-only chips. This chapter surveys the alternatives from an LLM practitioner's point of view; the [Chips guide](https://karam-nus.github.io/chips/) (Chapters 16, 21, 27) covers their internals.

> **The one-sentence version:** every accelerator is a bet on which resource LLMs will be short of — matmul throughput (TPUs, Trainium), memory bandwidth (SRAM-only and wafer-scale designs), memory capacity (AMD's big HBM stacks) — and its software stack decides whether that bet is usable.

## The Landscape

<div class="diagram">
<div class="diagram-title">AI Accelerator Ecosystem (late 2026)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">GPUs</div>
    <div class="card-desc">NVIDIA (Hopper, Blackwell, Rubin ramping), AMD Instinct (MI300X → MI355X, MI455X announced). Flexible, programmable, the default.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Hyperscaler ASICs</div>
    <div class="card-desc">Google TPU, AWS Trainium/Inferentia, Microsoft Maia, Meta MTIA, OpenAI–Broadcom. Built for their owners' models and clouds.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Specialist architectures</div>
    <div class="card-desc">Groq LPU (SRAM-only, now licensed to NVIDIA), Cerebras WSE (wafer-scale), SambaNova, Tenstorrent, Etched. Each bets on a different bottleneck.</div>
  </div>
</div>
</div>

## Google TPUs

TPUs are built around large **systolic arrays**: a grid of multiply-accumulate cells through which weights and activations flow rhythmically, so each value fetched from memory is reused across a whole row or column of cells.

<div class="diagram">
<div class="diagram-title">Systolic Array (simplified)</div>
<div class="flow">
  <div class="flow-step accent">Weights loaded into a 128×128 (v4–v5) or 256×256 (v6+) grid of MAC cells</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">Activations stream in from one edge, one wavefront per cycle</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">Each cell multiplies, adds to the partial sum passing through, and forwards its inputs to its neighbours</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step orange">Finished dot products leave from the opposite edge</div>
</div>
</div>

### TPU Generations (per chip, dense)

| TPU | Year | BF16 TFLOP/s | FP8 / FP4 | HBM | Bandwidth | Notes |
|-------------|------|------------|-----|---------|-------------|---|
| v4 | 2021 | 275 | — | 32 GB | 1.2 TB/s | Optical circuit-switched pods of 4,096 |
| v5e | 2023 | 197 | — | 16 GB | 0.82 TB/s | Cost-optimised inference and training |
| v5p | 2023 | 459 | — | 95 GB | 2.76 TB/s | Pods of 8,960 chips |
| v6e Trillium | 2024 | 918 | — | 32 GB | 1.64 TB/s | 256×256 MXUs |
| v7 Ironwood | 2025 | 2,307 | FP8 4,614 | 192 GB | 7.37 TB/s | Inference-first; pods up to 9,216 chips |
| TPU 8t / 8i | 2026, announced | — | FP4 12,600 / 10,100 | 216 / 288 GB | 6.5 / 8.6 TB/s | Split into training (8t) and inference (8i) parts |

TPUs connect through **ICI** links in 3D-torus pods with optical circuit switches, which is why Google can train on many thousands of chips without an InfiniBand-style fabric. Since 2025 TPUs are also sold beyond Google's own use: Anthropic agreed in October 2025 to use up to a million TPUs.

### Programming TPUs

TPUs are programmed through XLA, mainly via **JAX** (also PyTorch/XLA). Modern JAX expresses parallelism by annotating how arrays are sharded over a device mesh and letting the compiler insert the collectives:

```python
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((jax.device_count() // 4, 4), ("data", "model"))   # e.g. 4-way tensor parallel

@jax.jit
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return jax.tree.map(lambda p, g: p - 3e-4 * g, params, grads), loss

# Shard the batch over "data" and each weight matrix's output dim over "model";
# XLA inserts the all-reduces / all-gathers itself.
batch = jax.device_put(batch, NamedSharding(mesh, P("data", None)))
w = jax.device_put(w, NamedSharding(mesh, P(None, "model")))
```

Pallas lets you write custom TPU (and GPU) kernels in Python, the TPU counterpart of Triton.

## AMD Instinct

AMD competes head-on with NVIDIA, leading on memory capacity:

| Feature | NVIDIA H100 SXM | AMD MI300X | AMD MI355X | NVIDIA B300 |
|---------|-----------------|------------|---|---|
| Year | 2022 | 2023 | 2025 | 2025 |
| BF16 TFLOP/s (dense) | 989 | 1,307 | 2,500 | 2,250 |
| FP4 TFLOP/s | — | — | 10,000 | 15,000 |
| HBM | 80 GB | 192 GB | 288 GB | 288 GB |
| Bandwidth | 3.35 TB/s | 5.3 TB/s | 8.0 TB/s | 8.0 TB/s |
| Software | CUDA | ROCm | ROCm | CUDA |

More HBM per GPU means a large model fits on fewer GPUs — a 405B model in FP8 fits on two MI355X but needs six H100s. **ROCm** has matured (vLLM, SGLang and PyTorch support it as first-class targets), and OpenAI, Meta and others announced large MI-series deployments in 2025; the MI455X and AMD's "Helios" rack, announced for 2026, target NVIDIA's rack-scale systems directly. The remaining gap is kernel breadth: new attention variants, quantization formats and MoE kernels usually appear on CUDA first.

## AWS Trainium and Inferentia

| Chip | Year | Dense throughput | HBM | Bandwidth | Notes |
|------|---|----------------|-----|---|-------|
| Inferentia2 | 2023 | 190 TFLOP/s BF16 | 32 GB | 0.82 TB/s | Inference |
| Trainium2 | 2024 | 667 BF16 / 1,299 FP8 | 96 GB | 2.9 TB/s | Project Rainier: hundreds of thousands of chips for Anthropic |
| Trainium3 | 2025 | 2,520 FP8 | 144 GB | 4.9 TB/s | 3 nm; UltraServer racks |

Programmed through the AWS **Neuron SDK** (PyTorch/XLA and JAX front ends, plus the NKI kernel language). Trainium's pitch is price-performance inside AWS rather than peak speed per chip.

## Groq LPU: All-SRAM Inference

Groq's **Language Processing Unit** removes HBM entirely: each chip has 230 MB of on-chip SRAM at ~80 TB/s, and a compiler schedules every operation statically, so there are no caches and no runtime variability.

<div class="diagram">
<div class="diagram-title">GPU vs LPU for Decode</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">GPU</div>
    <ul>
      <li>Weights in HBM at 3–8 TB/s per chip</li>
      <li>Decode bound by HBM bandwidth</li>
      <li>Few chips hold a large model</li>
      <li>Dynamic scheduling, batching for efficiency</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">LPU</div>
    <ul>
      <li>Weights in SRAM at ~80 TB/s per chip</li>
      <li>Decode bandwidth multiplied across hundreds of chips</li>
      <li>230 MB per chip → a 70B model needs ~300+ chips</li>
      <li>Deterministic, very low per-token latency</li>
    </ul>
  </div>
</div>
</div>

Groq's early Llama 2 70B deployment used 576 chips across eight racks and served a few hundred tokens per second per user — far beyond GPU single-stream speeds at the time, at a high capital cost per model. In December 2025 **NVIDIA licensed Groq's inference technology** (a ~$20B non-exclusive deal, hiring its founder and much of the team); the resulting LPX racks are positioned as decode co-processors beside Rubin GPU racks. The SRAM-for-decode idea has gone mainstream.

## Cerebras Wafer-Scale Engine

**WSE-3** is one chip the size of a wafer: 4 trillion transistors, 900,000 cores, 44 GB of on-chip SRAM with 21 PB/s of aggregate bandwidth. For **training**, weights stream in from an external memory system (MemoryX) layer by layer, so model size is not limited by the 44 GB. For **inference**, models are split across several wafers with weights resident in SRAM — which is how Cerebras serves open models at thousands of tokens per second per user. The trade-offs are cost per system and a narrower software path than GPUs.

## In-House and Emerging Chips

| Chip | Owner | Role | Headline numbers |
|---|---|---|---|
| Maia 200 (Jan 2026) | Microsoft | Azure inference | ~10 PFLOP/s FP4, 216 GB HBM3e at 7 TB/s, TSMC 3 nm |
| MTIA v2 (2024) | Meta | Ranking/recommendation inference | 354 TOPS INT8, 256 MB SRAM, 90 W |
| Custom accelerator with Broadcom | OpenAI | Inference | 10 GW deployment plan announced 2025; first racks from late 2026 |
| Ascend 910C, CloudMatrix 384 | Huawei | Training and inference in China | Rack-scale systems built around export-control limits |

## Apple Silicon (Local Inference)

Apple's **unified memory** lets the GPU address most of system RAM, so large models fit on a desktop:

| Chip | Year | Unified memory | Bandwidth | Fits (4-bit) |
|------|------|----------------|-----------|---|
| M2 Ultra | 2023 | up to 192 GB | 800 GB/s | 70B comfortably |
| M4 Max | 2024 | up to 128 GB | 546 GB/s | 70B, or ~120B MoE |
| M3 Ultra | 2025 | up to 512 GB | 819 GB/s | ~400B-class models (e.g., quantized DeepSeek/Qwen MoEs) |
| M5 | 2025 | up to 32 GB | 153 GB/s | 8–14B; adds neural accelerators inside GPU cores |

(There is no M4 Ultra; the 2025 Mac Studio paired M4 Max with M3 Ultra.) Decode speed follows the bandwidth ceiling of [Chapter 22](./22_quantization_benchmarks.md): an MoE with ~20B active parameters at 4 bits reads ~11 GB per token, so ~70 tokens/s is the ceiling at 819 GB/s. **llama.cpp** (Metal) and Apple's **MLX** are the main runtimes.

## Software Decides

Peak FLOPs are easy to put on a slide; usable FLOPs depend on the stack:

| Platform | Stack | Maturity for LLMs (2026) |
|---|---|---|
| NVIDIA | CUDA, cuDNN, NCCL, TensorRT-LLM, Triton, every framework | Reference platform; new methods land here first |
| AMD | ROCm, RCCL, MIOpen, Triton for ROCm | First-class in PyTorch, vLLM, SGLang; kernel lag for new ideas |
| Google TPU | XLA, JAX, Pallas, PyTorch/XLA | Excellent for JAX; vLLM TPU backend for serving |
| AWS | Neuron SDK, NKI | Good for supported architectures; porting effort for new ones |
| Apple | Metal, MLX, llama.cpp | Excellent for local inference; not for training at scale |

## Choosing Hardware

| Use case | Strong options (late 2026) | Why |
|----------|-------------|-----|
| Frontier pre-training | NVIDIA GB200/GB300 NVL72, TPU Ironwood/8t, Trainium3 at hyperscaler scale | Scale-up domains, mature fault tolerance |
| Cost-efficient training and fine-tuning | H100/H200, MI300X/MI355X, Trainium2, TPU v5e/v6e | Price per useful FLOP |
| High-throughput serving | B200/B300, MI355X, TPU Ironwood | HBM capacity and bandwidth, FP8/FP4 |
| Lowest latency per user | SRAM-based systems (Cerebras, Groq/NVIDIA LPX) | Bandwidth from on-chip memory |
| Local development | Apple M-series Max/Ultra, RTX 4090/5090 | Unified memory (Apple) or fast GDDR (NVIDIA) |
| Free experimentation | TPUs and GPUs via Colab / Kaggle | Zero cost for small runs |

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Google split its 2026 TPU into training and inference chips, NVIDIA added an SRAM-based decode rack beside its GPUs, and Microsoft's Maia 200 is inference-only. If prefill, decode and training each want different silicon, how should a serving system (Chapter 31) route a single request across them?</p>
</div>

## Key Takeaways

- TPUs (systolic arrays, ICI pods, JAX/XLA) are the most mature non-GPU platform; Ironwood and TPU 8 put Google at parity with NVIDIA's per-chip numbers.
- AMD leads on HBM capacity (192–288 GB) and ROCm is now first-class in major serving stacks; kernel breadth still trails CUDA.
- SRAM-only designs (Groq, Cerebras) win per-user latency by attacking the bandwidth wall; NVIDIA's Groq licensing deal brought the idea into the mainstream.
- Hyperscalers build their own chips (Trainium, Maia, MTIA, OpenAI–Broadcom) for cost and supply control.
- Apple's unified memory makes 70B–400B-class quantized models runnable on desktops; decode speed is bandwidth ÷ bytes.

## Check Yourself

<details class="check">
<summary>Why does a systolic array need less memory bandwidth per FLOP than a naive matrix unit?</summary>
<p>Each weight or activation fetched from memory is passed from cell to cell and reused by an entire row or column of the array (128 or 256 times), instead of being fetched once per multiply. Data reuse raises arithmetic intensity without extra memory traffic.</p>
</details>

<details class="check">
<summary>How many 230 MB Groq chips are needed just to hold a 70B model's weights in INT8?</summary>
<p>70 GB / 0.23 GB ≈ 305 chips — before KV cache or activations. That is why SRAM-only systems are many-chip by design.</p>
</details>

<details class="check">
<summary>A 405B model in FP8 needs ~405 GB for weights. How many MI355X versus H100 GPUs is that at minimum, ignoring KV cache?</summary>
<p>MI355X: 405/288 → 2 GPUs. H100: 405/80 → 6 GPUs. Fewer GPUs per replica also means less tensor-parallel communication.</p>
</details>

<details class="check">
<summary>What is the decode ceiling for a dense 70B model at 4.25 bits on an M3 Ultra (819 GB/s)?</summary>
<p>70 × 10⁹ × 4.25 / 8 ≈ 37 GB per token; 819 / 37 ≈ 22 tokens/s. Real runtimes typically reach 60–80% of that.</p>
</details>

<details class="check">
<summary>Why can JAX code say nothing about all-reduce and still train in parallel?</summary>
<p>The arrays carry sharding annotations over a named device mesh; when jit-compiled, XLA's SPMD partitioner propagates shardings through the computation and inserts the required collectives (all-reduce of gradients over "data", all-gathers for "model"-sharded weights) automatically.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| In-Datacenter Performance Analysis of a Tensor Processing Unit | Jouppi et al., 2017 | [arxiv.org/abs/1704.04760](https://arxiv.org/abs/1704.04760) | Systolic-array TPU |
| TPU v4: An Optically Reconfigurable Supercomputer | Jouppi et al., 2023 | [arxiv.org/abs/2304.01433](https://arxiv.org/abs/2304.01433) | ICI pods, optical switching |
| Ironwood: The first Google TPU for the age of inference | Google, 2025 | [blog.google](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/) | Ironwood specifications |
| AMD Instinct MI355X | AMD, 2025 | [amd.com](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) | MI355X specifications |
| AWS Trainium | AWS, 2025 | [aws.amazon.com/ai/machine-learning/trainium](https://aws.amazon.com/ai/machine-learning/trainium/) | Trainium2/3 |
| Think Fast: A Tensor Streaming Processor | Abts et al. (Groq), 2020 | [doi.org/10.1109/ISCA45697.2020.00023](https://doi.org/10.1109/ISCA45697.2020.00023) | LPU architecture |
| Cerebras WSE-3 | Cerebras, 2024 | [cerebras.ai/chip](https://www.cerebras.ai/chip) | Wafer-scale specs |
| Maia 200: The AI accelerator built for inference | Microsoft, 2026 | [blogs.microsoft.com](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) | Maia 200 |
| Chips guide: accelerator atlas and Chapters 16, 21, 27 | karam-nus, 2026 | [karam-nus.github.io/chips/atlas.html](https://karam-nus.github.io/chips/atlas.html) | Sourced specs and architecture detail |

---

**Next:** [Chapter 28 — The Hugging Face Ecosystem](./28_huggingface_ecosystem.md) — the software layer that connects all of this to practitioners.

[← Back to Table of Contents](./README.md)
