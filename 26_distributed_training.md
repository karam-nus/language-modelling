---
title: "Chapter 26 — Distributed Training"
description: "Splitting a training run across thousands of GPUs: memory accounting that includes master weights, data parallelism and ZeRO/FSDP2, Megatron tensor and sequence parallelism, pipeline schedules and bubbles, context parallelism for long sequences, expert parallelism for MoE, the collectives underneath, real 4D/5D configurations from Llama 3 and DeepSeek-V3, and low-communication training."
---

[← Back to Table of Contents](./README.md)

# Chapter 26 — Distributed Training

Llama 3.1 405B trained on up to 16,384 H100s; DeepSeek-V3 on 2,048 H800s; 2025–26 frontier runs on clusters of 100,000+ accelerators. Each GPU holds only a slice of the model, the data or the sequence. This chapter is about how those slices are cut.

> **The one-sentence version:** every parallelism scheme trades memory for communication — replicate and you need memory; shard and you need bandwidth — and real runs combine several, each matched to the network tier it communicates over.

## Why Distribute?

Training state, in standard mixed precision with AdamW (16 bytes per parameter, [Chapter 11](./11_optimizers_and_loss_functions.md)), before activations:

| Component | Llama 3.1 70B | Llama 3.1 405B |
|-----------|--------------------|--------------------|
| BF16 weights | 141 GB | 810 GB |
| BF16 gradients | 141 GB | 810 GB |
| FP32 master weights | 282 GB | 1,620 GB |
| FP32 Adam m and v | 565 GB | 3,240 GB |
| **Total state** | **1.13 TB** | **6.48 TB** |
| Minimum 80 GB GPUs just for state | 15 | 81 |

Activations come on top and grow with micro-batch size and sequence length — tens to hundreds of GB per sequence at long context unless recomputed. And even if memory were free, one GPU would need centuries: 405B × 15.6T tokens × 6 = 3.8 × 10²⁵ FLOPs is about 3,000 years on one H100 at 40% MFU.

## The Dimensions of Parallelism

<div class="img-caption">
  <img src="{{ '/assets/images/distributed_training.svg' | relative_url }}" alt="Side-by-side comparison of Data Parallel, Tensor Parallel, and Pipeline Parallel training strategies">
  <figcaption>Data, tensor and pipeline parallelism — production runs add context and expert parallelism on top</figcaption>
</div>

<div class="diagram">
<div class="diagram-title">What Each Dimension Splits</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">Data (DP, FSDP/ZeRO)</div>
    <div class="card-desc">Split the batch. Replicate (DDP) or shard (FSDP) the model state. Gradient all-reduce or reduce-scatter once per step.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Tensor (TP + SP)</div>
    <div class="card-desc">Split each weight matrix (heads, FFN columns). Collectives inside every layer — needs NVLink.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Pipeline (PP)</div>
    <div class="card-desc">Split the layers into stages. Point-to-point activations between stages; micro-batches fill the pipeline.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Context (CP)</div>
    <div class="card-desc">Split one long sequence across GPUs. Ring attention or all-to-all (Ulysses) exchanges keys and values.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Expert (EP)</div>
    <div class="card-desc">Put different MoE experts on different GPUs. All-to-all dispatch and combine of tokens every MoE layer.</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-title">Recompute</div>
    <div class="card-desc">Not parallelism, but the same trade: discard activations in the forward pass and recompute them in backward (selective or full checkpointing).</div>
  </div>
</div>
</div>

## Data Parallelism (DDP)

Each GPU holds a full replica, processes a different slice of the batch, and the gradients are averaged with an **all-reduce** before every optimizer step:

<div class="diagram">
<div class="diagram-title">DDP All-Reduce Gradient Sync</div>
<div class="flow-h">
  <div class="flow-step green">GPU 0: forward + backward → ∇W₀</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step green">GPU 1: forward + backward → ∇W₁</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step green">GPU 2: forward + backward → ∇W₂</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step accent">All-reduce: ∇W = mean(∇W₀, ∇W₁, ∇W₂)</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step orange">Every GPU applies the same update</div>
</div>
</div>

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = DDP(MyTransformer().cuda(), device_ids=[local_rank])
for batch in dataloader:                 # a DistributedSampler gives each rank different data
    loss = model(**batch).loss
    loss.backward()                      # gradients all-reduced in buckets, overlapped with backward
    optimizer.step(); optimizer.zero_grad()
# torchrun --nproc_per_node=8 train.py
```

**Limitation**: every GPU holds the full 16 bytes per parameter. DDP alone stops working above a few billion parameters.

## ZeRO and FSDP: Sharded Data Parallelism

**ZeRO** (DeepSpeed) and PyTorch's **FSDP** keep data parallelism but shard the model state across the data-parallel group, gathering pieces only when needed:

| ZeRO stage | Sharded across N GPUs | Per-GPU state, 70B on 8 GPUs (excl. activations) |
|------------|----------------|------|
| 0 (DDP) | Nothing | 1,130 GB |
| 1 | Optimizer state (master + m + v) | 141 + 141 + 847/8 ≈ 388 GB |
| 2 | + gradients | 141 + 141/8 + 847/8 ≈ 265 GB |
| 3 / FSDP full shard | + parameters | 1,130/8 ≈ 141 GB |

Stage 3 costs extra communication — an **all-gather** of each layer's parameters before its forward and backward computation and a **reduce-scatter** of its gradients afterwards — but overlaps it with compute layer by layer. (It is also why 70B still needs 16+ GPUs: 141 GB per GPU before activations does not fit in 80 GB.)

PyTorch's current API is **FSDP2**, which shards each parameter as a DTensor:

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:              # shard block by block so all-gathers overlap with compute
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)        # root: embeddings, final norm, head
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)   # state is created sharded
```

## Tensor Parallelism (TP) and Sequence Parallelism (SP)

Megatron-style TP splits the matrices inside each layer so that a pair of matmuls needs only one collective:

<div class="diagram">
<div class="diagram-title">Megatron MLP: Column-Parallel, then Row-Parallel</div>
<div class="flow">
  <div class="flow-node accent wide">X [B, T, d] — replicated on all TP ranks</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Column-parallel W_up: rank k computes X · W_up[:, k-th slice] → [B, T, d_ff/TP] <small>no communication; SwiGLU applied locally</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Row-parallel W_down: rank k computes H_k · W_down[k-th slice, :] → partial [B, T, d] <small>each rank holds a partial sum</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">All-reduce (or reduce-scatter with SP) → Y [B, T, d]</div>
</div>
</div>

Attention splits the same way: Q, K, V projections by heads (column-parallel; each rank computes whole heads), the output projection row-parallel. So each transformer block needs **two all-reduces in the forward pass and two in the backward** — every layer, every micro-batch — which is why TP stays inside a node or an NVLink domain (8 GPUs on HGX, 72 on a GB200 NVL72 rack).

**Sequence parallelism** splits the remaining non-matmul work (norms, dropout, residual) along the sequence dimension, turning each all-reduce into a reduce-scatter + all-gather pair and dividing activation memory for those regions by TP.

<div class="diagram">
<div class="diagram-title">Shapes on TP rank k (TP = t)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row"><span class="ts-label">Input (replicated)</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">W_up shard</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim feature">d</span><span class="ts-sep">,</span><span class="ts-dim feature">d_ff / t</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">Hidden shard</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d_ff / t</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">After W_down + all-reduce</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div></div>
</div>
</div>

## Pipeline Parallelism (PP)

Pipeline parallelism assigns consecutive layers to different GPUs and streams micro-batches through them:

<div class="diagram">
<div class="diagram-title">Pipeline Parallelism with Micro-batches (GPipe-style)</div>
<div class="layer-stack">
  <div class="layer green">Stage 0 — layers 0–7:   F1 F2 F3 F4 · · · · · · B4 B3 B2 B1</div>
  <div class="layer accent">Stage 1 — layers 8–15:  · F1 F2 F3 F4 · · · · B4 B3 B2 B1 ·</div>
  <div class="layer purple">Stage 2 — layers 16–23: · · F1 F2 F3 F4 · · B4 B3 B2 B1 · ·</div>
  <div class="layer orange">Stage 3 — layers 24–31: · · · F1 F2 F3 F4 B4 B3 B2 B1 · · ·</div>
</div>
</div>

The dots are the **pipeline bubble**: stages idle while the pipeline fills and drains. With p stages and m micro-batches the idle fraction is about **(p − 1)/(m + p − 1)** — 3/35 ≈ 9% for p = 4, m = 32. Schedules attack it:

- **1F1B**: alternate one forward and one backward per stage after warm-up, so at most p micro-batches' activations are alive (memory), with the same bubble.
- **Interleaved 1F1B**: each GPU holds several non-adjacent chunks of layers, shrinking the bubble by the number of chunks at the cost of more communication.
- **Zero-bubble schedules** (2023–24) split the backward into input-gradient and weight-gradient parts and fill bubbles with the latter.
- **DualPipe** (DeepSeek-V3) feeds micro-batches from both ends of the pipeline and overlaps MoE all-to-all communication with computation.

## Context Parallelism (CP)

A single 128K-token sequence produces activations too big for one GPU. **Context parallelism** splits the sequence across GPUs: each holds a chunk of queries and, for attention, needs keys and values from all chunks. **Ring attention** passes K/V blocks around a ring while computing (overlapping communication with the online-softmax accumulation of [Chapter 25](./25_cuda_and_kernel_development.md)); **Ulysses** uses all-to-all to switch from sequence-sharding to head-sharding for the attention step. Causal masking makes naive splits unbalanced (later chunks do more work), so chunks are assigned in a zig-zag order. Llama 3's long-context stage used CP = 16.

## Expert Parallelism (EP)

In an MoE layer each token visits only k of E experts, so experts can live on different GPUs. Every MoE layer then does two **all-to-all** exchanges: dispatch (send each token's hidden state to the GPUs holding its experts) and combine (bring the results back). All-to-all traffic crosses the slower inter-node network, so MoE training lives or dies by it. DeepSeek-V3 used EP = 64 across 8 nodes, limited each token to experts on at most 4 nodes, wrote custom all-to-all kernels (later open-sourced as DeepEP) and overlapped communication with computation so that, by its report, nearly all of it was hidden. See [Chapter 33](./33_mixture_of_experts.md).

## Communication Patterns

| Collective | Used for | Bytes each GPU sends/receives (ring) | Network tier |
|-----------|-----------|------------|----------|
| **All-reduce** | DDP gradients; TP partial sums | ≈ 2 × message size | Any (TP: NVLink) |
| **All-gather** | FSDP parameters; SP | ≈ full tensor received | NVLink or fast IB |
| **Reduce-scatter** | FSDP gradients; SP | ≈ full tensor sent | NVLink or fast IB |
| **Send/recv** | PP activations and gradients | One activation per micro-batch | Inter-node fine |
| **All-to-all** | EP token dispatch/combine; Ulysses CP | Tokens × d per layer | The hardest; needs high bisection bandwidth |

The art is matching each dimension to a network tier: TP (and often EP) inside the NVLink domain, CP and PP across nodes with fast InfiniBand/RoCE, data parallelism (whose all-reduce happens once per step and overlaps with backward) across the slowest links.

## Putting It Together: Real Configurations

| Run | Accelerators | Parallelism |
|---|---|---|
| Llama 3.1 405B, 8K pre-training | 16,384 H100 | TP 8 × PP 16 × DP 128 (FSDP) |
| Llama 3.1 405B, 128K stage | 16,384 H100 | TP 8 × CP 16 × PP 16 × DP 8 |
| DeepSeek-V3 (671B MoE) | 2,048 H800 | PP 16 (DualPipe) × EP 64 across 8 nodes × ZeRO-1 DP; no TP |

Note DeepSeek's choice: on H800s with reduced NVLink bandwidth, it avoided tensor parallelism entirely and relied on expert and pipeline parallelism plus heavy overlap — a design shaped by the hardware it had. Estimate per-GPU memory for your own combination of DP, TP, PP and ZeRO stage:

<div class="lab" data-lab="parallel"><noscript>Interactive lab: per-GPU memory for weights, gradients, optimizer state and activations under data, tensor and pipeline parallelism with ZeRO stages. Enable JavaScript to use it.</noscript></div>

## Frameworks

| Framework | Strengths |
|---|---|
| **Megatron-Core / NeMo** (NVIDIA) | TP/SP/PP/CP/EP, the reference for large-scale GPU training |
| **PyTorch FSDP2 + TorchTitan** | Native PyTorch 4D parallelism, Float8, compile |
| **DeepSpeed** | ZeRO stages and offload; strong for fine-tuning on few GPUs |
| **JAX / MaxText, Pathways** | TPU-native sharding with named mesh axes |
| **Hugging Face Accelerate / Nanotron** | Simple multi-GPU wrappers; educational large-scale code |

```python
# DeepSpeed ZeRO-3 via Hugging Face Accelerate
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

ds_config = {
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 3, "overlap_comm": True, "contiguous_gradients": True},
}
accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config))
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

<div class="callout trend">
<span class="callout-title">2025–26 trend — training across data centres</span>
<p>Power limits are pushing frontier runs across multiple sites. Low-communication methods such as DiLoCo — each group trains locally for hundreds of steps and only occasionally synchronises an averaged update — cut cross-site bandwidth by orders of magnitude, and decentralised runs (Prime Intellect's INTELLECT series) used them to train over the public internet. Asynchronous RL (Chapter 15) is the same idea applied to post-training.</p>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A GB200 NVL72 rack puts 72 GPUs in one NVLink domain. If tensor and expert parallelism can now span 72 GPUs at NVLink speed instead of 8, which of the tricks in this chapter — pipeline parallelism, DualPipe, node-limited routing — become unnecessary, and which new bottleneck appears?</p>
</div>

## Key Takeaways

- Mixed-precision AdamW needs 16 bytes per parameter: 1.13 TB for 70B before activations.
- DDP replicates; ZeRO/FSDP shards optimizer state, gradients and parameters across the data-parallel group (FSDP2 is PyTorch's current API).
- TP splits matrices with two all-reduces per block per pass (keep it in NVLink); PP splits layers and pays a bubble ≈ (p−1)/(m+p−1); CP splits sequences; EP splits experts with all-to-all.
- Map each dimension to a network tier; real runs combine four or five (Llama 3: TP × CP × PP × DP; DeepSeek-V3: PP × EP × DP).
- Low-communication and asynchronous methods are pushing training across sites.

## Check Yourself

<details class="check">
<summary>How much per-GPU state does a 13B model need under ZeRO-3 on 16 GPUs?</summary>
<p>16 bytes × 13 × 10⁹ = 208 GB total, ÷ 16 = 13 GB per GPU, plus activations and temporarily gathered parameters for the layer being computed.</p>
</details>

<details class="check">
<summary>Why does Megatron pair a column-parallel matmul with a row-parallel one?</summary>
<p>The column-parallel layer leaves each rank with a slice of the hidden dimension, which is exactly the input the row-parallel layer needs; the element-wise activation in between needs no communication. Only the row-parallel output needs one all-reduce, instead of a collective after each matmul.</p>
</details>

<details class="check">
<summary>With 8 pipeline stages, how many micro-batches keep the bubble below 5%?</summary>
<p>(p − 1)/(m + p − 1) < 0.05 → 7/(m + 7) < 0.05 → m > 133. More micro-batches means a larger global batch (or smaller micro-batches), which is the tension interleaved and zero-bubble schedules ease.</p>
</details>

<details class="check">
<summary>Why did DeepSeek-V3 limit each token's experts to at most 4 nodes?</summary>
<p>All-to-all traffic across the inter-node network is the bottleneck of expert parallelism. Capping the number of distinct nodes per token bounds cross-node traffic per token, so it can be overlapped with computation.</p>
</details>

<details class="check">
<summary>Why is data parallelism usually placed on the slowest network tier?</summary>
<p>Its gradient synchronisation happens once per optimizer step and can overlap with the backward pass, so it tolerates lower bandwidth and higher latency. TP and EP communicate inside every layer and would stall compute on slow links.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Megatron-LM: Training Multi-Billion Parameter LMs Using Model Parallelism | Shoeybi et al., 2019 | [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053) | Tensor parallelism |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al., 2019 | [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) | ZeRO stages |
| GPipe | Huang et al., 2018 | [arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965) | Pipeline parallelism, bubbles |
| Efficient Large-Scale LM Training on GPU Clusters (PTD-P) | Narayanan et al., 2021 | [arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473) | Interleaved 1F1B, 3D parallelism |
| Reducing Activation Recomputation in Large Transformer Models | Korthikanti et al., 2022 | [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198) | Sequence parallelism, selective recompute |
| Zero Bubble Pipeline Parallelism | Qi et al., 2023 | [arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241) | Split backward to fill bubbles |
| Ring Attention with Blockwise Transformers | Liu et al., 2023 | [arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889) | Context parallelism |
| PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | Zhao et al., 2023 | [arxiv.org/abs/2304.11277](https://arxiv.org/abs/2304.11277) | FSDP design |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | 4D parallel configuration |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | DualPipe, EP, node-limited routing |
| DiLoCo: Distributed Low-Communication Training | Douillard et al., 2023 | [arxiv.org/abs/2311.08105](https://arxiv.org/abs/2311.08105) | Cross-site training |
| TorchTitan | Liang et al., 2024 | [arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511) | PyTorch-native 4D parallelism |

---

**Next:** [Chapter 27 — ASICs & Specialized Accelerators](./27_asics_and_accelerators.md) — TPUs, Trainium, wafer-scale engines and the rest of the non-GPU world.

[← Back to Table of Contents](./README.md)
