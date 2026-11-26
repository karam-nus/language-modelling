---
title: "Chapter 26 — Distributed Training"
description: "Splitting a training run across thousands of GPUs, step by step: the collectives underneath (ring all-reduce, all-gather, reduce-scatter, all-to-all) and their costs; data parallelism and bucketed overlap; ZeRO and FSDP2 sharding; Megatron tensor and sequence parallelism; pipeline schedules from GPipe to zero-bubble and DualPipe; context parallelism with ring attention and Ulysses; expert parallelism for MoE; device meshes and 4D/5D configurations from Llama 3 and DeepSeek-V3; activation memory, MFU, failures, and training across data centres."
related:
  - { id: chips, why: "Interconnects, NVLink domains and multi-GPU systems in hardware." }
  - { id: training, why: "Training runs in practice." }
  - { id: networking, why: "The networking under collective communication." }
---

[← Back to Table of Contents](./README.md)

# Chapter 26 — Distributed Training

Llama 3.1 405B trained on up to 16,384 H100s; DeepSeek-V3 on 2,048 H800s; 2025–26 frontier runs span 100,000 or more accelerators, sometimes across several data centres. Each GPU holds only a slice of the model, the data or the sequence, and spends part of every step exchanging slices with others. This chapter shows exactly how each slicing works — which GPU holds what, which messages move, and what they cost.

> **The one-sentence version:** every parallelism scheme trades memory for communication — replicate and you need memory, shard and you need bandwidth — and a real run combines several, placing the chattiest ones on the fastest links.

## Why Distribute?

Training state in standard mixed precision with AdamW (16 bytes per parameter, [Chapter 11](./11_optimizers_and_loss_functions.md)), before activations:

| Component | Llama 3.1 70B | Llama 3.1 405B |
|-----------|--------------------|--------------------|
| BF16 weights | 141 GB | 810 GB |
| BF16 gradients | 141 GB | 810 GB |
| FP32 master weights | 282 GB | 1,620 GB |
| FP32 Adam m and v | 565 GB | 3,240 GB |
| **Total state** | **1.13 TB** | **6.48 TB** |
| Minimum 80 GB GPUs just for state | 15 | 81 |

Activations come on top: for the 70B shape at 8K tokens, about 2.3 GB per layer per sequence even with FlashAttention — 183 GB for all 80 layers (computed below). And even if memory were free, one GPU would need centuries: 405B × 15.6T tokens × 6 = 3.8 × 10²⁵ FLOPs is about 3,000 years on one H100 at 40% MFU.

There are five ways to split the work, plus one trick that trades compute for memory:

| Dimension | What is split | Communication | Typical network tier |
|---|---|---|---|
| **Data (DP, FSDP/ZeRO)** | The batch (and optionally the model state) | Gradient all-reduce or reduce-scatter; parameter all-gather with FSDP | Anywhere; tolerant of slow links |
| **Tensor (TP + SP)** | Each weight matrix | All-reduce (or reduce-scatter + all-gather) inside every layer | NVLink only |
| **Pipeline (PP)** | The layers | Point-to-point activations between stages | Across nodes |
| **Context (CP)** | The sequence | K/V ring exchange or all-to-all | NVLink or fast InfiniBand |
| **Expert (EP)** | MoE experts | All-to-all token dispatch and combine | The hardest: high bisection bandwidth |
| *Recompute* | — | None: recompute activations in backward | — |

## The Toolbox: Collective Operations

Every scheme in this chapter is built from a handful of **collectives** — operations in which a group of GPUs exchanges data in a fixed pattern. NCCL (NVIDIA), RCCL (AMD) and the TPU ICI runtime implement them.

<div class="diagram">
<div class="diagram-title">The Collectives, Before and After</div>
<div class="fig" data-fig="collectives"><noscript>Figure: four ranks with four chunks each, before and after all-reduce (everyone ends with the sum), reduce-scatter (rank r ends with the sum of chunk r), all-gather (everyone ends with every rank's shard), all-to-all (rank r ends with chunk r from every rank), broadcast and point-to-point send.</noscript></div>
<div class="flow-caption">Each square is a chunk of a tensor; colour shows which rank it came from, stripes show a sum of several ranks' chunks.</div>
</div>

The workhorse is the **ring all-reduce**. Naively, every GPU sending its whole gradient to every other GPU would cost N − 1 copies each. The ring algorithm splits the tensor into N chunks and passes them around a ring: N − 1 steps of **reduce-scatter** (send one chunk right, add what arrives), then N − 1 steps of **all-gather** (pass the finished chunks around):

<div class="diagram">
<div class="diagram-title">Ring All-Reduce, Step by Step</div>
<div class="fig" data-fig="ringallreduce" data-n="4"><noscript>Figure: four GPUs each holding four gradient chunks. In three reduce-scatter steps each GPU sends one chunk to its right neighbour, which adds it; afterwards each GPU owns the full sum of one chunk. In three all-gather steps the finished chunks circulate until every GPU has the full sum.</noscript></div>
<div class="flow-caption">Every link is busy at every step, and each GPU sends only 2(N − 1)/N of the data in total — almost independent of N.</div>
</div>

| Collective | Bytes each GPU sends (ring, message size S) | Used for |
|---|---|---|
| All-reduce | 2 (N − 1)/N · S | DDP gradients; TP partial sums |
| Reduce-scatter | (N − 1)/N · S | FSDP gradients; sequence parallelism |
| All-gather | (N − 1)/N · S | FSDP parameters; sequence parallelism |
| All-to-all | (N − 1)/N · S | MoE dispatch and combine; Ulysses context parallelism |
| Send / recv | S (to one neighbour) | Pipeline activations and gradients |

Time ≈ bytes ÷ per-GPU bandwidth + steps × latency. Bandwidth depends brutally on where the GPUs are:

| Link | Per-GPU bandwidth (each direction) | Scope |
|---|---|---|
| NVLink 4 (H100) | ~450 GB/s | 8 GPUs in an HGX node |
| NVLink 5 (B200, GB200) | ~900 GB/s | 8 GPUs, or 72 in a GB200 NVL72 rack |
| InfiniBand / RoCE, 400 Gb/s per GPU | ~50 GB/s | Across nodes |
| 800 Gb/s NICs (2025–26) | ~100 GB/s | Across nodes |

Collectives inside an NVLink domain are roughly 10× faster than across nodes. NCCL also picks algorithms beyond the ring — trees for latency-bound small messages, and **NVLS**, which uses NVSwitch hardware to do the reduction inside the switch. The placement rule that follows from this table drives every design below: **communicate often only where bandwidth is high.**

In PyTorch the collectives are one call each (inside a process group launched by `torchrun`):

```python
import torch, torch.distributed as dist

dist.init_process_group("nccl")               # one process per GPU
world = dist.get_world_size()
x = torch.ones(4096, device="cuda")

dist.all_reduce(x, op=dist.ReduceOp.SUM)                        # every rank: sum over ranks
shard = torch.empty(4096 // world, device="cuda")
dist.reduce_scatter_tensor(shard, x)                            # rank r: sum of chunk r
full = torch.empty(4096, device="cuda")
dist.all_gather_into_tensor(full, shard)                        # every rank: all shards
out = torch.empty_like(x)
dist.all_to_all_single(out, x)                                  # chunk j of rank i → rank j
```

## Data Parallelism (DDP)

Every GPU holds a full replica of the model and processes a different slice of the global batch. After the backward pass the gradients are averaged with an all-reduce, so every replica applies the same update and stays identical.

The trick that makes DDP efficient is **overlap**. Gradients become available layer by layer, last layer first, during the backward pass. DDP groups them into buckets (25 MB by default) and launches each bucket's all-reduce as soon as it is complete, while backward continues on earlier layers:

<div class="diagram">
<div class="diagram-title">One Data-Parallel Step, Without and With Overlap</div>
<div class="fig" data-fig="ddp"><noscript>Figure: timelines for two GPUs and the network. Without overlap the all-reduce of all gradients starts after the backward pass and the optimizer waits for it. With bucketed overlap, each layer's gradient bucket is all-reduced while earlier layers are still in backward, and the step finishes sooner.</noscript></div>
</div>

```python
import os, contextlib, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = DDP(MyTransformer().cuda(), device_ids=[local_rank], bucket_cap_mb=25)
for step, batch in enumerate(dataloader):     # a DistributedSampler gives each rank different data
    sync = (step + 1) % accum_steps == 0
    with (contextlib.nullcontext() if sync else model.no_sync()):  # accumulate locally, all-reduce on the last
        loss = model(**batch).loss / accum_steps
        loss.backward()                       # buckets all-reduced during backward when sync
    if sync:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
# torchrun --nproc_per_node=8 train.py
```

**Global batch** = micro-batch per GPU × accumulation steps × DP size. **Cost**: one all-reduce of all gradients per optimizer step — 2 bytes × P in BF16.

How small is it? For Llama 3.1 8B the BF16 gradients are 16.1 GB. A ring all-reduce over 8 GPUs on NVLink (450 GB/s) takes about 62 ms; over 64 GPUs on 50 GB/s InfiniBand about 630 ms. One step of 8 × 8K-token sequences per GPU takes about 8 s of compute at 40% MFU. Even across slow links the communication is under 10% of the step and mostly overlapped — which is why data parallelism sits on the slowest network tier.

**Limitation**: every GPU holds the full 16 bytes per parameter. DDP alone stops at a few billion parameters.

## ZeRO and FSDP: Sharded Data Parallelism

**ZeRO** (DeepSpeed, 2019) and PyTorch's **FSDP** keep the data-parallel structure but stop replicating the model state: each of the N ranks stores only 1/N of it and fetches the rest when needed.

<div class="diagram">
<div class="diagram-title">What Each Rank Stores Under ZeRO Stages 0–3</div>
<div class="fig" data-fig="zero"><noscript>Figure: four ranks' memory for weights, gradients and optimizer state. ZeRO-0 replicates everything (16 bytes per parameter per rank); ZeRO-1 shards the optimizer state (7); ZeRO-2 also shards gradients (5.5); ZeRO-3 / FSDP also shards weights (4).</noscript></div>
</div>

| Stage | Sharded across N ranks | Per-rank state (N = 4) | Per-rank state, 70B on 8 GPUs | Communication per step |
|------------|----------------|------|------|------|
| 0 (DDP) | Nothing | 16 B/param | 1,130 GB | All-reduce gradients: 2Ψ |
| 1 | Optimizer state (master + m + v) | 7 B/param | 141 + 141 + 847/8 ≈ 388 GB | Reduce-scatter grads + all-gather updated params: 2Ψ |
| 2 | + gradients | 5.5 B/param | 141 + 141/8 + 847/8 ≈ 265 GB | Same: 2Ψ |
| 3 / FSDP | + parameters | 4 B/param | 1,130/8 ≈ 141 GB | All-gather (fwd) + all-gather (bwd) + reduce-scatter: 3Ψ |

(Ψ = the size of the parameters in BF16; an all-reduce costs about 2Ψ per rank.) Stages 1 and 2 are free in bandwidth — an all-reduce *is* a reduce-scatter plus an all-gather, so splitting it around the optimizer step moves the same bytes. Stage 3 costs 1.5× because parameters must be gathered twice, once for the forward pass and once for the backward. Here is what one rank does through a step:

<div class="diagram">
<div class="diagram-title">An FSDP Step on One Rank</div>
<div class="fig" data-fig="fsdp"><noscript>Figure: a rank holding one shard of each of four layers. In the forward pass it all-gathers layer 0, computes, frees the other shards and moves to layer 1. In the backward pass it re-gathers each layer, computes full gradients, reduce-scatters them so it keeps only its own gradient shard, and finally runs the optimizer on its shard.</noscript></div>
<div class="flow-caption">Prefetching the next layer's all-gather while computing the current one hides most of the extra communication.</div>
</div>

Even ZeRO-3 is not enough for 70B on 8 GPUs: 141 GB of state per GPU before activations does not fit in 80 GB — hence 16+ GPUs, or the other dimensions below. **HSDP** (hybrid sharding) shards within a node and replicates across nodes, keeping the frequent all-gathers on NVLink. **ZeRO-Offload / ZeRO-Infinity** push optimizer state to CPU RAM or NVMe, useful for fine-tuning big models on few GPUs.

PyTorch's current API is **FSDP2**, which shards each parameter as a DTensor:

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:              # wrap block by block so each all-gather covers one block
    fully_shard(block, mp_policy=mp)    # reshard_after_forward=True (default) frees gathered weights
fully_shard(model, mp_policy=mp)        # root: embeddings, final norm, head
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)   # optimizer state is created sharded
```

## Tensor Parallelism (TP)

Megatron-style tensor parallelism splits the matrices *inside* each layer across t GPUs. The trick is to pair two matmuls so that only one collective is needed:

<div class="diagram">
<div class="diagram-title">Megatron MLP on Two Ranks</div>
<div class="fig" data-fig="tp" data-variant="mlp"><noscript>Figure: the input X is replicated on both ranks; W_up is split by columns so each rank computes half of the hidden units; SwiGLU runs locally; W_down is split by rows so each rank produces a partial sum of Y; one all-reduce sums the partials.</noscript></div>
</div>

Attention splits the same way, by heads:

<div class="diagram">
<div class="diagram-title">Attention Heads Divided Across Ranks</div>
<div class="fig" data-fig="tp" data-variant="attn"><noscript>Figure: eight attention heads split four per rank; each rank computes Q, K and V for its heads and runs attention locally; the output projection is row-parallel and needs one all-reduce.</noscript></div>
</div>

<div class="diagram">
<div class="diagram-title">Shapes on TP Rank k (TP = t)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row"><span class="ts-label">Input (replicated)</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">W_up shard</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim feature">d</span><span class="ts-sep">,</span><span class="ts-dim feature">d_ff / t</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">Hidden shard</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d_ff / t</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">Q, K, V for H/t heads</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim heads">H / t</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d_head</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">After W_down + all-reduce</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div></div>
</div>
</div>

So each transformer block needs **two all-reduces in the forward pass and two in the backward** — every layer, every micro-batch. Put numbers on it for Llama 3.1 70B (d = 8,192, 80 layers) and one 8K-token sequence: each all-reduce moves a [1, 8192, 8192] BF16 tensor = 134 MB; 4 per layer × 80 layers = 43 GB of payload per micro-batch, or 75 GB of traffic per GPU at TP = 8.

| TP = 8 for Llama 3.1 70B, one 8K sequence | Time |
|---|---|
| Compute per GPU (40% MFU on H100) | ~1.1 s |
| TP traffic over NVLink (450 GB/s) | ~0.17 s — can hide behind compute |
| TP traffic over InfiniBand (50 GB/s) | ~1.5 s — longer than the compute |

That is the whole argument for keeping TP inside an NVLink domain: 8 GPUs on HGX, up to 72 on a GB200 NVL72 rack. With GQA there is one more constraint: TP cannot exceed the number of KV heads without replicating them.

PyTorch expresses the Megatron plan with DTensor:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

tp_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))
for block in model.layers:
    parallelize_module(block, tp_mesh, {
        "attn.q_proj": ColwiseParallel(), "attn.k_proj": ColwiseParallel(), "attn.v_proj": ColwiseParallel(),
        "attn.o_proj": RowwiseParallel(),                       # partial sums → all-reduce
        "mlp.gate_proj": ColwiseParallel(), "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    })
```

## Sequence Parallelism (SP)

Between the TP regions sit LayerNorm, dropout and the residual add. Under plain TP every rank computes them redundantly on the full [B, T, d] tensor and stores the full activations. **Sequence parallelism** (Korthikanti et al., 2022) splits those regions along the sequence instead:

<div class="diagram">
<div class="diagram-title">Sequence-Parallel and Tensor-Parallel Regions in One Block</div>
<div class="fig" data-fig="sp"><noscript>Figure: within a transformer block on t ranks, LayerNorm runs on T/t tokens per rank; an all-gather assembles the full sequence for attention, which is tensor-parallel; a reduce-scatter returns to sequence shards for dropout and the residual; the same pattern repeats for the MLP.</noscript></div>
</div>

Each all-reduce becomes a reduce-scatter (into sequence shards) plus an all-gather (back to the full sequence) — the same bytes, rearranged — and the activations of the non-matmul regions shrink by t. Combined with **selective recomputation** (recompute the cheap, memory-hungry attention internals instead of storing them), the same paper cut activation memory by about 5× for large models.

## Pipeline Parallelism (PP)

Pipeline parallelism assigns consecutive layers to different GPUs ("stages") and sends activations forward and gradients backward between neighbours — one point-to-point message per micro-batch per boundary (134 MB for the 70B shape at 8K tokens). It needs little bandwidth, so it is the natural dimension to stretch across nodes. Its problem is idle time.

**GPipe** splits the batch into m micro-batches and runs all forwards, then all backwards. While the pipeline fills and drains, stages idle — the **bubble**:

<div class="diagram">
<div class="diagram-title">GPipe: All Forwards, Then All Backwards (p = 4, m = 8)</div>
<div class="fig" data-fig="pipeline" data-variant="gpipe" data-p="4" data-m="8"><noscript>Figure: a Gantt chart of four stages processing eight micro-batches: forwards cascade diagonally, then backwards cascade back; grey triangles at the start and end are idle time, 27% of the total.</noscript></div>
</div>

With p stages and m micro-batches the bubble is **(p − 1)/(m + p − 1)** of the time — 27% here, 9% for p = 4, m = 32. Worse, stage 0 must keep the activations of all m micro-batches alive until their backward passes.

**1F1B** keeps the same bubble but starts backward passes as early as possible and then alternates one forward with one backward, so each stage holds at most p micro-batches of activations:

<div class="diagram">
<div class="diagram-title">1F1B: Same Bubble, Bounded Memory (p = 4, m = 8)</div>
<div class="fig" data-fig="pipeline" data-variant="1f1b" data-p="4" data-m="8"><noscript>Figure: the 1F1B schedule — after a warm-up of forwards, each stage alternates one forward and one backward; stage 0 never holds more than four micro-batches of activations.</noscript></div>
</div>

Two refinements cut the bubble itself:

- **Interleaved 1F1B** (Narayanan et al., 2021): each GPU holds v non-adjacent chunks of layers (e.g. GPU 0 has layers 0–3 and 16–19), so the pipeline fills v times faster. Bubble ≈ (p − 1)/(v·m + p − 1), at the cost of v× more point-to-point messages.
- **Zero-bubble** schedules (Qi et al., 2023) split each backward into the input-gradient part B (needed by the previous stage right away) and the weight-gradient part W (needed only by the optimizer), and slide the W work into the gaps. With equal F, B and W costs the bubble drops to about a third of 1F1B's.

**DualPipe** (DeepSeek-V3) feeds micro-batches from both ends of the pipeline at once and overlaps each chunk's computation with the expert-parallel all-to-all of another, so that, per DeepSeek's report, communication is almost entirely hidden — at the price of keeping two copies of some parameters.

Run every schedule yourself — the simulator below lays out each stage's operations from the schedule's rules and the dependencies between stages, and reproduces the formulas above exactly:

<div class="lab" data-lab="pipeline"><noscript>Interactive lab: pipeline schedule simulator for GPipe, 1F1B, interleaved 1F1B and zero-bubble schedules, with stages, micro-batches and chunks; shows the Gantt chart, bubble fraction and activation memory. Enable JavaScript to use it.</noscript></div>

Practicalities: the layer count must divide sensibly across stages, and the embedding and output head make the first and last stages heavier (frameworks put fewer transformer layers there). More micro-batches shrink the bubble but each must be large enough to keep the GPU busy, so PP pushes towards large global batches.

## Context Parallelism (CP)

A single 128K-token sequence produces activations too large for one GPU even with everything else sharded. **Context parallelism** splits the sequence: each GPU holds a chunk of the tokens for every layer. All the per-token work (norms, MLPs, projections) is naturally local; attention is not, because each query needs keys and values from every earlier chunk.

**Ring attention** keeps each GPU's queries in place and passes key/value chunks around a ring. At every step a GPU computes attention between its queries and the KV chunk it currently holds — merging partial results with the online softmax of [Chapter 25](./25_cuda_and_kernel_development.md) — while sending that chunk on and receiving the next:

<div class="diagram">
<div class="diagram-title">Ring Attention Across Four GPUs</div>
<div class="fig" data-fig="ring"><noscript>Figure: four GPUs in a ring, each holding a query chunk; key/value chunks rotate one position per step; a 4×4 grid of query chunk × key chunk fills in step by step, with blocks above the diagonal skipped by the causal mask.</noscript></div>
</div>

Causal masking makes a naive split unbalanced — the GPU with the first chunk has almost nothing to do — so implementations give each GPU two chunks from opposite ends of the sequence (zig-zag), equalising the work.

**DeepSpeed Ulysses** takes the other route: an all-to-all switches the layout from "my tokens, all heads" to "all tokens, my heads" just for the attention step, then switches back:

<div class="diagram">
<div class="diagram-title">Ulysses: Sequence Shards to Head Shards and Back</div>
<div class="fig" data-fig="ulysses"><noscript>Figure: four GPUs each holding a quarter of the tokens for all eight heads; an all-to-all regroups the data so each GPU holds all tokens for two heads; attention runs locally per head; a second all-to-all restores sequence shards.</noscript></div>
</div>

| | Ring attention | Ulysses |
|---|---|---|
| Communication | K/V chunks point-to-point around a ring, overlapped with compute | Two all-to-alls of Q, K, V and outputs per layer |
| Limit on degree | None in principle | ≤ number of KV heads |
| Best on | Long rings, including across nodes | NVLink domains with enough heads |

Hybrids (e.g. "unified sequence parallelism") use Ulysses inside a node and the ring across nodes. Llama 3's 128K long-context stage used CP = 16.

## Expert Parallelism (EP)

In a mixture-of-experts layer ([Chapter 33](./33_mixture_of_experts.md)) each token uses only k of E experts, so the experts can live on different GPUs. Every MoE layer then runs four steps:

<div class="diagram">
<div class="diagram-title">Route, Dispatch, Compute, Combine</div>
<div class="fig" data-fig="ep"><noscript>Figure: four GPUs each holding four tokens and one expert. The router assigns each token an expert; a dispatch all-to-all moves every token to the GPU that holds its expert; each expert processes the tokens it received; a combine all-to-all returns the outputs to the tokens' home GPUs.</noscript></div>
</div>

The traffic is large and irregular. For DeepSeek-V3's shape (d = 7,168, k = 8 experts per token), 4,096 tokens on one GPU send 235 MB per MoE layer in the dispatch (FP8) and receive 470 MB in the combine (BF16) — about 41 GB per forward pass over 58 MoE layers. And because routing is data-dependent, some GPUs receive far more tokens than others (load balancing, Chapter 33).

DeepSeek-V3 made this work on H800s with three choices: EP = 64 across 8 nodes; **node-limited routing** (each token's experts on at most 4 nodes, bounding cross-node copies); and custom all-to-all kernels — open-sourced as **DeepEP** — overlapped with computation by DualPipe. A common layout in 2025–26 MoE training and serving is "attention data-parallel, experts expert-parallel": each GPU runs attention for its own sequences, and only the MoE layers exchange tokens.

## Putting It Together: Device Meshes and 4D/5D Parallelism

Real runs multiply the dimensions. The GPUs are arranged as a **device mesh** — a multi-dimensional grid of ranks — and each dimension's groups are chosen so that the chattiest ones sit on the fastest links:

<div class="diagram">
<div class="diagram-title">A 32-GPU Mesh: TP × CP × PP × DP = 4 × 2 × 2 × 2</div>
<div class="fig" data-fig="mesh"><noscript>Figure: four 8-GPU nodes. Tensor-parallel groups are four GPUs within a node; context-parallel pairs also stay within the node; pipeline-parallel pairs span two nodes; data-parallel pairs span the other node pair. Buttons highlight each dimension's groups.</noscript></div>
<div class="flow-caption">Click a dimension to colour its groups. TP varies fastest in the rank numbering, so its groups are consecutive GPUs on one NVLink switch.</div>
</div>

The mapping rule, from most to least communication: **TP (and EP) innermost on NVLink, then CP, then PP across nodes, DP outermost.** In PyTorch the mesh is one call, and each parallelism takes its own sub-mesh:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (2, 2, 2, 4), mesh_dim_names=("dp", "pp", "cp", "tp"))   # 32 GPUs
tp_mesh, dp_mesh = mesh["tp"], mesh["dp"]    # innermost dim = adjacent ranks = same NVLink node
# parallelize_module(model, tp_mesh, plan); fully_shard(block, mesh=dp_mesh); …
```

| Run | Accelerators | Parallelism |
|---|---|---|
| Llama 3.1 405B, 8K pre-training | 16,384 H100 | TP 8 × PP 16 × DP 128 (FSDP) = 16,384 |
| Llama 3.1 405B, 128K stage | 16,384 H100 | TP 8 × CP 16 × PP 16 × DP 8 = 16,384 |
| DeepSeek-V3 (671B MoE) | 2,048 H800 | PP 16 (DualPipe) × EP 64 across 8 nodes × ZeRO-1 DP; no TP |

DeepSeek's choice is instructive: on H800s with reduced NVLink bandwidth it avoided tensor parallelism entirely and relied on expert and pipeline parallelism plus aggressive overlap — a design shaped by the hardware it had. Estimate per-GPU model state for your own combination:

<div class="lab" data-lab="parallel"><noscript>Interactive lab: per-GPU memory for weights, gradients and optimizer state under data, tensor and pipeline parallelism with ZeRO stages. Enable JavaScript to use it.</noscript></div>

## Activation Memory and Recomputation

Model state is only half the memory. Korthikanti et al. (2022) give the activations stored per layer, per sequence, in 16-bit precision without recomputation as about s·b·h·(34 + 5·a·s/h) bytes (s tokens, b batch, h hidden size, a heads). The second term is the T × T attention matrix, which FlashAttention never stores, leaving about 34·s·b·h:

| Llama 3.1 70B shape, one 8K sequence | Activation memory |
|---|---|
| Per layer, storing attention scores | ~23.8 GB |
| Per layer, FlashAttention | ~2.3 GB |
| All 80 layers, FlashAttention | ~183 GB |
| ÷ TP 8 with sequence parallelism | ~23 GB per GPU |

The remaining lever is **recomputation** (activation checkpointing): store only each layer's input and recompute the layer during backward, at about a third more compute; or recompute selectively — only the cheap operations with big outputs. Frameworks also **offload** activations to CPU memory and stream them back.

## Making It Fast: MFU, Overlap and Failures

Model FLOPs utilisation (MFU) — useful FLOPs ÷ peak — is the headline efficiency metric. Llama 3.1 405B reported 38–43% in BF16 on H100s. The gap to 100% comes from communication that is not hidden, pipeline bubbles, recomputation, memory-bound kernels (norms, softmax) and stragglers. The tools that close it:

- **Overlap everything**: prefetch FSDP all-gathers, bucket gradient reductions, decompose TP matmuls so communication runs alongside compute ("async TP"), and let DualPipe-style schedules hide all-to-alls.
- **Lower precision**: FP8 training ([Chapter 10](./10_pretraining_at_scale.md)) doubles tensor-core throughput and halves some communication (DeepSeek dispatched tokens in FP8).
- **Compilation**: `torch.compile` fuses the memory-bound operations.

At 16,000 GPUs, failures are routine. During a 54-day period of Llama 3 pre-training Meta recorded 466 job interruptions, 419 of them unexpected and most caused by hardware (GPUs and HBM above all); frequent asynchronous checkpointing, fast restarts and automated diagnosis kept effective training time above 90%. **Silent data corruption** — a chip that returns wrong numbers without crashing — is caught by monitoring losses and gradient norms ([Chapter 10](./10_pretraining_at_scale.md)).

## Frameworks

| Framework | Strengths |
|---|---|
| **Megatron-Core / NeMo** (NVIDIA) | TP/SP/PP/CP/EP, the reference for large-scale GPU training |
| **PyTorch FSDP2 + TorchTitan** | Native PyTorch 4D/5D parallelism, Float8, async TP, compile |
| **DeepSpeed** | ZeRO stages, offload and Ulysses; strong for fine-tuning on few GPUs |
| **JAX / MaxText, Pathways** | TPU-native sharding with named mesh axes ([Chapter 27](./27_asics_and_accelerators.md)) |
| **Hugging Face Accelerate / Nanotron** | Simple multi-GPU wrappers; educational large-scale code and the "Ultra-Scale Playbook" |

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
<p>Power limits are pushing frontier runs across multiple sites. Low-communication methods such as DiLoCo — each group trains locally for hundreds of steps and only occasionally synchronises an averaged update — cut cross-site bandwidth by orders of magnitude, and decentralised runs (Prime Intellect's INTELLECT series) used them to train over the public internet. Asynchronous RL ([Chapter 15](./15_alignment_rlhf_and_beyond.md)) applies the same idea to post-training: generation and training run on different machines and exchange weights only occasionally.</p>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A GB200 NVL72 rack puts 72 GPUs in one NVLink domain. If tensor and expert parallelism can now span 72 GPUs at NVLink speed instead of 8, which of the tricks in this chapter — pipeline parallelism, DualPipe, node-limited routing — become unnecessary, and which new bottleneck appears?</p>
</div>

## Key Takeaways

- Mixed-precision AdamW needs 16 bytes per parameter (1.13 TB for 70B) before activations; activations for one 8K sequence of a 70B model add ~183 GB even with FlashAttention.
- Every scheme is built from collectives; a ring all-reduce sends 2(N − 1)/N of the data per GPU, and NVLink is ~10× faster than the network between nodes.
- DP replicates and all-reduces gradients once per step, overlapped with backward; ZeRO/FSDP shard the state (16 → 4 bytes per parameter on 4 ranks) for 1–1.5× DP's communication.
- TP splits matrices with two all-reduces per block per pass — for a 70B model over InfiniBand they would take longer than the compute, so TP stays in NVLink; SP shards the rest of the block along the sequence.
- PP splits layers and pays a bubble of (p − 1)/(m + p − 1); 1F1B bounds memory, interleaving and zero-bubble schedules shrink the bubble, DualPipe hides communication.
- CP (ring attention, Ulysses) splits long sequences; EP splits experts with two all-to-alls per MoE layer.
- Real runs combine 4–5 dimensions on a device mesh, placing TP/EP innermost and DP outermost.

## Check Yourself

<details class="check">
<summary>How much per-GPU state does a 13B model need under ZeRO-3 on 16 GPUs?</summary>
<p>16 bytes × 13 × 10⁹ = 208 GB total, ÷ 16 = 13 GB per GPU, plus activations and the temporarily gathered parameters of the layer being computed.</p>
</details>

<details class="check">
<summary>Why does a ring all-reduce cost each GPU about 2× the message size regardless of how many GPUs take part?</summary>
<p>The data is split into N chunks. In the reduce-scatter phase each GPU sends one chunk per step for N − 1 steps, and the same again in the all-gather phase: 2(N − 1) chunks of size S/N, i.e. 2(N − 1)/N · S, which approaches 2S as N grows. Every link carries traffic at every step, so the bandwidth of all links is used in parallel.</p>
</details>

<details class="check">
<summary>Why are ZeRO stages 1 and 2 "free" in communication, while stage 3 costs 1.5×?</summary>
<p>An all-reduce is a reduce-scatter followed by an all-gather. ZeRO-1/2 reduce-scatter the gradients (each rank keeps the shard it updates), run the optimizer on that shard, then all-gather the updated parameters — the same bytes as DDP's all-reduce. ZeRO-3 does not keep full parameters, so it must all-gather them before the forward pass and again before the backward pass, plus reduce-scatter the gradients: three transfers instead of two.</p>
</details>

<details class="check">
<summary>Why does Megatron pair a column-parallel matmul with a row-parallel one?</summary>
<p>The column-parallel layer leaves each rank with a slice of the hidden dimension, which is exactly the input the row-parallel layer needs; the element-wise activation in between needs no communication. Only the row-parallel output needs one all-reduce, instead of a collective after each matmul.</p>
</details>

<details class="check">
<summary>With 8 pipeline stages, how many micro-batches keep the GPipe/1F1B bubble below 5%? And with 4-way interleaving?</summary>
<p>(p − 1)/(m + p − 1) < 0.05 → 7/(m + 7) < 0.05 → m > 133. With v = 4: 7/(4m + 7) < 0.05 → m > 33.25, so 34 micro-batches (in practice a multiple of p, i.e. 40). Interleaving buys a smaller global batch for the same bubble, at the cost of more point-to-point messages.</p>
</details>

<details class="check">
<summary>Why did DeepSeek-V3 limit each token's experts to at most 4 nodes?</summary>
<p>All-to-all traffic across the inter-node network is the bottleneck of expert parallelism. Capping the number of distinct nodes per token bounds the cross-node copies of each token (to 4 rather than up to 8), so the dispatch and combine can be overlapped with computation.</p>
</details>

<details class="check">
<summary>In a TP × CP × PP × DP mesh, why is TP the innermost (fastest-varying) dimension?</summary>
<p>Ranks that differ only in the innermost dimension are numerically adjacent, and adjacent ranks share a node and its NVLink switch. TP communicates in every layer of every micro-batch and needs that bandwidth; DP communicates once per step and can tolerate the slowest links, so it goes outermost.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Megatron-LM: Training Multi-Billion Parameter LMs Using Model Parallelism | Shoeybi et al., 2019 | [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053) | Tensor parallelism |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al., 2019 | [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) | ZeRO stages and their communication |
| GPipe | Huang et al., 2018 | [arxiv.org/abs/1811.06965](https://arxiv.org/abs/1811.06965) | Pipeline parallelism, bubbles |
| PipeDream: Fast and Efficient Pipeline Parallel DNN Training | Harlap et al., 2018 | [arxiv.org/abs/1806.03377](https://arxiv.org/abs/1806.03377) | 1F1B scheduling |
| Efficient Large-Scale LM Training on GPU Clusters (PTD-P) | Narayanan et al., 2021 | [arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473) | Interleaved 1F1B, 3D parallelism |
| Reducing Activation Recomputation in Large Transformer Models | Korthikanti et al., 2022 | [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198) | Sequence parallelism, activation memory, selective recompute |
| Zero Bubble Pipeline Parallelism | Qi et al., 2023 | [arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241) | Split backward to fill bubbles |
| Ring Attention with Blockwise Transformers | Liu et al., 2023 | [arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889) | Ring context parallelism |
| DeepSpeed Ulysses | Jacobs et al., 2023 | [arxiv.org/abs/2309.14509](https://arxiv.org/abs/2309.14509) | All-to-all context parallelism |
| PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | Zhao et al., 2023 | [arxiv.org/abs/2304.11277](https://arxiv.org/abs/2304.11277) | FSDP design |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | 4D configuration, MFU, failure statistics |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | DualPipe, EP, node-limited routing, FP8 dispatch |
| DeepEP | DeepSeek-AI, 2025 | [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | Expert-parallel all-to-all kernels |
| TorchTitan | Liang et al., 2024 | [arxiv.org/abs/2410.06511](https://arxiv.org/abs/2410.06511) | PyTorch-native multi-dimensional parallelism |
| DiLoCo: Distributed Low-Communication Training | Douillard et al., 2023 | [arxiv.org/abs/2311.08105](https://arxiv.org/abs/2311.08105) | Cross-site training |
| NCCL documentation: collective operations | NVIDIA | [docs.nvidia.com/deeplearning/nccl](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | Collective semantics |

---

**Next:** [Chapter 27 — ASICs & Specialized Accelerators](./27_asics_and_accelerators.md) — TPUs, Trainium, wafer-scale engines and the rest of the non-GPU world.

[← Back to Table of Contents](./README.md)
