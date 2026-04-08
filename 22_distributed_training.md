[← Back to Table of Contents](./README.md)

# Chapter 22 — Distributed Training

> *"No single GPU can train a frontier model. The art of distributed training is splitting computation across thousands of GPUs while minimizing communication overhead."*

GPT-4, LLaMA-3 405B, and DeepSeek-V3 were each trained on thousands of GPUs for months. This chapter covers the parallelism strategies that make this possible.

## Why Distribute?

The largest models today have hundreds of billions of parameters. A single GPU can't hold the model, optimizer states, gradients, and activations simultaneously:

| Component | LLaMA-3 70B (BF16) | LLaMA-3 405B (BF16) |
|-----------|--------------------|--------------------|
| Model weights | 140 GB | 810 GB |
| Adam optimizer states | 560 GB | 3,240 GB |
| Gradients | 140 GB | 810 GB |
| Activations (B=1, T=8192) | ~30 GB | ~170 GB |
| **Total** | **~870 GB** | **~5,030 GB** |
| H100 80GB GPUs needed | minimum 11 | minimum 63 |

## The Three Dimensions of Parallelism

<div class="diagram">
<div class="diagram-title">3D Parallelism</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">Data Parallelism (DP)</div>
    <div class="card-desc">Replicate model on each GPU. Split the batch across GPUs. All-reduce gradients after backward pass.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Tensor Parallelism (TP)</div>
    <div class="card-desc">Split individual weight matrices across GPUs (column-wise or row-wise). Requires all-reduce within each layer.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Pipeline Parallelism (PP)</div>
    <div class="card-desc">Assign groups of layers to different GPUs. Forward activations flow GPU-to-GPU. Reduce idle "bubble" time with micro-batches.</div>
  </div>
</div>
</div>

Frontier training combines all three — **3D parallelism** — plus additional techniques like sequence parallelism and expert parallelism (for MoE models).

## Data Parallelism (DDP)

The simplest strategy: each GPU has a full copy of the model and processes different data.

<div class="diagram">
<div class="diagram-title">DDP All-Reduce Gradient Sync</div>
<div class="flow-h">
  <div class="flow-step green">GPU 0: forward + backward → ∇W₀</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step green">GPU 1: forward + backward → ∇W₁</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step green">GPU 2: forward + backward → ∇W₂</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step accent">All-Reduce: ∇W = mean(∇W₀, ∇W₁, ∇W₂)</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step orange">Each GPU updates with same ∇W</div>
</div>
</div>

```python
# PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")  # NVIDIA Collective Communications Library
local_rank = int(os.environ["LOCAL_RANK"])

model = MyTransformer().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop is identical — DDP handles gradient sync automatically
for batch in dataloader:
    loss = model(batch)
    loss.backward()       # gradients synced via all-reduce
    optimizer.step()
    optimizer.zero_grad()

# Launch: torchrun --nproc_per_node=8 train.py
```

**Limitation**: every GPU must hold the full model + optimizer states. For a 70B model, that's ~870 GB — impossible with DDP alone.

## FSDP / ZeRO: Sharded Data Parallelism

**Fully Sharded Data Parallel (FSDP)** — PyTorch's implementation of DeepSpeed ZeRO — shards not just data but optimizer states, gradients, and even parameters across GPUs:

| ZeRO Stage | What is sharded | Memory per GPU (70B, 8 GPUs) |
|------------|----------------|------|
| Stage 0 (DDP) | Nothing — full replicas | ~870 GB |
| Stage 1 | Optimizer states | ~210 GB |
| Stage 2 | + Gradients | ~158 GB |
| Stage 3 / FSDP Full | + Parameters | ~109 GB |

```python
# PyTorch FSDP (Fully Sharded Data Parallel)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    auto_wrap_policy=transformer_auto_wrap_policy,
)

# Training loop remains the same
# FSDP all-gathers parameters before forward, shards after backward
```

The tradeoff: FSDP introduces more communication (all-gather before forward, reduce-scatter after backward) but dramatically reduces per-GPU memory.

## Tensor Parallelism (TP)

Tensor parallelism splits individual weight matrices across GPUs. For a linear layer `Y = XW`:

<div class="diagram">
<div class="diagram-title">Column-Parallel Linear</div>
<div class="flow-h">
  <div class="flow-step accent">X [B, T, d]</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step green">GPU 0: X × W₁ → Y₁ [B, T, d/2]</div>
  <div class="flow-arrow">↘</div>
  <div class="flow-step purple">All-Gather → Y [B, T, d]</div>
</div>
<div class="flow-h" style="margin-top: 0">
  <div class="flow-step" style="visibility: hidden">X [B, T, d]</div>
  <div class="flow-arrow" style="visibility: hidden">→</div>
  <div class="flow-step orange">GPU 1: X × W₂ → Y₂ [B, T, d/2]</div>
  <div class="flow-arrow">↗</div>
  <div class="flow-step" style="visibility: hidden">concat</div>
</div>
</div>

In a transformer, TP is applied to:
- **Attention**: Q, K, V projections split column-wise across heads; output projection split row-wise
- **FFN**: first linear split column-wise, second split row-wise

**Constraint**: TP requires fast interconnect (NVLink/NVSwitch) since it communicates within every layer. Typically used within a single node (8 GPUs).

<div class="tensor-shape">
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Input (replicated):</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim feature">d</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Weight shard (GPU k):</span>
    <span class="ts-dim feature">d</span> × <span class="ts-dim feature">d/TP</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Output shard (GPU k):</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim feature">d/TP</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After all-gather:</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim feature">d</span>
  </div>
</div>
</div>

## Pipeline Parallelism (PP)

Pipeline parallelism assigns different layers to different GPUs:

<div class="diagram">
<div class="diagram-title">Pipeline Parallelism with Micro-batches</div>
<div class="layer-stack">
  <div class="layer green">GPU 0 — Layers 0–7: [μ1]→[μ2]→[μ3]→[μ4]  idle  [μ4']→[μ3']→[μ2']→[μ1']</div>
  <div class="layer accent">GPU 1 — Layers 8–15:  idle [μ1]→[μ2]→[μ3]→[μ4] [μ4']→[μ3']→[μ2']→[μ1'] idle</div>
  <div class="layer purple">GPU 2 — Layers 16–23:  idle  idle [μ1]→[μ2]→[μ3]→[μ4] → backward →</div>
  <div class="layer orange">GPU 3 — Layers 24–31:  idle  idle  idle [μ1]→[μ2]→[μ3]→[μ4] → backward →</div>
</div>
</div>

The **pipeline bubble** (idle time) is reduced by splitting a batch into micro-batches. Schedules like **1F1B** (one forward, one backward) and **interleaved PP** minimize the bubble.

- Bubble fraction ≈ (PP_size - 1) / n_microbatches
- With 4 pipeline stages and 32 micro-batches: ~9% bubble overhead

## Communication Patterns

| Operation | When Used | Data Volume | Requires |
|-----------|-----------|------------|----------|
| **All-Reduce** | DDP gradient sync | 2 × model_size | Inter-node OK |
| **All-Gather** | FSDP param collection, TP column-parallel | model_size / world_size | Fast interconnect preferred |
| **Reduce-Scatter** | FSDP gradient sharding | model_size / world_size | Inter-node OK |
| **Point-to-Point** | Pipeline parallelism activations | activation_size | Between adjacent stages |
| **All-to-All** | MoE expert routing | tokens × d | Needs careful scheduling |

## Putting It All Together: 3D Parallelism

A practical configuration for training LLaMA-3 405B on 1024 H100 GPUs:

<div class="diagram">
<div class="diagram-title">3D Parallelism Configuration Example</div>
<div class="layer-stack">
  <div class="layer green">Data Parallelism (DP=16) — 16 replicas, each processes different data</div>
  <div class="layer accent">Tensor Parallelism (TP=8) — within each node (NVLink), splits attention heads + FFN</div>
  <div class="layer purple">Pipeline Parallelism (PP=8) — across 8 nodes, splits 126 layers into 8 stages</div>
</div>
<div style="text-align: center; color: var(--text-secondary); margin-top: 0.5rem;">
  Total GPUs: DP × TP × PP = 16 × 8 × 8 = 1,024 H100s
</div>
</div>

## DeepSpeed Integration

```python
# DeepSpeed ZeRO-3 config
ds_config = {
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 128,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "none"},
        "offload_optimizer": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
    },
}

# Initialize with HuggingFace Accelerate
from accelerate import Accelerator
accelerator = Accelerator(deepspeed_plugin=ds_config)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

## What's Next

Not all accelerators are NVIDIA GPUs. The next chapter surveys **ASICs and alternative accelerators** — TPUs, Gaudi, Groq LPUs, and more.

[← Previous: Chapter 21 — CUDA & Kernel Development](./21_cuda_and_kernel_development.md) · **Next: [Chapter 23 — ASICs & Accelerators →](./23_asics_and_accelerators.md)**

---

*Last updated: April 2026*
