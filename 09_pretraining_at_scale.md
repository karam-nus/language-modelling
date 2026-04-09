---
title: "Chapter 9 — Pre-Training at Scale"
---

[← Back to Table of Contents](./README.md)

# Chapter 9 — Pre-Training at Scale

> *"The remarkable thing about LLMs isn't the architecture — it's that next-token prediction on internet-scale data produces intelligence."*

## The Pre-Training Pipeline

Pre-training is where the bulk of compute and data investment goes. A modern LLM pre-training run involves months of GPU time, trillions of tokens, and careful orchestration.

<div class="diagram">
<div class="diagram-title">Pre-Training Data Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Raw Data: Common Crawl, GitHub, Wikipedia, books, etc. <small>~100 TB raw text</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Quality Filtering: language ID, perplexity filter, URL blocklist, classifier <small>remove low-quality pages</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Deduplication: exact + fuzzy (MinHash/SimHash) <small>~30-50% data removed</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">PII/Toxicity Removal: regex, classifiers <small>safety filtering</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Tokenization: BPE/SentencePiece → token sequences <small>~15 trillion tokens</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Packing & Shuffling: concatenate, pack to max_seq_len, global shuffle</div>
</div>
</div>

## Training Objective

The training objective is deceptively simple — minimize cross-entropy loss at every position:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

where $P_\theta$ is the model's predicted probability for the correct next token.

This is computed efficiently: a single forward pass processes all T tokens in parallel (thanks to causal masking), and the loss is computed at every position simultaneously.

```python
import torch
import torch.nn.functional as F

def compute_lm_loss(model, input_ids):
    """Standard causal LM loss computation."""
    # input_ids: [B, T]
    outputs = model(input_ids[:, :-1])          # predict from tokens 0..T-2
    logits = outputs.logits                      # [B, T-1, vocab_size]

    targets = input_ids[:, 1:]                   # targets are tokens 1..T-1
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),     # [B*(T-1), vocab_size]
        targets.reshape(-1),                      # [B*(T-1)]
    )
    return loss
```

## Learning Rate Schedule

Almost all modern LLMs use a **warmup + cosine decay** schedule:

<div class="diagram">
<div class="diagram-title">Learning Rate Schedule</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Warmup Phase</div>
    <ul>
      <li>Linear ramp from ~0 to peak LR</li>
      <li>Typically 0.1-2% of total steps (500–2000 steps)</li>
      <li>Stabilizes training at the start</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Cosine Decay Phase</div>
    <ul>
      <li>Smoothly decays from peak to min LR</li>
      <li>Min LR typically 0.1× or 0.01× of peak</li>
      <li>Gradual decay preserves learned representations</li>
    </ul>
  </div>
</div>
</div>

```python
import math

def cosine_lr_schedule(step, warmup_steps, total_steps, max_lr, min_lr):
    """Warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### Typical Hyperparameters

| Hyperparameter | Small (1-3B) | Medium (7-13B) | Large (70B+) |
|---------------|:---:|:---:|:---:|
| Peak Learning Rate | 3e-4 | 3e-4 | 1.5e-4 |
| Min Learning Rate | 3e-5 | 3e-5 | 1.5e-5 |
| Warmup Steps | 2000 | 2000 | 2000 |
| Weight Decay | 0.1 | 0.1 | 0.1 |
| Batch Size (tokens) | 1-4M | 4-8M | 8-16M |
| Optimizer | AdamW | AdamW | AdamW |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| Gradient Clipping | 1.0 | 1.0 | 1.0 |
| Total Tokens | 1-3T | 3-15T | 15T+ |

## Pre-Training Datasets

| Dataset | Size (tokens) | Sources | Used By |
|---------|:---:|---------|---------|
| **FineWeb** | 15T | CommonCrawl (filtered) | Community standard |
| **FineWeb-Edu** | 1.3T | FineWeb educational subset | Quality benchmark |
| **RedPajama v2** | 30T | CommonCrawl, multi-source | Open replication |
| **The Pile** | 300B | 22 diverse sources | GPT-NeoX, early open models |
| **Dolma** | 3T | CommonCrawl, code, books, papers | OLMo |
| **StarCoder data** | 780B | GitHub (licensed code) | StarCoder, CodeLLaMA |

### Data Mix

Models aren't trained on just web text. The training mix typically includes:

<div class="diagram">
<div class="diagram-title">Typical Pre-Training Data Mix</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card accent">
    <div class="card-title">Web Text ~80%</div>
    <div class="card-desc">CommonCrawl filtered for quality. The bulk of training data.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Code ~10%</div>
    <div class="card-desc">GitHub, StackOverflow. Improves reasoning and code generation.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Books/Papers ~5%</div>
    <div class="card-desc">High quality, long-form. Knowledge and coherence.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Curated ~5%</div>
    <div class="card-desc">Wikipedia, math, scientific data. Factual accuracy.</div>
  </div>
</div>
</div>

## Training Loop with Gradient Accumulation

Real LLM training uses large effective batch sizes (millions of tokens) achieved through gradient accumulation across micro-batches:

```python
def train_step(model, optimizer, dataloader, accumulation_steps, max_grad_norm=1.0):
    """Training loop with gradient accumulation and mixed precision."""
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for micro_step in range(accumulation_steps):
        batch = next(dataloader)  # [micro_batch_size, seq_len]
        input_ids = batch["input_ids"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids[:, :-1], labels=input_ids[:, 1:])
            loss = outputs.loss / accumulation_steps

        loss.backward()
        total_loss += loss.item()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return total_loss
```

Example: micro_batch_size=4, seq_len=8192, accumulation_steps=32, 8 GPUs with DDP → effective batch = 4 × 8192 × 32 × 8 = **~8M tokens per step**.

## Training Infrastructure

<div class="diagram">
<div class="diagram-title">Pre-Training Infrastructure Stack</div>
<div class="layer-stack">
  <div class="layer accent">Training Framework: PyTorch + FSDP / DeepSpeed / Megatron-LM</div>
  <div class="layer purple">Distributed Strategy: 3D Parallelism (TP + PP + DP) — see Ch 22</div>
  <div class="layer green">Mixed Precision: BF16 compute, FP32 master weights — see Ch 16</div>
  <div class="layer orange">Hardware: NVIDIA H100 clusters (100s–10000s of GPUs)</div>
  <div class="layer cyan">Networking: 400 Gbps InfiniBand / NVLink, NCCL collective comms</div>
</div>
</div>

### Checkpointing and Recovery

Training runs crash. Hardware fails. Pre-training runs need robust checkpointing:

- **Periodic checkpoints**: Save every 500–1000 steps (model + optimizer state + RNG state)
- **Async checkpointing**: Save to storage without blocking training
- **Loss spike detection**: Monitor for loss spikes indicative of instability
- **Automatic restart**: Resume from last checkpoint on failure
- **Deterministic recovery**: Save RNG states for exact reproducibility

## Loss Curves

A healthy pre-training loss curve shows rapid initial improvement followed by slow, steady decline:

| Phase | Steps | Behavior |
|-------|:-----:|----------|
| **Early** | 0–1K | Rapid drop from ~11 (random) to ~4–5 |
| **Middle** | 1K–100K | Steady decline, ~log-linear |
| **Late** | 100K+ | Diminishing returns, approaching data limit |
| **Anomalies** | — | Loss spikes → learning rate issues, data quality, hardware faults |

Typical final pre-training loss: **~1.5–2.0** cross-entropy on general web text for a well-trained 7B model.

## Compute Estimation

The compute needed for one forward+backward pass through the full dataset:

$$C \approx 6ND$$

where $N$ = parameters, $D$ = tokens, and the factor 6 accounts for forward (2N FLOPs per token) + backward (4N FLOPs per token).

| Model | Params (N) | Tokens (D) | Compute (6ND) | GPU-hours (H100) |
|-------|:---:|:---:|:---:|:---:|
| LLaMA-3 8B | 8B | 15T | 7.2×10²³ | ~180K |
| LLaMA-3 70B | 70B | 15T | 6.3×10²⁴ | ~1.6M |
| LLaMA-3.1 405B | 405B | 15T | 3.6×10²⁵ | ~9M |

## What's Next

Pre-training produces a raw "base model" — capable but not aligned to human instructions or specialized domains. The next chapter covers **mid-training**: the intermediate stage where models gain long-context ability and domain specialization before fine-tuning.

[← Previous: Chapter 8 — Multimodal Models](./08_multimodal_models.md) · **Next: [Chapter 10 — Mid-Training →](./10_mid_training.md)**

---

*Last updated: April 2026*
