---
title: "Chapter 10 — Pre-Training at Scale"
description: "The data pipeline, the loss and how to compute it without the off-by-one bug, warmup-stable-decay schedules, multi-token prediction, FP8 and FP4 training, the 6ND compute rule checked against published runs, MFU, and keeping a 16,000-GPU job alive."
---

[← Back to Table of Contents](./README.md)

# Chapter 10 — Pre-Training at Scale

Pre-training is where most of the data and a large share of the compute of a model's life is spent: tens of trillions of tokens, thousands to hundreds of thousands of GPUs, weeks to months of wall-clock time. This chapter is about what such a run actually consists of.

> **The one-sentence version:** pre-training is next-token prediction over a filtered, deduplicated, carefully mixed corpus, at a scale where the engineering problems — numerics, schedules, throughput and hardware failures — matter as much as the objective.

## The Pre-Training Pipeline

<div class="diagram">
<div class="diagram-title">Pre-Training Data Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Raw data: Common Crawl, code, books, papers, licensed and synthetic sources <small>petabytes of raw HTML and text</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Extraction + quality filtering: language ID, heuristics, URL blocklists, model-based classifiers <small>often keep only 10–30% of pages</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Deduplication: exact (hash) + fuzzy (MinHash LSH) <small>removes a large fraction of web text</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">PII / toxicity / decontamination filtering <small>safety and benchmark hygiene</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Tokenization → token shards <small>10–40 trillion tokens for a 2025–26 frontier run</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Mixing, packing & shuffling: sample sources by weight, pack to max_seq_len, shuffle globally</div>
</div>
</div>

[Chapter 9](./09_data_for_llms.md) covers the datasets and data mixes themselves.

## Training Objective

Minimise cross-entropy at every position:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

One forward pass over a `[B, T]` batch produces T predictions per sequence, thanks to the causal mask.

```python
import torch.nn.functional as F

def lm_loss(model, input_ids):              # input_ids: [B, T]
    logits = model(input_ids).logits         # [B, T, V]
    # position t predicts token t+1: drop the last logit and the first label
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),   # [B*(T-1), V]
                           input_ids[:, 1:].reshape(-1))                   # [B*(T-1)]
```

<div class="callout warn">
<span class="callout-title">The double-shift bug</span>
<p>Hugging Face models shift labels <em>internally</em>. Call them as <code>model(input_ids, labels=input_ids)</code>. Passing <code>model(input_ids[:, :-1], labels=input_ids[:, 1:])</code> — a pattern copied into many tutorials — shifts twice and silently trains the model to predict two tokens ahead. Loss still goes down, just to a worse value.</p>
</div>

### Multi-Token Prediction

DeepSeek-V3 (2024) added a **multi-token prediction (MTP)** objective: small extra modules, chained after the main model, predict tokens t+2 (and further) from the main model's hidden state. The extra loss densifies the training signal and, at inference, the MTP head doubles as a built-in draft model for speculative decoding ([Chapter 16](./16_inference_and_sampling.md)) — DeepSeek reports 85–90% acceptance for the second token. MTP has since appeared in GLM-4.5, Qwen3-Next and DeepSeek-V4.

## Learning-Rate Schedules

The classic schedule is **linear warmup + cosine decay** to ~10% of peak. Since 2024 many labs instead use **warmup-stable-decay (WSD)**: hold the peak learning rate for most of training, then decay quickly (often linearly or as 1−√) over the last 10–20% of tokens.

<div class="diagram">
<div class="diagram-title">Cosine vs Warmup-Stable-Decay</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Warmup + cosine</div>
    <ul>
      <li>Linear ramp to peak over ~0.1–1% of steps</li>
      <li>Smooth decay to ~0.1× peak over the whole run</li>
      <li>Total length must be fixed in advance</li>
      <li>Used by Llama 3, OLMo 2, most 2020–24 runs</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Warmup-stable-decay</div>
    <ul>
      <li>Long constant phase at peak LR</li>
      <li>Short final "cool-down", where the loss drops sharply</li>
      <li>Branch a decay from any checkpoint: cheap scaling-law sweeps, easy continuation</li>
      <li>Pairs naturally with data annealing (Chapter 12); MiniCPM, DeepSeek, many 2025–26 runs</li>
    </ul>
  </div>
</div>
</div>

```python
import math

def lr_at(step, total, peak, warmup=2000, decay_frac=0.2, floor=0.0, kind="wsd"):
    if step < warmup:                                   # linear warmup
        return peak * step / warmup
    if kind == "cosine":
        p = (step - warmup) / (total - warmup)
        return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))
    decay_start = int(total * (1 - decay_frac))         # WSD
    if step < decay_start:
        return peak
    p = (step - decay_start) / (total - decay_start)
    return floor + (peak - floor) * (1 - p)             # linear cool-down
```

### Typical Hyperparameters

| Hyperparameter | Small (1–3B) | Medium (7–30B) | Large (70B+ dense / big MoE) |
|---------------|:---:|:---:|:---:|
| Peak learning rate (AdamW) | 3e-4 – 1e-3 | 3e-4 | 8e-5 – 2e-4 (Llama 3 405B: 8e-5) |
| Warmup | 1–2K steps | 2K steps | 2–8K steps |
| Weight decay | 0.1 | 0.1 | 0.1 |
| Global batch (tokens) | 1–4M | 4–8M | 8–60M, often ramped up during training |
| Sequence length | 4–8K | 8K | 8K (then extended in mid-training) |
| Optimizer | AdamW (or Muon) | AdamW (or Muon) | AdamW; Muon in Kimi K2, DeepSeek-V4 |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Total tokens | 2–36T | 15–36T | 15–40T |

Note the "total tokens" row: even 1B models are now trained on trillions of tokens — far past the Chinchilla-optimal ~20 tokens per parameter — because a smaller model trained longer is cheaper to *serve* ([Chapter 32](./32_scaling_laws.md)).

## Training Loop with Gradient Accumulation

Large effective batches (millions of tokens) are built from micro-batches across GPUs and accumulation steps:

```python
import torch

def train_step(model, optimizer, scheduler, micro_batches, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n_tokens = sum((mb["labels"] != -100).sum() for mb in micro_batches)   # normalise by tokens, not micro-steps
    total = 0.0
    for mb in micro_batches:                                   # each: [micro_bs, seq_len]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(mb["input_ids"].cuda()).logits
        loss_sum = torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1).float(), mb["labels"][:, 1:].flatten().cuda(),
            ignore_index=-100, reduction="sum")
        (loss_sum / n_tokens).backward()
        total += loss_sum.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step(); scheduler.step()
    return total / n_tokens.item()
```

Dividing each micro-batch's *mean* loss by the number of accumulation steps — the obvious approach — is wrong when micro-batches contain different numbers of real tokens (padding, masked prompts): short micro-batches get over-weighted. Hugging Face fixed exactly this bug across its trainers in late 2024. Summing the loss and dividing by the total token count, as above, matches a single big batch.

Example: micro-batch 4 × 8,192 tokens, 32 accumulation steps, 64 data-parallel GPUs → 4 × 8,192 × 32 × 64 ≈ **67M tokens per optimizer step**.

## Numerics: BF16, FP8 and FP4 Training

<div class="diagram">
<div class="diagram-title">Pre-Training Infrastructure Stack</div>
<div class="layer-stack">
  <div class="layer accent">Framework: PyTorch + FSDP2 / Megatron-Core / DeepSpeed; TorchTitan, NeMo, custom stacks at labs</div>
  <div class="layer purple">Parallelism: data + tensor + pipeline + expert + context parallel — Chapter 26</div>
  <div class="layer green">Precision: BF16 compute with FP32 master weights; FP8 (and emerging FP4) matmuls — Chapter 19</div>
  <div class="layer orange">Hardware: H100/H200 → B200/GB200/GB300 NVL72 racks, TPU v5p/v6/7, Trainium 2/3 — Chapters 24, 27</div>
  <div class="layer cyan">Network: NVLink inside a node or rack; 400–800 Gb/s InfiniBand or RoCE between them; NCCL collectives</div>
</div>
</div>

BF16 mixed precision was the default from 2020 to 2024. Two newer steps:

- **FP8 training.** DeepSeek-V3 (2024) trained a 671B MoE with FP8 matrix multiplies using fine-grained scaling (1×128 tiles for activations, 128×128 blocks for weights) and higher-precision accumulation, with loss within 0.25% of a BF16 baseline. FP8 recipes (and MXFP8 on Blackwell) are now standard at the frontier.
- **FP4 training.** NVIDIA demonstrated NVFP4 pre-training of a 12B model on 10T tokens in 2025, matching FP8 loss closely using random Hadamard transforms, stochastic rounding and a few BF16 layers. Expect FP4 in production pre-training as GB300-class hardware spreads.

## Stability: Keeping the Loss Curve Boring

A healthy curve drops fast — from ln V ≈ 11.8 nats at initialisation (uniform over 128K tokens) to around 4 within a few thousand steps — then declines slowly and roughly as a power law. Absolute final values depend on the tokenizer and data and are not comparable across models.

**Loss spikes** are the enemy: a sudden jump that sometimes recovers and sometimes diverges. Common causes and fixes:

| Cause | Fix |
|---|---|
| Attention logits growing without bound | QK-norm, QK-clip (Kimi K2's MuonClip), softcapping (Gemma 2) |
| Output-logit drift | z-loss (penalise log Z, PaLM), tied or normalised embeddings |
| Bad data batch | Skip the batch, rewind to the last checkpoint with a different data order |
| LR too high for the batch | Lower peak LR, longer warmup, batch-size ramp |
| Low-precision overflow | Better scaling granularity, keep sensitive layers in BF16 |

Kimi K2 reported pre-training a 1T-parameter MoE on 15.5T tokens with **zero** loss spikes, attributing it to MuonClip — a sign that stability is now an engineering deliverable, not luck.

## Fault Tolerance

At 16,000 GPUs, hardware failure is a daily event. Llama 3 405B's 54-day pre-training snapshot had **466 job interruptions**, 419 of them unexpected — about 78% traced to hardware (GPUs, HBM, network) — yet achieved over 90% effective training time thanks to:

- **Frequent, asynchronous checkpoints** (model + optimizer + data-loader + RNG state) written in parallel shards without pausing training.
- **Automated health checks and restarts**: detect a stuck or slow rank, cordon the node, restart from the last checkpoint in minutes.
- **Straggler detection**: one slow GPU slows every synchronous collective.
- **Silent data corruption checks**: some faulty chips produce wrong numbers without crashing; loss and gradient-norm monitors catch them.

[Chapter 13 of the Chips guide](https://karam-nus.github.io/chips/13_reliability_and_fault_tolerance.html) goes deeper on the hardware side.

## Compute Estimation

A dense transformer needs about **6 FLOPs per parameter per token** for training — 2 in the forward pass and 4 in the backward pass:

$$C \approx 6ND$$

For an MoE model, N is the **active** parameter count. Attention adds 6·L·T·d per token on top (non-negligible at long context). Checked against published runs:

| Model | N (active) | D (tokens) | 6ND | Reported GPU-hours | Implied throughput |
|-------|:---:|:---:|:---:|:---:|:---:|
| Llama 3.1 8B | 8B | 15T | 7.2 × 10²³ | 1.46M (H100) | ~137 TFLOP/s per GPU |
| Llama 3.1 70B | 70B | 15T | 6.3 × 10²⁴ | 7.0M (H100) | ~250 TFLOP/s |
| Llama 3.1 405B | 405B | 15.6T | 3.8 × 10²⁵ | 30.8M (H100) | ~340 TFLOP/s |
| DeepSeek-V3 | 37B of 671B | 14.8T | 3.3 × 10²⁴ | 2.79M (H800) | ~330 TFLOP/s |

The H100's dense BF16 peak is 989 TFLOP/s, so the bigger runs reach **MFU** (model FLOPs utilisation) around 35–40% — Meta reports 38–43% for 405B. (Reported GPU-hours include restarts and evaluation, so implied throughput understates the steady state, especially for the smaller models.) DeepSeek-V3's row shows why MoE took over: a model with 671B parameters' worth of capacity for a tenth of Llama 405B's compute, aided by FP8.

Try your own budget — parameters, tokens, GPU type, utilisation, cluster size:

<div class="lab" data-lab="compute"><noscript>Interactive lab: estimate training FLOPs, GPU-hours, wall-clock days and cost from parameters, tokens, GPU type and MFU. Enable JavaScript to use it.</noscript></div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>DeepSeek-V3 used about a tenth of Llama 3.1 405B's training compute and matched or beat it on most benchmarks. How much of that gap is architecture (MoE, MLA), how much numerics (FP8), and how much data and post-training — and how would you design an experiment to tell?</p>
</div>

## Key Takeaways

- The pipeline is extract → filter (now classifier-driven) → deduplicate → decontaminate → tokenize → mix and pack.
- Get the loss right: one shift only, token-weighted accumulation, chunked logits for big vocabularies.
- WSD schedules (long constant LR, short decay) have joined cosine as a default because they make continuation and scaling sweeps cheap.
- FP8 training is mainstream at the frontier; FP4 has been demonstrated; stability tools (QK-norm, z-loss, MuonClip) keep loss curves spike-free.
- C ≈ 6·N_active·D; at 35–40% MFU, a 15T-token 405B dense run costs ~30M H100-hours — and MoE cuts that by roughly an order of magnitude for comparable quality.

## Check Yourself

<details class="check">
<summary>What is the cross-entropy at initialisation for a 151,936-token vocabulary, and why?</summary>
<p>A randomly initialised model predicts close to uniformly, so the loss is ln(151,936) ≈ 11.93 nats. A first-step loss far from ln V suggests a bug (wrong labels, wrong initialisation, or logits scaled badly).</p>
</details>

<details class="check">
<summary>Estimate training FLOPs for a 30B-total / 3B-active MoE trained on 36T tokens. How many H100-days at 40% MFU?</summary>
<p>6 × 3 × 10⁹ × 36 × 10¹² = 6.5 × 10²³ FLOPs. At 0.4 × 989 TFLOP/s ≈ 396 TFLOP/s per GPU: 6.5 × 10²³ / 3.96 × 10¹⁴ ≈ 1.6 × 10⁹ GPU-seconds ≈ 19,000 GPU-days (e.g., ~19 days on 1,024 GPUs), ignoring attention FLOPs.</p>
</details>

<details class="check">
<summary>Why does WSD make scaling-law experiments cheaper than cosine?</summary>
<p>A cosine schedule's shape depends on the total length, so each token budget needs its own run. With WSD, one long constant-LR run can be branched at several checkpoints, each followed by a short decay, giving fully trained models at many budgets for the price of one run plus a few cool-downs.</p>
</details>

<details class="check">
<summary>Two micro-batches have 1,000 and 100 real tokens with mean losses 2.0 and 4.0. What loss does "average the means" give, and what is correct?</summary>
<p>Averaging means: (2.0 + 4.0)/2 = 3.0. Token-weighted: (1,000 × 2.0 + 100 × 4.0) / 1,100 ≈ 2.18. The naive version over-weights the short micro-batch by 10×.</p>
</details>

<details class="check">
<summary>Llama 3 405B lost a job roughly every three hours. Why does that make checkpoint speed matter as much as checkpoint frequency?</summary>
<p>Work since the last checkpoint is lost on each failure, so you want frequent checkpoints — but if saving blocks training, frequent saves eat throughput. Asynchronous, sharded checkpointing makes saves cheap enough to do often, and fast restarts minimise the downtime per failure.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Scaling Laws for Neural Language Models | Kaplan et al., 2020 | [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361) | 6ND compute accounting |
| PaLM | Chowdhery et al., 2022 | [arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311) | z-loss, MFU definition |
| MiniCPM | Hu et al., 2024 | [arxiv.org/abs/2404.06395](https://arxiv.org/abs/2404.06395) | Warmup-stable-decay schedule |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | Batch schedule, MFU, interruptions |
| Llama 3.1 model card | Meta AI, 2024 | [github.com/meta-llama/llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | Reported GPU-hours |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | FP8 training, MTP, GPU-hours |
| Kimi K2: Open Agentic Intelligence | Moonshot AI, 2025 | [arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534) | MuonClip, spike-free training |
| Pretraining LLMs with NVFP4 | NVIDIA, 2025 | [arxiv.org/abs/2509.25149](https://arxiv.org/abs/2509.25149) | FP4 pre-training recipe |
| Fixing Gradient Accumulation | Hugging Face, 2024 | [huggingface.co/blog/gradient_accumulation](https://huggingface.co/blog/gradient_accumulation) | Token-weighted loss normalisation |

---

**Next:** [Chapter 11 — Optimizers & Loss Functions](./11_optimizers_and_loss_functions.md) — the update rules that turn these gradients into a model.

[← Back to Table of Contents](./README.md)
