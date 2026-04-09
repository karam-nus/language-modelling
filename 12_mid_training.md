---
title: "Chapter 12 — Mid-Training & Continued Pre-Training"
---

[← Back to Table of Contents](./README.md)

# Chapter 12 — Mid-Training & Continued Pre-Training

> *"Mid-training is the bridge between raw pre-training and task-specific fine-tuning — where models gain long-context ability, domain expertise, and code fluency."*

## What Is Mid-Training?

Mid-training (also called "continued pre-training" or "post-training phase 0") is the stage between base pre-training and instruction fine-tuning. It uses the same next-token prediction objective as pre-training, but with a curated data mix designed for specific capabilities.

<div class="diagram">
<div class="diagram-title">Full Training Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Pre-Training <small>~15T tokens, general web data, 100% compute</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Mid-Training <small>~100B–1T tokens, curated domain data, 1–10% compute</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Supervised Fine-Tuning (SFT) <small>~1M examples, instruction-response pairs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Alignment (RLHF/DPO) <small>~100K preferences, safety + helpfulness</small></div>
</div>
</div>

## Long-Context Extension

One of the most important mid-training applications: extending a model's context window from 8K to 128K+ tokens.

### The Challenge

Models pre-trained with 8K context can't simply be used with 128K inputs — the positional encoding (RoPE) hasn't seen those positions, and attention patterns degrade. The solution: progressive context extension during mid-training.

### RoPE Scaling Methods

| Method | Description | Models |
|--------|-------------|--------|
| **Position Interpolation** | Scale frequencies by L_new/L_old — compress positions to fit | CodeLLaMA (16K→100K) |
| **NTK-aware Scaling** | Adjust RoPE base frequency instead of scaling | YaRN |
| **YaRN** | NTK + attention scaling + temperature correction | Mistral, various |
| **ABF (Adjusted Base Frequency)** | Increase θ_base (e.g., 500K → 8M) | LLaMA-3.1 (128K) |

LLaMA-3.1 increased RoPE base frequency from 500,000 to 8,000,000 and trained on progressively longer sequences:

<div class="diagram">
<div class="diagram-title">LLaMA-3.1 Context Extension Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Stage 0: Pre-train at 8K context <small>15T tokens, θ_base = 500K</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Stage 1: Extend to 16K <small>~800B tokens, short documents</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Stage 2: Extend to 64K <small>~200B tokens, medium documents</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Stage 3: Extend to 128K <small>~100B tokens, long documents, θ_base = 8M</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Context Extension Tensor Impact</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Short context</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">8192</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">pre-training context</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Extended context</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">131072</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">16× more attention compute + KV-cache</span>
  </div>
</div>
</div>

The attention matrix goes from [B, H, 8K, 8K] to [B, H, 128K, 128K] — that's 256× more computation. This is why Flash Attention ([Ch 4](./04_attention_sdpa_and_mha.md)) and efficient KV-cache strategies ([Ch 17–18](./17_kv_cache_mechanics.md)) are essential.

## Domain Adaptation

Mid-training adapts a general model to a specific domain by continuing to train on domain-specific data while mixing in general data to prevent forgetting.

### Key Examples

<div class="diagram">
<div class="diagram-title">Domain-Adapted Models</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">CodeLLaMA</div>
    <div class="card-desc">LLaMA-2 + 500B code tokens. Infilling objective. Python specialization. 16K→100K context.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">DeepSeek-Coder</div>
    <div class="card-desc">Base model + 2T code tokens over 87 languages. Fill-in-middle training.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Llemma (Math)</div>
    <div class="card-desc">LLaMA-2 + 55B tokens from Proof-Pile-2 (math papers + code). Strong on MATH/GSM8K.</div>
  </div>
</div>
</div>

### Data Mixing to Avoid Catastrophic Forgetting

The critical challenge: training on domain data without forgetting general capabilities. The solution is **data mixing** — blending domain data with a fraction of general pre-training data:

```python
# Typical mid-training data mix
data_mix = {
    "domain_code": 0.60,        # 60% new domain data
    "general_web": 0.25,        # 25% general data (prevent forgetting)
    "high_quality_curated": 0.10, # 10% Wikipedia, books, math
    "instruction_adjacent": 0.05, # 5% natural QA-like patterns
}
```

| Mix Strategy | Domain Quality | General Retention | Risk |
|-------------|:-:|:-:|------|
| 100% domain | High | Low | Catastrophic forgetting |
| 80/20 domain/general | High | Medium | Mild degradation on general tasks |
| 60/40 domain/general | Good | Good | Best balance for most cases |
| Progressive (↑ domain over time) | High | Good | More complex to implement |

## Annealing

Near the end of mid-training (or the end of pre-training), **learning rate annealing** with high-quality data can significantly improve model capabilities:

<div class="diagram">
<div class="diagram-title">Annealing Phase</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Without Annealing</div>
    <ul>
      <li>Cosine decay continues to min_lr</li>
      <li>Final data mix unchanged</li>
      <li>Good but not optimal final quality</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">With Annealing</div>
    <ul>
      <li>Aggressive LR drop (→ 0) in final 5–10% of training</li>
      <li>Switch to high-quality data only</li>
      <li>Significant benchmark improvements (~1–3%)</li>
    </ul>
  </div>
</div>
</div>

LLaMA-3 used annealing on the final 40M tokens with a curated high-quality data mix, which notably improved benchmarks.

## Continued Pre-Training Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Load domain data (e.g., code)
dataset = load_dataset("bigcode/starcoderdata", split="train", streaming=True)

training_args = TrainingArguments(
    output_dir="./llama-code-mid-training",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,           # Lower than pre-training peak (3e-4)
    lr_scheduler_type="cosine",
    warmup_steps=500,
    max_steps=50000,
    bf16=True,
    gradient_checkpointing=True,  # Save memory at cost of ~30% speed
    logging_steps=10,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

Key differences from pre-training:
- **Lower learning rate** (typically 10–100× lower than pre-training peak)
- **Fewer steps** (thousands, not millions)
- **Curated data** (quality over quantity)
- **Often single-stage** (no complex parallelism — fits on fewer GPUs)

## Fill-in-the-Middle (FIM) Training

Code mid-training often adds a **fill-in-the-middle** objective alongside standard next-token prediction. This enables code infilling (autocomplete in the middle of a function):

```python
# Standard causal: predict left-to-right
# "def add(a, b):\n    return a + b"

# FIM transformation (50% of training samples):
# "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n<|fim_middle|>return a + b"

# The model learns to generate the middle given prefix + suffix
```

This is how Copilot-style code completion works — the model sees code before and after the cursor, and fills in the gap.

## What's Next

After mid-training produces a capable base model, **supervised fine-tuning** teaches it to follow instructions and behave as a helpful assistant.

[← Previous: Chapter 11 — Optimizers & Loss Functions](./11_optimizers_and_loss_functions.md) · **Next: [Chapter 13 — Fine-Tuning & Adaptation →](./13_finetuning_and_adaptation.md)**

---

*Last updated: April 2026*
