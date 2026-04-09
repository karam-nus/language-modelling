---
title: "Chapter 11 — Fine-Tuning & Adaptation"
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — Fine-Tuning & Adaptation

> *"Fine-tuning is where a model goes from 'knows a lot' to 'does what you want' — the difference between a base model and an assistant."*

## Supervised Fine-Tuning (SFT)

SFT trains a base model on instruction-response pairs, teaching it to follow instructions and generate helpful outputs. The training objective is the same cross-entropy loss, but computed **only on the response tokens** (the instruction tokens are masked from the loss):

<div class="diagram">
<div class="diagram-title">SFT Loss Masking</div>
<div class="flow-h">
  <div class="flow-node purple">System: You are a helpful assistant</div>
  <div class="flow-node orange">User: What is attention?</div>
  <div class="flow-node green">Assistant: Attention is a mechanism that...</div>
</div>
<div style="text-align: center; padding: 0.5rem 0; color: var(--text-muted);">↓ Loss computation ↓</div>
<div class="flow-h">
  <div class="flow-node purple" style="opacity: 0.3;">masked (no loss)</div>
  <div class="flow-node orange" style="opacity: 0.3;">masked (no loss)</div>
  <div class="flow-node green">loss computed here ✓</div>
</div>
</div>

### SFT Datasets

| Dataset | Size | Source | Format |
|---------|:----:|--------|--------|
| **OpenHermes 2.5** | 1M | GPT-4 generated | Multi-turn chat |
| **UltraChat 200K** | 200K | GPT-3.5/4 generated | Filtered conversations |
| **Alpaca** | 52K | GPT-3.5 generated | Single-turn instruction |
| **FLAN Collection** | 1.8M | Human + model | Diverse NLP tasks |
| **SlimOrca** | 517K | GPT-4 curated | Deduped multi-task |

## LoRA — Low-Rank Adaptation

Full fine-tuning updates all parameters — expensive for large models (7B+ parameters). **LoRA** (Hu et al., 2021) decomposes weight updates into low-rank matrices, training only ~0.1–1% of parameters.

### The Core Idea

Instead of updating the full weight matrix W, LoRA adds a low-rank update:

$$W' = W + \Delta W = W + \alpha \cdot B A$$

where $W \in \mathbb{R}^{d \times k}$, $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$.

<div class="diagram">
<div class="diagram-title">LoRA — Low-Rank Decomposition</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Full Fine-Tuning</div>
    <ul>
      <li>Update W directly: [d, k]</li>
      <li>For W_q in LLaMA-3-8B:</li>
      <li>4096 × 4096 = 16.8M params per layer</li>
      <li>Memory: optimizer states for all params</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">LoRA (r=16)</div>
    <ul>
      <li>Freeze W, train A [r, k] + B [d, r]</li>
      <li>A: [16, 4096] = 65K params</li>
      <li>B: [4096, 16] = 65K params</li>
      <li>130K vs 16.8M → 129× fewer params</li>
    </ul>
  </div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">LoRA Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Input x</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">W·x (frozen)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">original path</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">A·x (down-project)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">r</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">r ≪ d (e.g., 16)</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">B·(A·x) (up-project)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">LoRA delta</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Output (W + BA)·x</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">frozen + learned combined</span>
  </div>
</div>
</div>

### LoRA Rank Selection

| Rank (r) | Trainable Params (8B model) | Quality | Use Case |
|:---:|:---:|:---:|------|
| 8 | ~6M | Good for simple tasks | Classification, style |
| 16 | ~13M | Good general purpose | Instruction following |
| 32 | ~26M | Better for complex tasks | Code, reasoning |
| 64 | ~52M | Near full fine-tune | Domain adaptation |
| 128 | ~104M | Diminishing returns | Rarely needed |

## QLoRA — Quantized LoRA

**QLoRA** (Dettmers et al., 2023) combines 4-bit quantized base weights with LoRA adapters, enabling fine-tuning of large models on consumer GPUs:

<div class="diagram">
<div class="diagram-title">QLoRA Memory Savings</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Full Fine-Tuning (LLaMA-3-8B)</div>
    <div class="card-desc">BF16 weights: 16 GB<br>Adam states: 32 GB<br>Gradients: 16 GB<br><strong>Total: ~64 GB → needs A100</strong></div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">QLoRA (LLaMA-3-8B)</div>
    <div class="card-desc">NF4 weights: 4 GB<br>LoRA adapters (BF16): ~26 MB<br>Adam for LoRA: ~52 MB<br><strong>Total: ~5 GB → fits on RTX 4090</strong></div>
  </div>
</div>
</div>

Key innovations: **NF4** (Normal Float 4-bit quantization, optimal for normally distributed weights), **double quantization** (quantize the quantization constants too), and **paged optimizers** (offload optimizer states to CPU).

## DoRA — Weight-Decomposed LoRA

**DoRA** (Liu et al., 2024) decomposes weights into magnitude and direction components, applying LoRA only to the direction:

$$W' = m \cdot \frac{W + BA}{\|W + BA\|}$$

This consistently outperforms LoRA at the same rank, closing the gap to full fine-tuning.

## Other PEFT Methods

<div class="diagram">
<div class="diagram-title">Parameter-Efficient Fine-Tuning Landscape</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Adapters</div>
    <div class="card-desc">Small bottleneck layers inserted after attention/FFN. ~2–5% trainable params. Adds latency.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Prefix Tuning</div>
    <div class="card-desc">Prepend learnable "soft prompts" to keys/values at each layer. No weight changes to model.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Prompt Tuning</div>
    <div class="card-desc">Prepend learnable embeddings to input. Simplest PEFT — only ~100K params. Works at scale.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">IA³</div>
    <div class="card-desc">Learn scaling vectors for K, V, and FFN. Even fewer params than LoRA. Good for few-shot.</div>
  </div>
</div>
</div>

## Full Fine-Tuning vs PEFT

| Factor | Full Fine-Tuning | LoRA/QLoRA |
|--------|:---:|:---:|
| Trainable parameters | 100% | 0.1–1% |
| Memory (8B model) | 64+ GB | 5–16 GB |
| Training speed | Baseline | ~1.2–1.5× faster |
| Quality (general) | Best | ~95–99% of full |
| Quality (complex domain) | Best | May need higher rank |
| Multi-task | One model each | Stack/swap adapters |
| Merge into base? | N/A | Yes (W + BA) |

## Fine-Tuning with TRL

[TRL](https://github.com/huggingface/trl) is the standard library for LLM fine-tuning:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,               # scaling factor: alpha/r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Training
training_args = SFTConfig(
    output_dir="./llama-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_seq_length=2048,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)
trainer.train()

# Merge LoRA weights back into the base model
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./llama-sft-merged")
```

## What's Next

SFT makes models follow instructions, but it doesn't make them **safe**, **truthful**, or **aligned** with human preferences. The next chapter covers **RLHF, DPO**, and other alignment techniques.

[← Previous: Chapter 10 — Mid-Training](./10_mid_training.md) · **Next: [Chapter 12 — Alignment: RLHF & Beyond →](./12_alignment_rlhf_and_beyond.md)**

---

*Last updated: April 2026*
