---
title: "Chapter 15 — Alignment: RLHF & Beyond"
---

[← Back to Table of Contents](./README.md)

# Chapter 15 — Alignment: RLHF & Beyond

> *"Alignment is the difference between a model that can answer any question and one that should — it's about making AI helpful, harmless, and honest."*

## Why Alignment Matters

An SFT model follows instructions, but it may also: generate toxic content, make up facts confidently, comply with harmful requests, or be sycophantic. **Alignment** tunes the model's behavior to match human preferences.

<div class="diagram">
<div class="diagram-title">The Alignment Problem</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Before Alignment (SFT Only)</div>
    <ul>
      <li>Follows instructions but may be harmful</li>
      <li>Confidently wrong (hallucinations)</li>
      <li>No refusal behavior for dangerous requests</li>
      <li>May be verbose or sycophantic</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">After Alignment</div>
    <ul>
      <li>Refuses harmful requests appropriately</li>
      <li>Expresses uncertainty when unsure</li>
      <li>Balanced helpfulness vs safety</li>
      <li>Concise, honest responses</li>
    </ul>
  </div>
</div>
</div>

## The RLHF Pipeline

**Reinforcement Learning from Human Feedback** is the classic alignment approach, used in ChatGPT and Claude.

<div class="diagram">
<div class="diagram-title">RLHF — Three-Stage Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Stage 1: Supervised Fine-Tuning (SFT) <small>Train on instruction-response pairs → SFT model</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Stage 2: Reward Model Training <small>Collect human preferences (A > B) → train reward model R(x, y)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Stage 3: RL Optimization (PPO) <small>Optimize SFT model to maximize R(x, y) with KL penalty</small></div>
</div>
</div>

### Stage 2: Reward Model

Human annotators compare pairs of model responses and pick the better one. These preferences train a reward model that predicts a scalar quality score:

$$\mathcal{L}_{\text{RM}} = -\log\sigma(R(x, y_w) - R(x, y_l))$$

where $y_w$ is the preferred response and $y_l$ is the rejected response.

The reward model is typically initialized from the SFT model with the LM head replaced by a scalar head.

### Stage 3: PPO Optimization

PPO (Proximal Policy Optimization) updates the policy (SFT model) to maximize reward while staying close to the original SFT model via a KL divergence penalty:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[R(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}})\right]$$

The KL penalty prevents **reward hacking** — where the model exploits the reward model's weaknesses rather than genuinely improving.

<div class="diagram">
<div class="diagram-title">RLHF Components</div>
<div class="layer-stack">
  <div class="layer accent">Policy Model (π_θ): The model being optimized — generates responses</div>
  <div class="layer purple">Reference Model (π_SFT): Frozen SFT model — used for KL penalty</div>
  <div class="layer green">Reward Model (R): Predicts human preference scores</div>
  <div class="layer orange">Value Model (V): Estimates expected future reward (PPO critic)</div>
</div>
</div>

RLHF requires **4 models in memory** simultaneously (policy, reference, reward, value) — extremely expensive. This motivated simpler alternatives.

## DPO — Direct Preference Optimization

**DPO** (Rafailov et al., 2023) eliminates the separate reward model and RL phase entirely. It directly optimizes the policy from preference pairs:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)$$

The key insight: the optimal RLHF policy has an analytical form that depends only on the log-probability ratio of the policy and reference model. No need for a separate reward model — the policy **is** the reward model.

<div class="diagram">
<div class="diagram-title">DPO vs RLHF</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">RLHF (PPO)</div>
    <ul>
      <li>4 models in memory</li>
      <li>Online sampling required</li>
      <li>Reward model training + PPO</li>
      <li>Hyperparameter sensitive</li>
      <li>Highest ceiling (with enough tuning)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">DPO</div>
    <ul>
      <li>2 models (policy + frozen reference)</li>
      <li>Offline — uses static preference dataset</li>
      <li>Single training stage</li>
      <li>Much simpler to implement</li>
      <li>Near-PPO quality for most use cases</li>
    </ul>
  </div>
</div>
</div>

### DPO Training with TRL

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "your-sft-model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
ref_model = AutoModelForCausalLM.from_pretrained(
    "your-sft-model",  # Same starting point — frozen
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# Preference dataset: each example has prompt, chosen, rejected
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

training_args = DPOConfig(
    output_dir="./dpo-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,           # Very low LR for DPO
    beta=0.1,                     # KL penalty strength
    num_train_epochs=1,
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Other Alignment Methods

<div class="diagram">
<div class="diagram-title">Alignment Method Landscape</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">KTO (Kahneman-Tversky)</div>
    <div class="card-desc">Works with binary feedback (good/bad) instead of pairs. Based on prospect theory. Easier data collection.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">ORPO (Odds Ratio)</div>
    <div class="card-desc">Combines SFT and alignment in one stage. No reference model needed. Simple loss function.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">RLAIF</div>
    <div class="card-desc">RL from AI Feedback — use a strong LLM (GPT-4, Claude) to generate preferences instead of humans.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Constitutional AI (CAI)</div>
    <div class="card-desc">Model self-critiques using a "constitution" of principles. Self-improvement loop. Used by Anthropic.</div>
  </div>
</div>
</div>

### Comparison Table

| Method | Models Needed | Data Required | Complexity | Quality |
|--------|:---:|:---:|:---:|:---:|
| **PPO (RLHF)** | 4 | Preferences + online | Very high | Highest ceiling |
| **DPO** | 2 | Preferences (offline) | Low | Near-PPO |
| **KTO** | 2 | Binary thumbs-up/down | Low | Good |
| **ORPO** | 1 | Chosen + rejected | Very low | Good |
| **RLAIF** | 2 + judge LLM | AI-generated preferences | Medium | Good (cheaper) |
| **SimPO** | 1 | Preferences | Very low | Near-DPO |

## The Alignment Tax

Alignment improves safety but can slightly reduce raw capability (the "alignment tax"):

| Benchmark | Base Model | SFT | + DPO |
|-----------|:---:|:---:|:---:|
| MMLU | Baseline | +2–3% | +0–1% |
| GSM8K | Baseline | +5–10% | -0–2% |
| HumanEval | Baseline | +5–8% | -0–1% |
| TruthfulQA | Baseline | +10% | +15–20% |
| Toxicity ↓ | High | Medium | Low ✓ |

The alignment tax is real but small — and the safety and usability gains far outweigh it.

## What's Next

With training complete (pre-training → mid-training → SFT → alignment), we now move to **inference** — how models generate text token by token, and the sampling strategies that control output quality.

[← Previous: Chapter 14 — PEFT: LoRA, QLoRA & Variants](./14_peft_lora_and_variants.md) · **Next: [Chapter 16 — Inference & Sampling →](./16_inference_and_sampling.md)**

---

*Last updated: April 2026*
