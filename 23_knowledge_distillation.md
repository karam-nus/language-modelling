---
title: "Chapter 23 — Knowledge Distillation & QAD"
description: "Teaching small models from big ones: soft targets and temperature, forward vs reverse KL, logit vs feature vs sequence-level distillation, on-policy distillation, pre-training distillation (Gemma, Llama 3.2) and prune-then-distill (Minitron), reasoning distillation (R1-Distill), and quantization-aware distillation with fake quantization and the straight-through estimator."
---

[← Back to Table of Contents](./README.md)

# Chapter 23 — Knowledge Distillation & QAD

Quantization (Chapters 20–22) keeps the architecture and shrinks the numbers. **Distillation** shrinks the architecture: a small *student* learns to reproduce a large *teacher*. Most small open models of 2025–26 — Gemma, Llama 3.2 1B/3B, Qwen3's small models, every "R1-Distill" — are distilled, and the same trick recovers accuracy lost to quantization.

> **The one-sentence version:** a teacher's full next-token distribution carries far more information per token than the one-hot "correct" token, so training a student to match it — on the student's own samples, ideally — transfers capability much faster than training from data alone.

## Why Distillation?

<div class="diagram">
<div class="diagram-title">Model Compression Landscape</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Quantization</div>
    <div class="card-desc">Same architecture, fewer bits: 8B BF16 → 8B 4-bit. Post-training or training-aware.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Pruning</div>
    <div class="card-desc">Same family, fewer weights: remove layers, heads, FFN channels or embedding dimensions — then heal with distillation.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Distillation</div>
    <div class="card-desc">Smaller model trained to imitate a larger one: logits, features or generated text.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">QAD (quantization-aware distillation)</div>
    <div class="card-desc">Fine-tune the quantized model to match its own full-precision version — the teacher guides recovery from quantization error.</div>
  </div>
</div>
</div>

## Knowledge Distillation — Fundamentals

<div class="img-caption">
  <img src="{{ '/assets/images/knowledge_distillation.svg' | relative_url }}" alt="Knowledge distillation showing teacher-student setup with soft labels, temperature scaling, and four distillation types">
  <figcaption>Knowledge distillation: a large teacher's soft probability distribution trains a small student more effectively than one-hot labels</figcaption>
</div>

### The Core Idea (Hinton et al., 2015)

A one-hot label says only which token occurred. The teacher's distribution also says which *other* tokens were plausible and by how much — "dark knowledge" about similarity:

```
hard label:      [0,     0,     0,    1,    0,    0,   …]   ← " cat" is correct
teacher output:  [0.001, 0.003, 0.002, 0.85, 0.09, 0.04, …]
                                       ↑cat  ↑kitten ↑dog
```

### Distillation Loss

1. **Hard-target loss** (ordinary next-token cross-entropy):
   $$\mathcal{L}_{\text{CE}} = -\sum_t \log p_S(x_t \mid x_{<t})$$

2. **Soft-target loss** (match the teacher's distribution at temperature τ):
   $$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \mathrm{KL}\!\left(p_T^{(\tau)} \,\|\, p_S^{(\tau)}\right)$$

3. **Combined**: $\mathcal{L} = (1 - \alpha)\,\mathcal{L}_{\text{CE}} + \alpha\,\mathcal{L}_{\text{KD}}$, with α typically 0.5–1.

τ > 1 softens both distributions so that small probabilities contribute; the τ² factor keeps gradient magnitudes comparable as τ changes. For LLMs, τ = 1 with α near 1 is common — the teacher's raw distribution is already informative.

| Tensor | Shape | Description |
|--------|-------|-------------|
| Teacher logits | `[B, T, V]` | Frozen teacher forward pass |
| Student logits | `[B, T, V]` | Trainable |
| Per-token KL | `[B, T]` | Summed over V |
| Loss | scalar | Mean over (non-masked) tokens |

With V = 128K–262K, storing teacher logits for an offline dataset is enormous — 8K tokens × 256K vocab × 2 bytes = 4 GB *per sequence*. Practical pipelines either run the teacher online, or store only the **top-k** logits per position (Gemma's pre-training distillation samples a few hundred).

### Forward or Reverse KL?

$\mathrm{KL}(p_T \| p_S)$ (forward) penalises the student for missing any mode the teacher covers — the student spreads probability over everything the teacher might say, which a small model cannot fit well. $\mathrm{KL}(p_S \| p_T)$ (reverse, used by MiniLLM) penalises the student for putting mass where the teacher does not — it becomes *mode-seeking*, more precise and less prone to generic, hallucinated continuations. Generalised and skewed mixtures (GKD) interpolate between them.

## Types of Distillation for LLMs

### Logit (Response) Distillation

Match output distributions token by token, as above. Used for **pre-training** small models: Gemma 2 and Gemma 3 were pre-trained against a larger teacher's distribution, and Llama 3.2 1B/3B were pruned from Llama 3.1 8B and then trained with logits from Llama 3.1 8B and 70B as targets. Distillation gives a small model a richer signal per token, which matters when data, not compute, is the constraint.

### Feature (Intermediate) Distillation

Also match hidden states (or attention maps) layer by layer:

$$\mathcal{L}_{\text{feat}} = \sum_l \lVert f_l^S - g(f_l^T) \rVert_2^2$$

with a learned projection g when widths differ. Classic for encoders (DistilBERT, TinyBERT); used for LLMs mostly when the student is a pruned copy of the teacher, so layers correspond naturally.

| Tensor | Shape | Notes |
|--------|-------|-------|
| Teacher hidden state (layer l) | `[B, T, d_T]` | e.g., `[4, 2048, 8192]` for 70B |
| Student hidden state (layer l) | `[B, T, d_S]` | e.g., `[4, 2048, 4096]` for 8B |
| Projection g | `[d_T, d_S]` | Learned |

### Prune, Then Distill (Minitron)

NVIDIA's **Minitron** recipe (2024) estimates the importance of layers, heads, FFN channels and embedding dimensions with a small calibration set, prunes the least important to reach a target size, then *heals* the pruned model with logit distillation from the original. Compressing Nemotron-4 15B to 8B and 4B this way needed up to 40× fewer training tokens than training those sizes from scratch; the recipe was then applied to Llama 3.1 8B and Mistral NeMo 12B.

### Sequence-Level Distillation (Synthetic Data)

In post-training, "distillation" usually means **training on teacher-generated text** with ordinary SFT:

<div class="diagram">
<div class="diagram-title">Sequence-Level Distillation Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Strong teacher (an open frontier model or your own large model)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Generate responses to many prompts (often several samples each)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Filter: verify answers, judge quality, deduplicate, decontaminate</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">SFT the student on (prompt, teacher response) pairs</div>
</div>
</div>

Milestones: **Alpaca** (2023, 52K instructions from text-davinci-003), **Orca** (2023, ~5M ChatGPT and 1M GPT-4 explanations), the **Phi** series (synthetic "textbooks"), and the reasoning distillations below. It discards the teacher's probabilities but needs only its outputs — which is also why many API terms of service restrict using outputs to train competing models.

### On-Policy Distillation

Off-policy distillation trains the student on the *teacher's* sequences; at inference the student conditions on its *own* earlier tokens, and its mistakes compound into states it never saw in training. **On-policy distillation** (GKD, 2023) samples sequences from the student and has the teacher score every token of them — dense, per-token feedback on exactly the states the student visits. It is "RL with a teacher instead of a reward", and far more sample-efficient than RL. Qwen3 trained its small models with off-policy then on-policy distillation from its flagships, and 2025 reports (e.g., Thinking Machines' *On-Policy Distillation*) found it reaches RL-level reasoning gains for a fraction of the compute.

## Distillation for Reasoning

The highest-profile application since 2025: distilling **long chains of thought**.

1. Generate long reasoning traces from a reasoning teacher.
2. Keep only traces with verified-correct answers (and filter for readability).
3. SFT the student on (problem, trace, answer).

DeepSeek-R1's team fine-tuned Qwen2.5 and Llama 3 bases on about 800K R1-generated samples (≈600K reasoning, 200K general). The resulting **R1-Distill** models were strong for their size — the 7B scored far above non-reasoning models many times larger on competition maths — and the paper reported that distilling from R1 beat running large-scale RL directly on the small model. Later work (s1, LIMO, OpenThoughts) showed carefully chosen sets of 1K–1M traces go a long way.

## Quantization-Aware Distillation (QAD)

PTQ accepts some quality loss. **QAT** simulates quantization during training so the weights adapt. **QAD** is QAT whose loss is distillation from the full-precision model — the best possible teacher for "behave like you used to":

<div class="diagram">
<div class="diagram-title">PTQ vs QAD</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Post-training quantization</div>
    <ul>
      <li>No training</li>
      <li>Small loss at 4-bit, larger at 2–3-bit</li>
      <li>Minutes to hours</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">QAD</div>
    <ul>
      <li>Short fine-tune with the original model as teacher</li>
      <li>Recovers most of the PTQ gap at 4-bit (incl. NVFP4)</li>
      <li>Makes 2–3-bit usable</li>
    </ul>
  </div>
</div>
</div>

Examples: Meta's quantized Llama 3.2 1B/3B used QAT with LoRA adapters (alongside SpinQuant); Google's Gemma 3 QAT checkpoints; NVIDIA's ModelOpt offers QAD for recovering NVFP4 accuracy; LLM-QAT (2023) used data generated by the model itself so no original training data was needed.

### QAD Training Loop

```python
import torch
import torch.nn.functional as F

def qad_step(student, teacher, batch, optimizer, tau=1.0, alpha=0.9):
    """student: model with fake-quantized linear layers; teacher: frozen full-precision copy."""
    ids, labels = batch["input_ids"], batch["labels"]                 # [B, T]
    with torch.no_grad():
        t_logits = teacher(ids).logits                                 # [B, T, V]
    s_logits = student(ids).logits                                     # [B, T, V]

    V = s_logits.size(-1)
    mask = (labels[:, 1:] != -100).reshape(-1)                         # predicted positions only
    s = (s_logits[:, :-1] / tau).reshape(-1, V)[mask]                  # [N, V]
    t = (t_logits[:, :-1] / tau).reshape(-1, V)[mask]
    kd = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),          # KL(teacher ‖ student) per token,
                  log_target=True, reduction="batchmean") * tau**2     # "batchmean" divides by N tokens here
    ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)

    loss = alpha * kd + (1 - alpha) * ce
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

Reshaping to `[N, V]` before `kl_div(..., reduction="batchmean")` makes the loss a per-*token* mean; applied to `[B, T, V]` directly, "batchmean" would divide by B only and scale the loss with sequence length.

### Fake Quantization and the Straight-Through Estimator

During QAT/QAD, weights are quantized and immediately dequantized in the forward pass, so the loss sees quantization error; the rounding is treated as identity in the backward pass:

```
Forward:  W_q = clamp(round(W / s), qmin, qmax) · s     ← discrete; zero gradient almost everywhere
Backward: ∂L/∂W ≈ ∂L/∂W_q  (inside the clamp range)     ← straight-through estimator (STE)
```

| Tensor | Shape | Precision |
|--------|-------|---------------------|
| Latent weights | `[d, k]` | BF16/FP32 (trainable) |
| Fake-quantized weights | `[d, k]` | values on the 4-bit grid |
| Gradients | `[d, k]` | Full precision via STE |
| Scales | `[d, k/g]` | Often trainable too (LSQ-style) |

After training, the latent weights are quantized for real and the latent copy is discarded.

## Comparison: Compression Routes

| Route | Size reduction | Training cost | Typical use |
|--------|:--------------:|:---------------:|:---------------:|
| FP8 PTQ | 2× | None | Datacenter default |
| 4-bit PTQ (GPTQ/AWQ/NVFP4) | ~3.5–4× | Calibration only | Most deployments |
| 4-bit QAT / QAD | ~3.5–4× | Short fine-tune | Release checkpoints, edge |
| 2–3-bit with QAD or codebooks | 5–8× | Fine-tune | Fitting giant models |
| Prune + distill (Minitron) | 2–4× in parameters | Fraction of pre-training | Model families from one parent |
| Distill into a new small model | 5–50× in parameters | A full (but cheaper) training run | Small and on-device models |
| Distill, then quantize | Multiplicative | Both | Phones and laptops |

## A Practical Pipeline: From a Large Model to a Deployable Small One

<div class="diagram">
<div class="diagram-title">End-to-End Compression Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Step 1: Pick a student: prune the teacher (Minitron) or start from a smaller base of the same family and tokenizer</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Step 2: Logit distillation on general data from the teacher (top-k logits if offline)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Step 3: Post-train: SFT on teacher-generated, verified responses; then on-policy distillation or RL</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Step 4: Quantize (PTQ); if quality drops, run QAD against the full-precision student</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Result: a small, low-precision model that keeps much of the teacher's skill on its target tasks</div>
</div>
</div>

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>On-policy distillation gives a dense per-token signal; RL gives one sparse reward per episode. If a capable teacher exists, distillation is far cheaper — so why do labs still run RL on their small models, and what can RL find that a teacher cannot teach?</p>
</div>

## Key Takeaways

- Soft targets carry similarity information one-hot labels discard; L = α·τ²·KL + (1 − α)·CE.
- Forward KL is mode-covering, reverse KL mode-seeking; on-policy distillation trains on the student's own samples and fixes exposure bias.
- Pre-training distillation (Gemma, Llama 3.2) and prune-then-distill (Minitron) are how small open models are made; sequence-level distillation (R1-Distill) is how small reasoning models are made.
- QAD = QAT with the full-precision model as teacher; fake quantization + STE lets weights adapt to the grid.
- Compute KD losses per token (reshape to [N, V]) and keep only top-k teacher logits when storing them offline.

## Check Yourself

<details class="check">
<summary>Why multiply the KD loss by τ²?</summary>
<p>Softening logits by τ shrinks the gradients of the soft cross-entropy by roughly 1/τ². Multiplying by τ² keeps the soft-target gradient on the same scale as the hard-target gradient, so α keeps the same meaning as τ changes.</p>
</details>

<details class="check">
<summary>How much storage would full teacher logits need for 1B training tokens with a 152K vocabulary in BF16? With top-64?</summary>
<p>Full: 10⁹ × 152,000 × 2 B ≈ 304 TB. Top-64 (values in BF16 plus 32-bit indices): 10⁹ × 64 × 6 B ≈ 384 GB.</p>
</details>

<details class="check">
<summary>A student distilled with forward KL produces bland, hedging text. What would you try?</summary>
<p>Reverse KL or a skewed/generalised divergence (mode-seeking), and on-policy distillation so the student is corrected on its own generations; also a lower teacher temperature.</p>
</details>

<details class="check">
<summary>Why is on-policy distillation more sample-efficient than RL for the same student?</summary>
<p>RL gives one scalar reward per episode, so credit must be inferred across thousands of tokens. On-policy distillation gives a full teacher distribution — a target — at every token of the student's own sample, far more bits of supervision per rollout.</p>
</details>

<details class="check">
<summary>What does the straight-through estimator approximate, and when does it break down?</summary>
<p>It approximates the derivative of the rounding function as 1, passing gradients to the latent weights as if quantization were identity. It breaks down at very low bit-widths, where the true function is far from identity and gradients mislead — one reason 1–2-bit methods need special training recipes.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Distilling the Knowledge in a Neural Network | Hinton et al., 2015 | [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531) | Soft targets, temperature |
| Sequence-Level Knowledge Distillation | Kim & Rush, 2016 | [arxiv.org/abs/1606.07947](https://arxiv.org/abs/1606.07947) | Training on teacher outputs |
| DistilBERT | Sanh et al., 2019 | [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108) | Feature + logit distillation |
| MiniLLM: Knowledge Distillation of LLMs | Gu et al., 2023 | [arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543) | Reverse KL |
| On-Policy Distillation of Language Models (GKD) | Agarwal et al., 2023 | [arxiv.org/abs/2306.13649](https://arxiv.org/abs/2306.13649) | Student-generated sequences |
| Compact Language Models via Pruning and Knowledge Distillation (Minitron) | Muralidharan et al. (NVIDIA), 2024 | [arxiv.org/abs/2407.14679](https://arxiv.org/abs/2407.14679) | Prune then distill |
| Gemma 2 | Gemma Team, 2024 | [arxiv.org/abs/2408.00118](https://arxiv.org/abs/2408.00118) | Pre-training distillation |
| LLM-QAT: Data-Free Quantization Aware Training | Liu et al., 2023 | [arxiv.org/abs/2305.17888](https://arxiv.org/abs/2305.17888) | QAT with self-generated data |
| DeepSeek-R1 | DeepSeek-AI, 2025 | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) | Reasoning distillation |
| Qwen3 Technical Report | Qwen Team, 2025 | [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) | Strong-to-weak, on-policy distillation |
| On-Policy Distillation | Thinking Machines, 2025 | [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | Cost vs RL |
| Introducing quantized Llama models | Meta AI, 2024 | [ai.meta.com/blog/meta-llama-quantized-lightweight-models](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/) | QAT + LoRA, SpinQuant |

---

**Next:** [Chapter 24 — GPU Architecture for ML](./24_gpu_architecture.md) — the hardware every one of these bytes and FLOPs runs on.

[← Back to Table of Contents](./README.md)
