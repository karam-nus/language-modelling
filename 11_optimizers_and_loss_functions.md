---
title: "Chapter 11 — Optimizers & Loss Functions for LLMs/VLMs"
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — Optimizers & Loss Functions for LLMs/VLMs

> *"The loss function defines what you want; the optimizer determines how you get there. Both choices profoundly shape model behaviour."*

## Loss Functions

### Cross-Entropy Loss — The Foundation

Every LLM is trained to minimise **cross-entropy** between the predicted token distribution and the true next token. This is also called **negative log-likelihood (NLL)**:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t})$$

where `N` is the number of labelled tokens (non-masked positions), and `p_θ` is the model's predicted probability for the true token.

**In code**:

```python
import torch.nn.functional as F

# logits: [B, T, V]  — model output (unnormalised)
# labels: [B, T]    — target token IDs (-100 = ignore)
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # [B*T, V]
    labels.view(-1),               # [B*T]
    ignore_index=-100,
    reduction="mean",
)
```

**Tensor shapes**:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `logits` | `[B, T, V]` | Unnormalised scores over vocabulary |
| `labels` | `[B, T]` | True token IDs; -100 for masked positions |
| `probs` | `[B, T, V]` | After softmax (not needed for loss) |
| `loss` | scalar | Mean NLL over non-masked positions |

### Perplexity

Perplexity is the exponentiated cross-entropy loss — a human-interpretable measure of how "surprised" the model is:

$$\text{PPL} = e^{\mathcal{L}_{\text{CE}}} = e^{-\frac{1}{N}\sum \log p(x_t \mid x_{<t})}$$

Lower perplexity = better model. Typical values: GPT-2 ~30, Llama-3-8B ~7–9 on Wikitext-103.

### Z-Loss (Auxiliary Stability Loss)

A regularisation term added during training to prevent the softmax logits from growing very large (logit explosion), which causes instability:

$$\mathcal{L}_z = \alpha \cdot \frac{1}{B \cdot T} \sum_{b,t} \left(\log \sum_v e^{z_{b,t,v}}\right)^2$$

where `z` are the pre-softmax logits and `α` is a small coefficient (e.g., 1e-4). Used in PaLM, Gemini, and others.

### Load-Balancing Loss (MoE)

For Mixture-of-Experts models, an auxiliary loss ensures experts are used roughly equally (preventing expert collapse):

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N_E \sum_{e=1}^{N_E} f_e \cdot P_e$$

where `f_e` is the fraction of tokens routed to expert `e`, and `P_e` is the average routing probability to `e`. This is added to the main cross-entropy loss. See Chapter 33 for full MoE coverage.

### Reward Model Loss

Reward models (used in RLHF) are trained with a **ranking loss** — the chosen response should score higher than the rejected one:

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

where `y_w` is the winning (chosen) response, `y_l` is the losing (rejected) response, and `r_θ` is the scalar reward. This is the **Bradley-Terry preference model**.

### DPO Loss

Direct Preference Optimisation (DPO) reformulates preference learning as a supervised loss:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

No reward model is needed — the reference policy `π_ref` (frozen SFT model) implicitly regularises the policy. `β` controls how far the policy strays from the reference.

### VLM-Specific Losses

Vision-Language Models combine image and text objectives:

| Loss Component | Objective | When Used |
|----------------|-----------|-----------|
| Captioning loss (NTP) | Predict text tokens conditioned on image | Pre-training + SFT |
| Contrastive loss (CLIP-style) | Align image and text embeddings | VLM pre-training |
| Image-text matching | Binary classification: does this text match this image? | Pre-training auxiliary |
| Object detection loss | Bounding box regression + classification | Grounding VLMs |

---

## Optimizers

### SGD with Momentum

The classical optimizer — rarely used for LLM training in practice:

$$v_t = \beta v_{t-1} + g_t \qquad \theta_t = \theta_{t-1} - \eta \cdot v_t$$

**Memory**: 1 momentum buffer per parameter → `1× params` extra.

### Adam — Adaptive Moment Estimation

Adam (Kingma & Ba, 2014) is the foundation of most LLM optimisers. It maintains running estimates of the first and second moments of gradients:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment — mean)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment — variance)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias-corrected)}$$
$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters**: `β₁=0.9`, `β₂=0.95` (LLMs; note: 0.999 is the original paper default but 0.95 is standard for LLMs), `ε=1e-8`, `η=1e-4 to 3e-4`.

**Memory cost**: 2 optimizer states (`m` and `v`) per parameter, both in FP32:

| Model Size | Weights (BF16) | Adam States (FP32) | Gradients (FP32) | Total |
|:----------:|:--------------:|:------------------:|:----------------:|:-----:|
| 7B | 14 GB | 56 GB | 28 GB | ~98 GB |
| 70B | 140 GB | 560 GB | 280 GB | ~980 GB |

This is why distributed training (FSDP/ZeRO) is essential — see Chapter 26.

### AdamW — Adam with Weight Decay

The standard choice for LLM training. AdamW (Loshchilov & Hutter, 2017) decouples weight decay from the gradient update:

**Adam** (incorrect weight decay): `θ_t = θ_{t-1} - η (m̂/√v̂ + λθ_{t-1})`  
**AdamW** (correct weight decay): `θ_t = (1 - ηλ)θ_{t-1} - η · m̂/√v̂`

The difference: in AdamW, weight decay is applied directly to weights, not scaled by the adaptive learning rate. This gives more consistent regularisation.

**Typical settings for LLM training**:
- `lr = 1e-4` to `3e-4` (pre-training); `lr = 2e-5` to `2e-4` (fine-tuning)
- `weight_decay = 0.1`
- `β₁ = 0.9`, `β₂ = 0.95`
- `gradient_clip = 1.0`

### Adam-mini

A recent (2024) memory-efficient variant that reduces optimizer memory by using one learning rate per parameter group (e.g., per attention head) instead of per parameter:

- **Memory**: ~45% of Adam memory
- **Performance**: matches Adam/AdamW in practice
- Particularly useful for large models where optimizer states dominate memory

### Adafactor

Designed for extreme memory efficiency — used to train early T5 models:

- Does **not** store full second moment vector; factorises it into rank-1 row/column factors
- **Memory**: ~O(√params) instead of O(params) for second moment
- Trade-off: can be less stable; often needs careful tuning
- Used in: T5, mT5, Switch Transformer

### SOAP

**SOAP** (Shampoo with Adam in the Preconditioner, 2024) applies second-order information via matrix preconditioning:

- Precomputes Kronecker-factored curvature (similar to K-FAC/Shampoo)
- Memory overhead: similar to Adam
- Speed: can reach the same loss in ~40% fewer steps than AdamW
- Increasingly used for smaller training runs where step count is the bottleneck

### Muon (Momentum + Orthogonalisation)

A newer optimiser (2024) that applies Nesterov momentum and then orthogonalises the gradient update:

- Applies to weight matrices only (not embeddings/biases)
- Reaches lower loss than AdamW at same compute in recent experiments
- Used in MicroGrad and some frontier training runs (reportedly used for DeepSeek V3's internal layers)

---

## Learning Rate Schedules

The learning rate schedule is as important as the optimiser choice for LLM training.

### Cosine Decay with Warmup

The de facto standard for pre-training:

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi(t - T_{\text{warmup}})}{T - T_{\text{warmup}}}\right) & t > T_{\text{warmup}} \end{cases}$$

<div class="diagram">
<div class="diagram-title">Learning Rate Schedule — Cosine with Warmup</div>
<div class="flow-h">
  <div class="flow-node accent">Warmup<br><small>~1000–2000 steps</small><br><small>linear ramp</small></div>
  <div class="flow-node purple">Cosine Decay<br><small>majority of training</small><br><small>smooth decrease</small></div>
  <div class="flow-node green">Final LR<br><small>η_min ≈ 0.1 × η_max</small><br><small>prevents overshoot</small></div>
</div>
</div>

**Typical warmup**: 1000–2000 steps. Starting cold (no warmup) causes early instability.

### WSD — Warmup, Stable, Decay

Used in MiniCPM and other recent models. Enables **continual training** by decoupling the decay phase:

1. **Warmup**: linear ramp from 0 to `η_max`
2. **Stable**: constant `η_max` for the bulk of training
3. **Decay**: cosine or linear decay to `η_min`

Key advantage: you can extend training (add more data) by extending the stable phase without restarting the decay. This enables the "annealing" trick — saving a checkpoint, then annealing on high-quality data.

### Trapezoidal / Linear Decay

Used in models like OLMo 2. Simpler than cosine, easier to reason about:
- Warmup → flat → linear decay to 0

### Constant LR (Fine-Tuning)

For SFT/LoRA fine-tuning over few epochs, a constant LR with short warmup often works well:
- `lr = 2e-4` for LoRA, `lr = 2e-5` for full fine-tune
- 5–10% of total steps for warmup

---

## Gradient Clipping

A near-universal component of LLM training. Clips the global gradient norm to prevent instability from gradient spikes:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{c}{\|\mathbf{g}\|_2}\right)$$

**Standard value**: `max_norm = 1.0`. Training loss spikes often coincide with large gradient norm events; clipping at 1.0 prevents most catastrophic updates.

```python
# In PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Optimizer Comparison Summary

| Optimizer | Memory (relative) | Speed | LLM Use | Notes |
|-----------|:-----------------:|:-----:|:-------:|-------|
| **AdamW** | 3× params | ★★★★ | Universal | Standard; most widely used |
| **Adam-mini** | ~1.5× params | ★★★★ | Growing | Drop-in replacement |
| **Adafactor** | ~1.1× params | ★★★ | T5-era | Memory-efficient; less stable |
| **SOAP** | ~3× params | ★★★★★ | Research | Faster convergence, same memory |
| **Muon** | ~2× params | ★★★★★ | Frontier | Orthogonalised momentum |
| **SGD+momentum** | 2× params | ★★★ | Rarely | No adaptivity; hard to tune |

---

## Hyperparameter Sensitivity

<div class="diagram">
<div class="diagram-title">Key Training Hyperparameters</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Learning Rate</div>
    <div class="card-desc">Most sensitive. Too high → divergence. Too low → slow convergence. Typical: 1e-4 to 3e-4 for pre-training.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Batch Size</div>
    <div class="card-desc">Linear scaling rule: double batch → double LR. Token-level batch is more meaningful (B × T). Typical: 4M–16M tokens/step.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Weight Decay</div>
    <div class="card-desc">Regularises weights. Too high → underfitting. Standard: 0.1 for pre-training, 0.01–0.1 for fine-tuning.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">β₂ in Adam</div>
    <div class="card-desc">0.95 (LLMs) vs 0.999 (original). Lower β₂ = faster adaptation to gradient changes. Important for stability.</div>
  </div>
</div>
</div>

## What's Next

With the training machinery understood — data, objectives, and optimisers — we'll look at **mid-training** strategies that extend and refine pre-trained models.

[← Previous: Chapter 10 — Pre-Training at Scale](./10_pretraining_at_scale.md) · **Next: [Chapter 12 — Mid-Training & Continued Pre-Training →](./12_mid_training.md)**

---

*Last updated: April 2026*
