---
title: "Chapter 11 — Optimizers & Loss Functions for LLMs/VLMs"
description: "Cross-entropy, perplexity and bits-per-byte, z-loss, MoE balancing, distillation, reward-model, DPO-family and policy-gradient losses; AdamW and its memory bill, Adafactor, Adam-mini, SOAP and Muon — including what Muon actually computes and why it spread in 2025–26."
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — Optimizers & Loss Functions for LLMs/VLMs

> **The one-sentence version:** the loss decides *what* a model is pushed towards — next tokens, a teacher's distribution, a preference, a reward — and the optimizer decides *how* each gradient becomes a weight update; in 2025–26 the second question got interesting again.

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

Lower perplexity = better model — **but only between models with the same tokenizer**. Perplexity is per *token*, and a model with a bigger vocabulary predicts fewer, harder tokens for the same text, so its per-token perplexity is not comparable with a smaller-vocabulary model's. For cross-model comparisons use **bits per byte** (BPB): total negative log-likelihood in bits divided by the number of UTF-8 bytes of the text, which is tokenizer-independent.

$$\text{BPB} = \frac{\sum_t -\log_2 p(x_t \mid x_{<t})}{\text{number of bytes}}$$

### Z-Loss (Auxiliary Stability Loss)

A regularisation term added during training to prevent the softmax logits from growing very large (logit explosion), which causes instability:

$$\mathcal{L}_z = \alpha \cdot \frac{1}{B \cdot T} \sum_{b,t} \left(\log \sum_v e^{z_{b,t,v}}\right)^2$$

where `z` are the pre-softmax logits and `α` is a small coefficient (e.g., 1e-4). It pushes the softmax normaliser log Z towards 0 so logits cannot drift upwards together. Used in PaLM, Chameleon, OLMo 2 and others.

### Load-Balancing Loss (MoE)

For Mixture-of-Experts models, an auxiliary loss ensures experts are used roughly equally (preventing expert collapse):

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N_E \sum_{e=1}^{N_E} f_e \cdot P_e$$

where `f_e` is the fraction of tokens routed to expert `e`, and `P_e` is the average routing probability to `e` (Switch Transformer). DeepSeek-V3 replaced most of it with an **auxiliary-loss-free** scheme — a per-expert bias added to routing scores and nudged up or down to even out load — keeping only a tiny sequence-level balance loss, because a strong auxiliary loss fights the language-modelling objective. See [Chapter 33](./33_mixture_of_experts.md).

### Distillation Loss

To train a student on a teacher's full distribution rather than one-hot labels, minimise the KL divergence between temperature-softened distributions:

$$\mathcal{L}_{\text{KD}} = \tau^2 \cdot \mathrm{KL}\!\left(\mathrm{softmax}(z^{T}/\tau)\,\big\|\,\mathrm{softmax}(z^{S}/\tau)\right)$$

usually mixed with ordinary cross-entropy. The τ² keeps gradient magnitudes comparable as τ changes. Gemma 2/3 and Llama 3.2 1B/3B were pre-trained this way; [Chapter 23](./23_knowledge_distillation.md) covers forward vs reverse KL and on-policy distillation.

### Reward Model Loss

Reward models (used in RLHF) are trained with a **ranking loss** — the chosen response should score higher than the rejected one:

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

where `y_w` is the winning (chosen) response, `y_l` is the losing (rejected) response, and `r_θ` is the scalar reward. This is the **Bradley-Terry preference model**.

### DPO Loss

Direct Preference Optimisation (DPO) reformulates preference learning as a supervised loss:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

No reward model is needed — the reference policy `π_ref` (frozen SFT model) implicitly regularises the policy. `β` controls how far the policy strays from the reference. Variants change the margin or the reference: **IPO** (squared loss, less overfitting), **KTO** (unpaired good/bad labels), **ORPO** and **SimPO** (no reference model; SimPO uses length-normalised log-probs).

### Policy-Gradient Losses (PPO, GRPO)

RL fine-tuning maximises expected reward. PPO and GRPO both use the **clipped surrogate** objective on each generated token:

$$\mathcal{L}_{\text{clip}} = -\,\mathbb{E}_t\left[\min\!\big(\rho_t A_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\right], \qquad \rho_t = \frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{old}}(y_t \mid \cdot)}$$

They differ in the advantage $A_t$: PPO learns a value network; GRPO uses the group-normalised reward of several samples for the same prompt, $A = (r - \bar r)/\sigma_r$, and needs no critic. [Chapter 15](./15_alignment_rlhf_and_beyond.md) derives them and covers the 2025 refinements (DAPO, Dr. GRPO, GSPO).

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

**Memory cost**: with standard mixed precision, every parameter carries **16 bytes** of state before any activations — BF16 weights (2) and gradients (2), plus an FP32 master copy (4) and FP32 `m` and `v` (4 + 4):

| Model Size | BF16 weights | BF16 grads | FP32 master + m + v | Total (16 B/param) |
|:----------:|:------------:|:----------:|:-------------------:|:-----:|
| 7B | 14 GB | 14 GB | 84 GB | **112 GB** |
| 70B | 140 GB | 140 GB | 840 GB | **1.12 TB** |

So a 7B model cannot be fully trained with AdamW on a single 80 GB GPU, and a 70B model needs its state sharded across at least 14 such GPUs before a single activation is stored. This is why ZeRO/FSDP sharding exists ([Chapter 26](./26_distributed_training.md)).

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

- **Memory**: cuts optimizer-state memory by roughly 45–50% (one `v` value per block instead of per parameter)
- **Performance**: matches AdamW in the paper's experiments up to 13B
- Particularly useful for large models where optimizer states dominate memory

### Adafactor

Designed for extreme memory efficiency — used to train early T5 models:

- Does **not** store full second moment vector; factorises it into rank-1 row/column factors
- **Memory**: ~O(√params) instead of O(params) for second moment
- Trade-off: can be less stable; often needs careful tuning
- Used in: T5, mT5, Switch Transformer

### SOAP

**SOAP** (Shampoo with Adam in the Preconditioner, 2024) applies second-order information via matrix preconditioning:

- Keeps Shampoo's Kronecker-factored preconditioners (a left and right matrix per layer) and runs Adam in their eigenbasis
- Memory overhead: more than Adam (the preconditioner matrices and their eigenvectors)
- Speed: the paper reports ~40% fewer iterations than AdamW at 360M–660M scale
- Mostly a research optimizer; overhead and complexity have limited large-scale adoption

### Muon (MomentUm Orthogonalized by Newton–Schulz)

Muon (Keller Jordan et al., 2024, first proven in the modded-nanoGPT speed-runs) treats each 2D weight matrix as a matrix rather than a bag of numbers. It keeps an ordinary momentum buffer, then replaces the update with the nearest **semi-orthogonal** matrix — same singular vectors, all singular values pushed towards 1 — computed with a few Newton–Schulz iterations:

```python
import numpy as np

def newton_schulz(G, steps=5, eps=1e-7):
    """Approximately orthogonalise G (push all singular values towards 1)."""
    a, b, c = 3.4445, -4.7750, 2.0315        # quintic coefficients from Keller Jordan's Muon
    X = G / (np.linalg.norm(G) + eps)         # Frobenius norm ≥ spectral norm, so σ_max ≤ 1
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X

rng = np.random.default_rng(0)
G = rng.standard_normal((256, 512)) @ np.diag(np.logspace(0, -3, 512))   # ill-conditioned "gradient"
print("singular values before:", np.round(np.linalg.svd(G, compute_uv=False)[[0, 63, 127, 255]], 4))
O = newton_schulz(G)
print("singular values after: ", np.round(np.linalg.svd(O, compute_uv=False)[[0, 63, 127, 255]], 3))
# singular values before: [19.0457  6.1886  2.1386  0.1189]
# singular values after:  [1.202 1.068 0.898 0.562]
```

A spread of 160× between the largest and smallest singular directions becomes about 2×: rare-but-important directions get as much update as dominant ones. (The tuned coefficients deliberately trade exact convergence to 1 for speed — five matmul-only iterations, cheap in BF16 on tensor cores.)

- **Applies to hidden weight matrices only**; embeddings, the output head, norms and biases still use AdamW.
- **Memory**: one momentum buffer per parameter — half of AdamW's optimizer state.
- **At scale**: Moonshot's *Muon is Scalable* (2025) added weight decay and per-shape update scaling and reported roughly 2× compute efficiency over AdamW; **Kimi K2** pre-trained a 1T MoE with **MuonClip** (Muon plus QK-clip against exploding attention logits); GLM-4.5 and **DeepSeek-V4** also report Muon.
- **Caveat**: a careful 2025 benchmark (*Fantastic Pretraining Optimizers and Where to Find Them*) found that against a properly tuned AdamW, the speed-up of matrix optimizers like Muon and SOAP shrinks with model size — to about 1.1× at 1.2B parameters in their runs. Claims of large speed-ups usually compare against an under-tuned baseline.

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

### Trapezoidal / Linear Decay to Zero

WSD with a *linear* decay is a trapezoid: warmup → flat → linear decay. A 2025 study (*Straight to Zero*) found that decaying linearly all the way to zero beats the traditional cosine-to-10% at compute-optimal and longer training durations — the final low-LR phase averages out gradient noise.

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
| **AdamW** | 3× params (weights + m + v) | Baseline | Universal | The default since 2019 |
| **Adam-mini** | ~2× params | ≈ AdamW | Niche | Block-wise second moment |
| **Adafactor** | ~1.1× params | Slightly worse | T5-era, TPU | Factored second moment, no momentum |
| **SOAP** | > 3× params | Faster per step at small scale | Research | Shampoo preconditioner + Adam |
| **Muon** (+ AdamW for non-matrix params) | ~2× params | ~1.1–2× vs AdamW depending on tuning and scale | Kimi K2, GLM-4.5, DeepSeek-V4 | Orthogonalised momentum for 2D weights |
| **SGD + momentum** | 2× params | Poor for transformers | Rarely | No per-parameter adaptivity |

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
    <div class="card-desc">Count it in tokens (B × T). Below the "critical batch size" larger batches give near-linear speed-ups; above it, returns vanish. For Adam, LR grows roughly with √batch, not linearly. Typical: 4M–60M tokens/step, often ramped up during training.</div>
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

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Adam normalises each parameter's update by its own gradient history; Muon normalises each matrix's update by its singular values. If the "right" geometry for a weight matrix is the operator norm rather than coordinate-wise scale, what other parts of training — learning-rate transfer across model widths, low-precision updates, MoE expert weights — should change?</p>
</div>

## Key Takeaways

- Cross-entropy (NLL) is the base loss; compare models with different tokenizers by bits per byte, not perplexity.
- Auxiliary losses keep training healthy: z-loss for logit drift, load balancing (or DeepSeek's bias-based balancing) for MoE.
- Post-training swaps the objective: KD (match a teacher), Bradley–Terry (reward model), DPO family (direct preferences), clipped policy gradient (PPO/GRPO).
- AdamW costs 16 bytes per parameter in mixed precision — 112 GB for a 7B model — before activations.
- Muon orthogonalises the momentum of each weight matrix with Newton–Schulz; it went from speed-run trick (2024) to frontier pre-training (Kimi K2, DeepSeek-V4) — with real but tuning-sensitive gains.

## Check Yourself

<details class="check">
<summary>Model A (32K vocabulary) has perplexity 8 on a document; model B (200K vocabulary) has perplexity 11 on the same document. Is A better?</summary>
<p>Not necessarily. B splits the text into fewer, harder tokens, which raises per-token perplexity even at equal quality. Convert both to bits per byte — total NLL in bits divided by the document's byte count — and compare those.</p>
</details>

<details class="check">
<summary>How much memory do weights, gradients and AdamW state need for a 13B model in standard mixed precision?</summary>
<p>16 bytes × 13 × 10⁹ = 208 GB, before activations — at least three 80 GB GPUs even with perfect sharding.</p>
</details>

<details class="check">
<summary>What does decoupled weight decay change, in one line?</summary>
<p>AdamW shrinks weights by (1 − ηλ) directly, instead of adding λθ to the gradient where Adam's per-parameter normalisation would rescale it — so decay strength no longer depends on each parameter's gradient history.</p>
</details>

<details class="check">
<summary>Why does Muon not apply to the embedding table and output head?</summary>
<p>Their rows correspond to tokens, which are updated sparsely and very unevenly; orthogonalising the whole matrix would spread updates onto rows whose tokens never appeared in the batch. Element-wise AdamW handles that sparsity naturally, as it does for 1D norm and bias parameters.</p>
</details>

<details class="check">
<summary>In GRPO, four samples for one prompt get rewards 1, 0, 0, 1. What are their advantages?</summary>
<p>Mean 0.5, standard deviation 0.5, so advantages are +1, −1, −1, +1: every token of the correct samples is pushed up, every token of the incorrect ones pushed down. If all four had the same reward, every advantage would be 0 and the prompt would contribute no gradient.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Adam: A Method for Stochastic Optimization | Kingma & Ba, 2014 | [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) | Adam |
| Decoupled Weight Decay Regularization | Loshchilov & Hutter, 2017 | [arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101) | AdamW |
| Adafactor | Shazeer & Stern, 2018 | [arxiv.org/abs/1804.04235](https://arxiv.org/abs/1804.04235) | Factored second moments |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al., 2019 | [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054) | 16 bytes per parameter accounting |
| Switch Transformers | Fedus et al., 2021 | [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961) | Load-balancing loss |
| Direct Preference Optimization | Rafailov et al., 2023 | [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) | DPO |
| Adam-mini | Zhang et al., 2024 | [arxiv.org/abs/2406.16793](https://arxiv.org/abs/2406.16793) | Block-wise learning rates |
| SOAP: Improving and Stabilizing Shampoo using Adam | Vyas et al., 2024 | [arxiv.org/abs/2409.11321](https://arxiv.org/abs/2409.11321) | SOAP |
| Muon: An optimizer for hidden layers | Jordan et al., 2024 | [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/) | Muon and Newton–Schulz coefficients |
| Muon is Scalable for LLM Training | Liu et al. (Moonshot), 2025 | [arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982) | Muon at scale |
| Fantastic Pretraining Optimizers and Where to Find Them | Wen et al., 2025 | [arxiv.org/abs/2509.02046](https://arxiv.org/abs/2509.02046) | Fair optimizer comparisons |
| Straight to Zero: Why Linearly Decaying the LR to Zero Works Best | Bergsma et al., 2025 | [arxiv.org/abs/2502.15938](https://arxiv.org/abs/2502.15938) | Linear decay to zero |

---

**Next:** [Chapter 12 — Mid-Training & Continued Pre-Training](./12_mid_training.md) — the stage between raw pre-training and post-training where long context, reasoning and data annealing happen.

[← Back to Table of Contents](./README.md)
