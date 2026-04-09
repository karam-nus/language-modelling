---
title: "Chapter 30 — Beyond Transformers — SSMs & Alternatives"
---

[← Back to Table of Contents](./README.md)

# Chapter 30 — Beyond Transformers — SSMs & Alternatives

> *"Attention is all you need — until your sequence is a million tokens long and the O(T²) cost makes you need something else."*

## The Quadratic Attention Problem

Standard attention computes a T×T score matrix:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- Compute: O(T² · d)
- Memory: O(T²) for the attention matrix (O(T) with Flash Attention, but compute remains O(T²))

At T = 1M tokens, even with Flash Attention, the FLOPs for attention alone become prohibitive. This motivates architectures with **linear or sub-quadratic** complexity.

## State Space Models (SSMs)

SSMs map an input sequence to an output sequence through a **latent state**:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t \quad \text{(state update)}
$$
$$
y_t = C h_t + D x_t \quad \text{(output)}
$$

Where:
- $h_t \in \mathbb{R}^N$ — hidden state (per channel)
- $\bar{A}, \bar{B}$ — discretized system matrices
- $C, D$ — output projection matrices

<div class="diagram">
<div class="diagram-title">SSM vs Attention</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Attention</div>
    <ul>
      <li>Each token can attend to all previous tokens</li>
      <li>KV-cache grows with sequence length</li>
      <li>O(T²d) compute per layer</li>
      <li>Excellent at precise retrieval from context</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">SSM (Mamba)</div>
    <ul>
      <li>Information compressed into fixed-size state</li>
      <li>Constant memory during generation (no KV-cache)</li>
      <li>O(Td) compute per layer</li>
      <li>Better at summarizing long sequences, weaker at precise recall</li>
    </ul>
  </div>
</div>
</div>

### S4: Structured State Spaces (2022)

The breakthrough that made SSMs competitive:
- **HiPPO initialization** for matrix A — enables remembering long-range dependencies
- **Parallel scan** — the recurrence can be computed as a convolution during training (parallelizable on GPUs)
- During training: O(T log T) via FFT-based convolution
- During inference: O(1) per step via the recurrence

### Mamba: Selective State Spaces (2023)

Mamba (Gu & Dao) made SSMs competitive with transformers by making the state matrices **input-dependent** (selective):

<div class="diagram">
<div class="diagram-title">Mamba Block</div>
<div class="flow">
  <div class="flow-step accent">Input: [B, T, d]</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">Linear projection → expand to [B, T, 2·d_inner]</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">1D convolution (causal, short kernel)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step orange">Selective SSM: B, C, Δ are functions of the input</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">Gate × SSM output → linear projection → [B, T, d]</div>
</div>
</div>

Key innovation: standard SSMs have **fixed** A, B, C matrices (same for every input). Mamba makes B, C, and the discretization step Δ **input-dependent**, allowing the model to selectively focus on or ignore different parts of the input.

```python
# Simplified selective SSM (conceptual)
def selective_ssm(x, A, D, dt_proj, B_proj, C_proj, conv1d):
    """
    x: [B, T, d_inner] — input after expansion
    """
    # Input-dependent parameters (SELECTIVE)
    delta = F.softplus(dt_proj(x))    # [B, T, d_inner] — controls forgetting
    B = B_proj(x)                      # [B, T, N] — input-to-state
    C = C_proj(x)                      # [B, T, N] — state-to-output
    
    # Discretize
    A_bar = torch.exp(delta.unsqueeze(-1) * A)  # [B, T, d_inner, N]
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2) # [B, T, d_inner, N]
    
    # Parallel selective scan (efficient CUDA kernel)
    y = selective_scan(x, A_bar, B_bar, C, D)  # [B, T, d_inner]
    return y
```

Mamba-1 3B matches Transformer 3B on language modeling, with **5× higher throughput** on long sequences.

### Mamba-2 (2024)

Mamba-2 showed that selective SSMs can be viewed as a form of **structured linear attention**:
- 2–8× faster than Mamba-1 via SSD (Structured State Space Duality) algorithm
- Connections to linear attention make it easier to reason about theoretically
- Still O(Td) but with better hardware utilization

## Linear Attention

Replace the softmax attention kernel with a linear kernel:

$$
\text{LinearAttn}(Q, K, V) = \phi(Q)(\phi(K)^T V)
$$

By computing $\phi(K)^T V$ first (an N×d matrix), we avoid the T×T attention matrix:
- Compute: O(T · d · N) where N is the feature dimension
- Memory: O(T · d) — no attention matrix stored

Variants: **RetNet**, **HGRN2**, **GLA** (Gated Linear Attention).

## RWKV: RNN-Transformer Hybrid

RWKV combines the **parallelizable training** of transformers with the **constant-memory inference** of RNNs:

- Training: attention-like parallelism using WKV operator
- Inference: recurrent — process one token at a time with fixed state
- No KV-cache — state size is fixed regardless of context length

| RWKV Version | Key Innovation |
|------|---------------|
| RWKV-4 | Initial architecture, WKV attention |
| RWKV-5 (Eagle) | Multi-head attention variant |
| RWKV-6 (Finch) | Data-dependent linear recurrence |
| RWKV-7 (Goose) | Latest, improved expressiveness |

## Hybrid Architectures

The most promising direction may be **combining** attention and SSMs:

### Jamba (AI21, 2024)

<div class="diagram">
<div class="diagram-title">Jamba Architecture — Hybrid Layers</div>
<div class="layer-stack">
  <div class="layer accent">Attention Layer — precise retrieval, in-context learning</div>
  <div class="layer green">Mamba Layer — efficient sequence mixing</div>
  <div class="layer green">Mamba Layer — most layers are Mamba (cheaper)</div>
  <div class="layer accent">Attention Layer — every few layers for recall tasks</div>
  <div class="layer green">Mamba Layer</div>
  <div class="layer green">Mamba Layer</div>
  <div class="layer purple">MoE FFN — some layers use MoE for capacity</div>
</div>
<div style="text-align: center; color: var(--text-secondary); margin-top: 0.5rem;">
  Jamba: Mamba + Attention + MoE — ratio of ~7:1 Mamba-to-Attention layers
</div>
</div>

Jamba 52B supports 256K context with **much less memory** than a pure attention model, because most layers don't have a KV-cache.

### Other Hybrids

| Model | Architecture | Context |
|-------|------------|---------|
| Zamba (Zyphra) | Mamba + shared attention layers | 4K–16K |
| RecurrentGemma (Google) | Griffin: local attention + linear recurrence | 8K |
| Samba (Microsoft) | Mamba + sliding window attention | up to 256K |

## Complexity Comparison

| Architecture | Training Compute (per layer) | Inference (per token) | Memory (generation) |
|-------------|-------|----------|---------|
| Standard Attention | O(T²d) | O(Td) | O(T · n_kv_heads · d_head) — KV-cache |
| Flash Attention | O(T²d) | O(Td) | O(T · n_kv_heads · d_head) — KV-cache |
| Sliding Window | O(T · w · d) | O(w · d) | O(w · n_kv_heads · d_head) |
| Linear Attention | O(T · d²) | O(d²) | O(d²) — fixed state |
| SSM (Mamba) | O(T · d · N) | O(d · N) | O(d · N) — fixed state |
| **Hybrid** | Mixed | Mixed | Reduced KV-cache (attention layers only) |

## Current State

As of early 2026:
- **Transformers dominate**: the ecosystem, tooling, and proven scaling make them the default
- **Mamba/SSMs**: competitive at small scale, promising at long context, but haven't dethroned transformers at frontier scale
- **Hybrids win on efficiency**: Jamba, Samba show that mixing architectures gives the best memory-quality tradeoff
- **The jury is still out**: no alternative has beaten a well-trained transformer at the same compute budget on standard benchmarks

## What's Next

Architecture innovations help, but there's another way to improve LLM output quality: **spend more compute at inference time**. The next chapter covers **reasoning models** — from chain-of-thought prompting to o1-style test-time compute scaling.

[← Previous: Chapter 33 — Mixture of Experts](./33_mixture_of_experts.md) · **Next: [Chapter 35 — Reasoning Models →](./35_reasoning_models.md)**

---

*Last updated: April 2026*
