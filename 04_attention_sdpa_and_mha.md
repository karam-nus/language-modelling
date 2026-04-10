---
title: "Chapter 4 — Attention Deep Dive: SDPA & Multi-Head Attention"
---

[← Back to Table of Contents](./README.md)

# Chapter 4 — Attention Deep Dive: SDPA & Multi-Head Attention

> *"The key innovation of attention is allowing the model to dynamically focus on relevant parts of the input, rather than compressing everything into a fixed-size vector."*

## Scaled Dot-Product Attention (SDPA)

Attention is the mechanism that lets each token look at every other token in the sequence and decide what's relevant. At its core, it's a soft lookup: queries ask questions, keys advertise content, and values hold the actual information.

Given three matrices — **Query (Q)**, **Key (K)**, and **Value (V)** — scaled dot-product attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

<div class="diagram">
<div class="diagram-title">SDPA — Step by Step</div>
<div class="flow">
  <div class="flow-node accent wide">1. Compute scores: QK^T <small>[B, H, T, T]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">2. Scale by 1/√d_k <small>prevents softmax saturation</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">3. Apply causal mask <small>−∞ for future positions</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">4. Softmax → attention weights <small>[B, H, T, T], rows sum to 1</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">5. Weighted sum of values <small>weights × V → [B, H, T, d_k]</small></div>
</div>
</div>

### Tensor Shapes Through SDPA

<div class="diagram">
<div class="diagram-title">SDPA Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Q (queries)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">K (keys)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">QK^T (scores)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">← This is O(T²) in memory!</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">V (values)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Output</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
</div>
</div>

### Why Scale by √d_k?

Without scaling, the dot product of Q and K grows proportionally to d_k. Large dot products push softmax into saturated regions where gradients vanish. Dividing by √d_k keeps the variance of the scores at ~1 regardless of dimension:

If $q_i, k_i \sim \mathcal{N}(0, 1)$, then $\text{Var}(q \cdot k) = d_k$. After scaling: $\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1$.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [B, H, T, d_k]
    mask: [T, T] or [B, 1, T, T] — True means IGNORE
    Returns: [B, H, T, d_k]
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, T, T]

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
    output = torch.matmul(weights, V)    # [B, H, T, d_k]
    return output, weights
```

## Causal Masking

In **decoder-only** models (GPT, LLaMA), each token can only attend to itself and previous tokens — it must not see the future. This is enforced with a **causal mask**: an upper-triangular matrix of `−∞` values that, after softmax, zero out attention to future positions.

<div class="diagram">
<div class="diagram-title">Causal (Lower-Triangular) Attention Mask</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Mask Matrix (T=5)</div>
    <ul>
      <li>✓ · · · ·</li>
      <li>✓ ✓ · · ·</li>
      <li>✓ ✓ ✓ · ·</li>
      <li>✓ ✓ ✓ ✓ ·</li>
      <li>✓ ✓ ✓ ✓ ✓</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">After Softmax</div>
    <ul>
      <li>1.0  0   0   0   0</li>
      <li>0.6 0.4  0   0   0</li>
      <li>0.2 0.3 0.5  0   0</li>
      <li>0.1 0.2 0.3 0.4  0</li>
      <li>0.1 0.1 0.2 0.3 0.3</li>
    </ul>
  </div>
</div>
</div>

```python
def causal_mask(T, device='cpu'):
    """Returns a boolean mask: True = masked (ignored)."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

mask = causal_mask(5)
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]])
```

## Multi-Head Attention (MHA)

<div class="img-caption">
  <img src="{{ '/assets/images/self_attention.svg' | relative_url }}" alt="Scaled dot-product attention showing Q/K/V projections, causal attention weight matrix, and multi-head concatenation">
  <figcaption>Scaled dot-product attention with causal masking and multi-head mechanism</figcaption>
</div>

Instead of computing a single attention function, **multi-head attention** runs H parallel attention operations (called "heads"), each on a different d_k-dimensional subspace. This lets different heads learn different types of relationships (syntactic, semantic, positional, etc.).

<div class="diagram">
<div class="diagram-title">Multi-Head Attention — Split, Attend, Concatenate</div>
<div class="flow">
  <div class="flow-node accent wide">Input x: [B, T, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Linear projections: W_Q, W_K, W_V <small>each [d_model, d_model] → Q, K, V: [B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Reshape to [B, H, T, d_k] <small>d_k = d_model / H</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">H parallel SDPA heads <small>each: [B, 1, T, d_k] → [B, 1, T, d_k]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Concatenate → [B, T, d_model] <small>reshape back</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Output projection W_O <small>[d_model, d_model] → [B, T, d_model]</small></div>
</div>
</div>

For LLaMA-3-8B: d_model = 4096, H = 32 heads, d_k = 4096/32 = **128 per head**.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # Project and reshape: [B, T, d_model] → [B, H, T, d_k]
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)  # [B, H, T, d_k]

        # Concatenate heads: [B, H, T, d_k] → [B, T, d_model]
        concat = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

        # Final projection
        return self.W_o(concat)  # [B, T, d_model]
```

## Self-Attention vs Cross-Attention

<div class="diagram">
<div class="diagram-title">Self-Attention vs Cross-Attention</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Self-Attention</div>
    <ul>
      <li>Q, K, V all come from the same sequence</li>
      <li>Each token attends to all other tokens in the sequence</li>
      <li>Used in both encoder and decoder</li>
      <li>Decoder uses causal masking</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Cross-Attention</div>
    <ul>
      <li>Q from decoder, K and V from encoder</li>
      <li>Decoder tokens attend to encoder representations</li>
      <li>Used in encoder-decoder models (T5, BART)</li>
      <li>Not present in decoder-only models (GPT, LLaMA)</li>
    </ul>
  </div>
</div>
</div>

In self-attention, the query, key, and value all come from the same input. In cross-attention, Q comes from one sequence (decoder) while K and V come from another (encoder output). Decoder-only models like GPT and LLaMA use only self-attention.

## Flash Attention

The attention score matrix `[B, H, T, T]` is **O(T²)** in memory. For T = 128K tokens with 32 heads and FP16, that's 128K × 128K × 32 × 2 bytes = **1 TB** — far too large to materialize. **Flash Attention** (Dao et al., 2022) solves this by:

1. **Tiling**: Compute attention in small blocks that fit in GPU SRAM (shared memory)
2. **Online softmax**: Compute softmax incrementally without materializing the full score matrix
3. **Kernel fusion**: Fuse the matmul, softmax, and output matmul into a single GPU kernel

<div class="diagram">
<div class="diagram-title">Flash Attention — Tiled Computation</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Standard Attention</div>
    <div class="card-desc">Materializes full T×T matrix in HBM. O(T²) memory. Multiple kernel launches.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Flash Attention</div>
    <div class="card-desc">Processes T×T in tiles within SRAM. O(T) memory. Single fused kernel. 2–4× faster.</div>
  </div>
</div>
</div>

In practice, you almost never implement attention yourself. PyTorch provides an optimized implementation that auto-selects the best backend (Flash Attention, memory-efficient attention, or math fallback):

```python
import torch.nn.functional as F

# PyTorch's optimized SDPA — automatically uses Flash Attention when available
output = F.scaled_dot_product_attention(
    Q, K, V,                    # [B, H, T, d_k]
    attn_mask=None,
    is_causal=True,             # applies causal mask internally
    dropout_p=0.0,
)  # → [B, H, T, d_k]
```

## Attention Head Specialization

Different heads learn to attend to different things. Research has shown that transformer heads naturally specialize:

| Head Type | What It Learns | Example |
|-----------|---------------|---------|
| **Positional** | Attend to nearby tokens | "the → cat" (adjacent word) |
| **Syntactic** | Subject-verb agreement | "The cats → are" (long-range grammar) |
| **Copying** | Attend to identical or similar tokens | Repetitions, references |
| **Induction** | [A][B]...[A] → predict [B] | In-context learning patterns |
| **Rare/dead** | Low-entropy, nearly uniform | Some heads are redundant |

Understanding head specialization is important for attention variant design — some heads are more important than others, which motivates the grouped and multi-query approaches in [Chapter 5](./05_attention_gqa_mqa_mla.md).

## What's Next

Standard multi-head attention gives each head its own Q, K, and V projections — but this creates a memory bottleneck when caching K and V during inference. The next chapter explores how modern models address this with **grouped-query attention (GQA)**, **multi-query attention (MQA)**, and **multi-latent attention (MLA)**.

[← Previous: Chapter 3 — The Transformer](./03_the_transformer.md) · **Next: [Chapter 5 — GQA, MQA & MLA →](./05_attention_gqa_mqa_mla.md)**

---

*Last updated: April 2026*
