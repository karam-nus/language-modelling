---
title: "Chapter 2 — Embeddings & Representations"
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — Embeddings & Representations

> *"You shall know a word by the company it keeps."*
> — J.R. Firth, 1957

## From IDs to Vectors

After tokenization (see [Appendix B](./appendix_b_tokenization.md)), each token is an integer ID — say, `4037` for the token "token". But a model can't learn meaningful patterns from raw integers. The **embedding layer** maps each token ID to a dense vector in a continuous space where similar tokens live near each other.

<div class="diagram">
<div class="diagram-title">Embedding Lookup</div>
<div class="flow">
  <div class="flow-node wide">Token IDs: [14658, 4211, 7963]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Embedding Matrix <small>E ∈ ℝ^(V × d_model)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Dense Vectors: [B, T, d_model]</div>
</div>
</div>

The embedding matrix **E** has shape `[V, d_model]` — one row per token in the vocabulary. Looking up a token is simply indexing into this matrix:

<div class="diagram">
<div class="diagram-title">Tensor Shapes — Embedding Lookup</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 140px;">Input token IDs</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 140px;">Embedding matrix</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim vocab">V</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 140px;">Output embeddings</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
</div>
</div>

For LLaMA-3-8B: V = 128,256 and d_model = 4,096, so the embedding matrix alone is 128,256 × 4,096 × 2 bytes (BF16) ≈ **1 GB**.

```python
import torch
import torch.nn as nn

vocab_size = 128256
d_model = 4096

# The embedding layer — a learnable lookup table
embedding = nn.Embedding(vocab_size, d_model)

# Input: batch of token IDs
token_ids = torch.tensor([[14658, 4211, 7963, 279]])  # [1, 4]
vectors = embedding(token_ids)                          # [1, 4, 4096]
print(f"Input shape:  {token_ids.shape}")   # [1, 4]
print(f"Output shape: {vectors.shape}")     # [1, 4, 4096]
```

## The Evolution of Word Representations

<div class="diagram">
<div class="diagram-title">Word Representation Timeline</div>
<div class="layer-stack">
  <div class="layer purple">🔄 Contextual Embeddings (2018+) <small>BERT, GPT — same word gets different vectors based on context</small></div>
  <div class="layer accent">📐 Static Embeddings (2013) <small>Word2Vec, GloVe — one fixed vector per word</small></div>
  <div class="layer orange">🔢 One-Hot Vectors <small>Sparse, no semantic similarity, V-dimensional</small></div>
  <div class="layer">📝 Symbolic / Discrete <small>Words as opaque symbols — no notion of similarity</small></div>
</div>
</div>

### One-Hot Encoding

The simplest representation: each word is a binary vector of length V with a single 1 at the word's index. For a vocabulary of 128K tokens, each vector is 128K-dimensional and sparse. The dot product between any two different one-hot vectors is zero — the representation carries no semantic information.

### Word2Vec and GloVe (Static Embeddings)

**Word2Vec** (Mikolov et al., 2013) demonstrated that training a simple neural network on "predict a word from its neighbors" produces vectors where semantic relationships are encoded as linear directions:

<div class="diagram">
<div class="diagram-title">Word2Vec — Semantic Arithmetic</div>
<div class="flow-h">
  <div class="flow-node accent narrow">king</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange narrow">− man</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green narrow">+ woman</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple narrow">≈ queen</div>
</div>
</div>

These embeddings are **static** — each word maps to one fixed vector regardless of context. "Bank" has the same embedding whether you're talking about a river bank or a financial bank.

### Contextual Embeddings (Transformers)

In modern LLMs, embeddings are **contextual**: the initial embedding lookup produces static vectors, but the transformer layers refine them based on the surrounding context. After multiple transformer layers, the representation for "bank" in "*river bank*" is completely different from "bank" in "*bank account*". The initial embedding is just the starting point.

## Positional Encodings

Attention (covered in [Chapter 4](./04_attention_sdpa_and_mha.md)) is inherently **position-agnostic** — it treats the input as a set, not a sequence. Without positional information, the model can't distinguish "the cat sat on the mat" from "the mat sat on the cat". Positional encodings inject sequence order.

### Sinusoidal (Original Transformer)

The original "Attention Is All You Need" paper used fixed, non-learned sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Each position gets a unique d_model-dimensional pattern. The beauty: relative positions are captured as linear transformations, and the model can extrapolate to unseen lengths (in theory).

```python
import torch
import math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [max_len, d_model]

pe = sinusoidal_pe(max_len=2048, d_model=512)  # [2048, 512]
print(f"Positional encoding shape: {pe.shape}")
```

### Learned Positional Embeddings (GPT-2)

GPT-2 used a learned embedding table for positions — just like the token embedding but indexed by position (0, 1, 2, ..., max_len-1). Simple and effective, but fixed to a maximum sequence length at training time.

```python
# Learned positional embedding
max_seq_len = 2048
position_embedding = nn.Embedding(max_seq_len, d_model)  # [2048, 4096]

positions = torch.arange(0, seq_len).unsqueeze(0)  # [1, T]
pos_emb = position_embedding(positions)              # [1, T, d_model]

# Combine token + position embeddings
hidden = token_embedding(input_ids) + pos_emb  # [B, T, d_model]
```

### RoPE — Rotary Position Embedding (LLaMA, Mistral, Qwen)

**RoPE** (Su et al., 2021) is the dominant positional encoding in modern LLMs. Instead of adding positional information to embeddings, RoPE **rotates** the query and key vectors in attention by an angle proportional to their position. The rotation ensures that the dot product between any two positions depends only on their **relative distance**.

For a pair of adjacent dimensions (2i, 2i+1), RoPE applies a 2D rotation:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

Where *m* is the position index and $\theta_i = 10000^{-2i/d}$.

<div class="diagram">
<div class="diagram-title">RoPE — Rotation in 2D Subspaces</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Low Frequency</div>
    <div class="card-desc">First dimensions rotate slowly → captures long-range position</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Mid Frequency</div>
    <div class="card-desc">Middle dimensions → medium-range dependencies</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">High Frequency</div>
    <div class="card-desc">Last dimensions rotate fast → fine-grained local position</div>
  </div>
</div>
</div>

```python
def rope_frequencies(d_head, max_len=8192, base=10000):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, freqs)  # [max_len, d_head/2]
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    """Apply RoPE to queries or keys. x: [B, H, T, d_head]"""
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    cos_t = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_half]
    sin_t = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

cos, sin = rope_frequencies(d_head=128)
# Apply to Q and K before attention: q_rot = apply_rope(q, cos, sin)
```

RoPE is applied **inside attention** (to Q and K only), not added to the embeddings. This is a key difference from sinusoidal and learned positional encodings.

### ALiBi — Attention with Linear Biases

**ALiBi** (Press et al., 2022) takes a different approach: no positional encoding at all. Instead, it adds a **linear bias** to attention scores based on the distance between query and key positions. Position *i* attending to position *j* gets a bias of $-m \cdot |i - j|$ where *m* is a head-specific slope.

ALiBi has strong length extrapolation properties — models trained on short sequences can generalize to longer ones at inference without re-training.

## Positional Encoding Comparison

| Method | Added Where | Relative Position | Length Extrapolation | Used By |
|--------|------------|-------------------|---------------------|---------|
| **Sinusoidal** | Embedding output | Implicit (via rotation) | Moderate | Original Transformer |
| **Learned** | Embedding output | No | None (fixed max_len) | GPT-2, BERT |
| **RoPE** | Q, K in attention | Yes (rotation angle) | Good (with scaling) | LLaMA, Mistral, Qwen, Gemma |
| **ALiBi** | Attention scores | Yes (linear bias) | Strong | BLOOM, MPT |

Modern models overwhelmingly use **RoPE** due to its balance of performance, relative position awareness, and extensibility (the base frequency can be scaled for longer contexts — discussed in [Chapter 12](./12_mid_training.md)).

## Tying Embeddings

Many models **tie** the input embedding matrix with the output projection (the "LM head"). This means the matrix used to look up token embeddings is the same matrix used to project hidden states back to vocabulary logits:

$$\text{logits} = h \cdot E^T$$

Where $h$ is the final hidden state `[B, T, d_model]` and $E^T$ is the transposed embedding matrix `[d_model, V]`.

This halves the number of embedding parameters (significant when V is large) and enforces consistency — the output space matches the input space.

## What's Next

Embeddings give tokens their initial representations. The next chapter covers the architecture that transforms these representations into something powerful — the **transformer**.

**Next: [Chapter 3 — The Transformer →](./03_the_transformer.md)**

---

*Last updated: April 2026*
