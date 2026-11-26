---
title: "Chapter 2 — Embeddings & Representations"
description: "Embedding tables, the road from one-hot to contextual vectors, and positional encodings: sinusoidal, learned, RoPE and its context-extension tricks, M-RoPE for images and video, ALiBi, NoPE and partial RoPE."
related:
  - { id: math, why: "Vectors, dot products and rotations behind embeddings and RoPE." }
---

[← Back to Table of Contents](./README.md)

# Chapter 2 — Embeddings & Representations

> *"You shall know a word by the company it keeps."*
> — J. R. Firth, 1957

> **The one-sentence version:** an embedding table turns each token ID into a learned vector, and a positional scheme — today almost always a rotation of the attention queries and keys — tells the model where each vector sits in the sequence.

## From IDs to Vectors

After tokenization (see [Appendix B](./appendix_b_tokenization.md)), each token is an integer ID — say, `4037` for `" token"` in some vocabulary. But a model can't learn meaningful patterns from raw integers. The **embedding layer** maps each token ID to a dense vector in a continuous space where similar tokens live near each other.

<div class="diagram">
<div class="diagram-title">Embedding Lookup</div>
<div class="flow">
  <div class="flow-node wide">Token IDs: [B, T] integers</div>
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
    <span class="ts-label">Input token IDs</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Embedding matrix</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim vocab">V</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Output embeddings</span>
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

For Llama 3.1 8B: V = 128,256 and d_model = 4,096, so the embedding matrix alone is 128,256 × 4,096 × 2 bytes (BF16) ≈ **1.05 GB** — 525M parameters, about 6.5% of the model. In a 0.6B model with a 152K vocabulary (Qwen3-0.6B, d_model = 1,024) the same table is 156M parameters, a quarter of the whole network, which is why small models share it with the output head (see *Tying Embeddings* below).

```python
import torch
import torch.nn as nn

vocab_size = 128256
d_model = 4096

# The embedding layer — a learnable lookup table
embedding = nn.Embedding(vocab_size, d_model)

# Input: batch of token IDs
token_ids = torch.randint(0, vocab_size, (1, 4))  # [1, 4]
vectors = embedding(token_ids)                     # [1, 4, 4096]
print(token_ids.shape, vectors.shape)
# torch.Size([1, 4]) torch.Size([1, 4, 4096])
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

The heatmap below shows positions (rows) vs dimensions (columns). High-frequency oscillations in the early dimensions uniquely identify close positions; low-frequency oscillations in later dimensions encode long-range order.

![Sinusoidal positional encoding heatmap — rows are positions, columns are dimensions]({{ '/assets/images/sinusoidal_pe.svg' | relative_url }})

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

**RoPE** (Su et al., 2021) is the dominant positional encoding in modern LLMs. Instead of adding a position vector to the embedding, RoPE **rotates** the query and key vectors inside each attention head by an angle proportional to the token's position. Because both Q and K are rotated the same way, the dot-product attention score naturally captures only the **relative distance** between positions — not absolute positions.

#### Core Idea: Rotation in 2D Subspaces

RoPE partitions each d_head-dimensional query/key vector into d_head/2 consecutive **pairs** of dimensions. For each pair (2i, 2i+1), it applies a 2D rotation matrix parameterised by the position index *m* and a base frequency $\theta_i$:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

Where the **base frequency** for pair *i* is:

$$\theta_i = \frac{1}{10000^{2i/d_{head}}}$$

This means each pair of dimensions rotates at a different angular speed — high-index pairs rotate slowly (long-range), low-index pairs rotate quickly (short-range).

#### Why Relative Position Falls Out Naturally

When two rotated vectors $q'$ (at position $m$) and $k'$ (at position $n$) are dot-producted in attention, the rotation angles cancel to leave only $(m - n)$:

$$q'^{\top} k' = \text{Re}\!\left[\sum_{i=0}^{d/2-1} q_{[i]} k_{[i]}^{*} e^{j(m-n)\theta_i}\right]$$

The model never needs to learn absolute positions — relative offsets are **built into the geometry**. Drag the sliders below: move both tokens together and the score does not change; change the gap and it does.

<div class="lab" data-lab="rope"><noscript>Interactive lab: rotate a query and key pair at several RoPE frequencies and watch the attention score depend only on their distance. Enable JavaScript to use it.</noscript></div>

#### Angle Rotation Visualization

The diagram below shows how a query vector rotates at three frequency bands as its position advances. The first pair (θ₀ = 1) turns a full radian — about 57° — per position, so it wraps around every ~6 tokens and resolves fine local order. The last pair (θ ≈ 10⁻⁴ with base 10,000) turns so slowly that it takes tens of thousands of positions to complete a revolution, so it encodes coarse, long-range position.

![RoPE angle rotation across frequency bands]({{ '/assets/images/rope_rotation.svg' | relative_url }})

#### Tensor Shapes

<div class="diagram">
<div class="diagram-title">RoPE Tensor Shapes Inside Attention</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Q or K (after linear proj)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_head</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Precomputed cos/sin cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_head/2</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">one cos &amp; one sin table</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Q or K after RoPE</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_head</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">same shape, values rotated</span>
  </div>
</div>
</div>

RoPE does **not** change the shape of Q or K — it only rotates the values in place.

<div class="callout warn">
<span class="callout-title">Two pairing conventions</span>
<p>The paper pairs <em>adjacent</em> dimensions (2i, 2i+1) — the "GPT-J / interleaved" layout. Hugging Face's Llama code instead pairs dimension <em>j</em> with <em>j + d/2</em> ("GPT-NeoX / rotate-half"), which is what the code below does. Both are the same mathematics on a permuted basis, and checkpoint converters permute the Q and K projection weights to switch between them — but mixing conventions between a checkpoint and a kernel silently scrambles positions. It is one of the most common bugs when porting a model.</p>
</div>

```python
import torch

def rope_frequencies(d_head: int, max_len: int = 8192, base: float = 10000.0):
    """Precompute cosine and sine tables for RoPE.
    Returns cos, sin each of shape [max_len, d_head/2].
    """
    # θᵢ = 1 / base^(2i/d_head)  for i = 0, 1, ..., d_head/2 - 1
    i = torch.arange(0, d_head, 2).float()          # step 2 → d_head/2 elements
    theta = 1.0 / (base ** (i / d_head))             # [d_head/2]

    positions = torch.arange(max_len).float()         # [max_len]
    angles = torch.outer(positions, theta)            # [max_len, d_head/2]
    return torch.cos(angles), torch.sin(angles)       # each [max_len, d_head/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to Q or K.
    x:   [B, H, T, d_head]
    cos: [T, d_head/2]
    sin: [T, d_head/2]
    Returns rotated tensor of same shape [B, H, T, d_head].
    """
    T, d_half = x.shape[2], x.shape[-1] // 2
    x1 = x[..., :d_half]                              # [B, H, T, d_head/2]
    x2 = x[..., d_half:]                              # [B, H, T, d_head/2]

    # Broadcast cos/sin from [T, d_half] → [1, 1, T, d_half]
    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)

    # Apply 2D rotation: [x1·cos − x2·sin, x2·cos + x1·sin]
    rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return rotated                                      # [B, H, T, d_head]


# Usage inside attention
d_head = 128
cos_cache, sin_cache = rope_frequencies(d_head, max_len=8192)

# q, k: [B, H, T, d_head]
q_rot = apply_rope(q, cos_cache, sin_cache)
k_rot = apply_rope(k, cos_cache, sin_cache)
# Then compute attention scores: q_rot @ k_rot.transpose(-1, -2) / sqrt(d_head)
```

#### Extending Context: Base, Interpolation and YaRN

A model trained on 8K-token sequences has never seen the slow-frequency pairs rotate past the angle they reach at position 8,192. Feed it position 50,000 and those pairs land at angles it has never been trained on, and attention degrades. Every context-extension method is a way to keep angles in familiar territory:

| Method | What it changes | Used by |
|---|---|---|
| **Larger base** | Raise `rope_theta` (10K → 500K–10M) so every pair rotates more slowly from the start | Llama 3 (500K), Qwen3 (1M), Gemma 3 global layers (1M) |
| **Position Interpolation** | Divide positions by the extension factor *s* — squeezes 32K positions into the trained 8K range | Early Llama 2 extensions |
| **NTK-aware / "by-parts"** | Interpolate only the low-frequency pairs; leave high-frequency pairs (local order) untouched | Llama 3.1's `rope_type: "llama3"` (factor 8, 8K → 128K) |
| **YaRN** | NTK-by-parts plus a softmax-temperature correction for the flatter attention at long range | Qwen2.5/Qwen3 (32K → 128K), DeepSeek-V3, gpt-oss |
| **LongRoPE** | Searches a separate rescale factor per frequency pair | Phi-3/Phi-4 long-context variants |

All of these are cheap to apply but still need some training at the longer length — the "long-context extension" phase of mid-training ([Chapter 12](./12_mid_training.md)). The `rope_scaling` block of a model's `config.json` tells you which one was used ([Appendix C](./appendix_c_model_config.md)).

---

### M-RoPE — Multimodal Rotary Position Embedding

**M-RoPE** (introduced in Qwen2-VL, 2024, and used in Qwen2.5-VL and Qwen3-VL) extends RoPE to handle inputs that have **multiple spatial or temporal dimensions** — images, video frames, or interleaved text+image sequences.

#### The Problem with 1D RoPE for Images

A standard 1D RoPE assigns each token a single scalar position: token 0, 1, 2, 3... For text this is natural. But an image patch at row *h*, column *w* has **two** spatial coordinates. Flattening patches into a 1D sequence loses the 2D structure — the model can't tell whether two adjacent patches are horizontally or vertically adjacent.

#### M-RoPE: Separate Frequency Bands per Axis

M-RoPE gives every token a **(t, h, w)** position triple and splits the d_head/2 rotation pairs into three sections, one per axis. In Qwen2-VL (d_head = 128, so 64 pairs) the config says `mrope_section: [16, 24, 24]`: 16 pairs rotate by the temporal index, 24 by the row, 24 by the column.

| Token type | (t, h, w) assignment | Effect |
|----------|------|---------------|
| **Text** | t = h = w = sequence position | Identical to ordinary 1D RoPE |
| **Image** | t fixed for the whole image; h, w = patch row and column | Neighbours above and beside a patch are distinguishable |
| **Video** | t = frame index (scaled to real time in Qwen2.5-VL); h, w as for images | Motion along time is encoded separately from space |

After an image or video, the next text position continues from one past the largest index used, so the sequence stays monotonic. Qwen3-VL's **interleaved M-RoPE** spreads t, h and w across the whole frequency range instead of giving each a contiguous block, so no axis is stuck with only high or only low frequencies.

Each token's RoPE rotation is computed from its (t, h, w) coordinates rather than a single scalar index:

$$q' = R(t \cdot \theta_{\text{time}}) \cdot R(h \cdot \theta_{\text{height}}) \cdot R(w \cdot \theta_{\text{width}}) \cdot q$$

Because each section rotates its own disjoint set of pairs, the three rotations commute and the product above is just "rotate these 16 pairs by t, those 24 by h, those 24 by w".

![M-RoPE positional grid for text, image, and video]({{ '/assets/images/mrope_grid.svg' | relative_url }})

#### M-RoPE Tensor Shapes

<div class="diagram">
<div class="diagram-title">M-RoPE Tensor Shapes — Image Tokens</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Image patch grid (H×W patches)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">H·W</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Position indices (h_ids, w_ids)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">H·W</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">one per spatial axis</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">RoPE angle tensor (t, h, w sections)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim seq">H·W</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_head/2</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">16 pairs ← t, 24 ← h, 24 ← w</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Q or K after M-RoPE</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">H·W</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_head</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
</div>
</div>

```python
import torch

def mrope_angles(t_ids, h_ids, w_ids, d_head=128, base=10000.0, section=(16, 24, 24)):
    """Qwen2-VL-style M-RoPE angles.
    t_ids, h_ids, w_ids: [N] position of each token along each axis.
    Returns angles [N, d_head/2]; pairs are split into contiguous t | h | w sections.
    """
    assert sum(section) == d_head // 2
    theta = 1.0 / base ** (torch.arange(0, d_head, 2).float() / d_head)   # [d_head/2]
    per_axis = torch.stack([torch.outer(ids.float(), theta)               # [3, N, d_head/2]
                            for ids in (t_ids, h_ids, w_ids)])
    chunks = per_axis.split(list(section), dim=-1)                        # take t's first 16 pairs, h's next 24, w's last 24
    return torch.cat([chunks[a][a] for a in range(3)], dim=-1)            # [N, d_head/2]

# A 14×14 grid of image patches that starts at sequence position 5
H, W, start = 14, 14, 5
t_ids = torch.full((H * W,), start)                    # same t for the whole image
h_ids = start + torch.arange(H).repeat_interleave(W)   # [196]
w_ids = start + torch.arange(W).repeat(H)              # [196]
angles = mrope_angles(t_ids, h_ids, w_ids)
print(angles.shape)                                    # torch.Size([196, 64])
cos, sin = angles.cos(), angles.sin()                  # applied to Q and K exactly as in 1D RoPE
```

### ALiBi — Attention with Linear Biases

**ALiBi** (Press et al., 2022) takes a different approach: no positional encoding at all. Instead, it adds a **linear bias** to attention scores based on the distance between query and key positions. Position *i* attending to position *j* gets a bias of $-m \cdot |i - j|$ where *m* is a head-specific slope.

ALiBi extrapolates well on perplexity — models trained on short sequences keep a sensible loss on longer ones — but the linear penalty also makes distant tokens hard to retrieve, which is one reason RoPE won.

### NoPE, Partial RoPE and Per-Layer Mixes

Since 2024 the trend has been to apply RoPE *less*, not more:

- **NoPE layers.** A causal decoder can infer position without any encoding (the causal mask itself breaks symmetry). Llama 4 interleaves one NoPE layer after every three RoPE layers ("iRoPE"), and SmolLM3 drops RoPE in every fourth layer; the NoPE layers carry long-range retrieval while the RoPE layers carry local order.
- **Partial RoPE.** Rotate only part of each head and leave the rest position-free. GLM-4 rotates half of each head's dimensions; DeepSeek's MLA attaches a separate 64-dimensional rotary component to a position-free compressed key ([Chapter 5](./05_attention_gqa_mqa_mla.md)).
- **Different bases per layer.** Gemma 3 uses base 10K in its sliding-window layers and 1M in its global layers: local layers need sharp local order, global layers need slowly varying angles.

## Positional Encoding Comparison

| Method | Added Where | Relative Position | Length Extrapolation | Used By |
|--------|------------|-------------------|---------------------|---------|
| **Sinusoidal** | Embedding output | Implicit (linear in offset) | Moderate | Original Transformer |
| **Learned** | Embedding output | No | None (fixed max_len) | GPT-2, BERT |
| **RoPE** | Q, K in attention | Yes (rotation angle) | Good with base scaling / YaRN | Llama, Mistral, Qwen, Gemma, DeepSeek, gpt-oss |
| **M-RoPE** | Q, K in attention | Yes (per axis) | Inherits from RoPE | Qwen2-VL, Qwen2.5-VL, Qwen3-VL |
| **ALiBi** | Attention scores | Yes (linear bias) | Strong on loss, weak on retrieval | BLOOM, MPT |
| **NoPE (in some layers)** | Nowhere | Learned implicitly | Good when mixed with RoPE layers | Llama 4 (iRoPE), SmolLM3 |

RoPE (M-RoPE for multimodal) remains the default because relative position falls out of the geometry and its frequencies can be rescaled for longer contexts. The 2025–26 refinements — larger bases, YaRN, partial rotation and RoPE-free layers — all tune *how much* of the model sees position, and where.

## Tying Embeddings

Many models **tie** the input embedding matrix with the output projection (the "LM head"). This means the matrix used to look up token embeddings is the same matrix used to project hidden states back to vocabulary logits:

$$\text{logits} = h \cdot E^T$$

Where $h$ is the final hidden state `[B, T, d_model]` and $E^T$ is the transposed embedding matrix `[d_model, V]`.

This halves the number of embedding parameters and enforces consistency between input and output spaces. Whether to tie is a size decision in practice: Llama 3.2 1B/3B, Qwen3 up to 4B and all Gemma models tie; Llama 3.1 8B/70B, Qwen3 8B+ and DeepSeek-V3 do not, because at large d_model the table is a small fraction of parameters and separate matrices train slightly better.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>If a causal decoder can learn position from the mask alone, what exactly is RoPE buying in the layers that keep it — and why might the answer be different for a 1K-token chat turn and a 1M-token codebase?</p>
</div>

## Key Takeaways

- The embedding table is a `[V, d_model]` lookup; its size is negligible in big models and dominant in tiny ones, which is why tying is a size-dependent choice.
- Embeddings start static; the transformer layers make them contextual.
- RoPE rotates Q and K by position so attention scores depend only on relative distance; high-frequency pairs encode local order, low-frequency pairs long range.
- Context extension (larger base, interpolation, NTK-by-parts, YaRN) is about keeping rotation angles inside the range seen in training — and still needs some long-sequence training.
- M-RoPE splits the rotation pairs between time, height and width; 2025–26 designs also mix RoPE with NoPE layers and partial rotation.

## Check Yourself

<details class="check">
<summary>With base 10,000 and d_head = 128, roughly how many positions does the slowest RoPE pair take to complete one revolution?</summary>
<p>θ<sub>63</sub> = 10000<sup>−126/128</sup> ≈ 1.15 × 10<sup>−4</sup> rad per position, so one revolution takes 2π / 1.15 × 10<sup>−4</sup> ≈ 55,000 positions. A model trained on 8K sequences has seen that pair cover only about 15% of a turn.</p>
</details>

<details class="check">
<summary>Why does RoPE make q·k depend only on the distance between positions?</summary>
<p>Each pair of q at position m is rotated by mθ and each pair of k at position n by nθ. A dot product of two rotated 2D vectors equals the dot product of the originals with one rotated by the difference of their angles, (m − n)θ. Absolute positions cancel.</p>
</details>

<details class="check">
<summary>Llama 3.1 extends context 8K → 128K. Why does it leave the high-frequency RoPE pairs alone?</summary>
<p>High-frequency pairs already wrap around many times within 8K tokens, so the model has seen every angle they can take; long sequences do not push them out of distribution. Only the slow pairs reach unseen angles, so only they are interpolated — which also preserves the model's sharp sense of local order.</p>
</details>

<details class="check">
<summary>In Qwen2-VL, which positions does a 2×2 grid of image patches that starts at sequence position 10 receive, and where does the following text continue?</summary>
<p>All four patches get t = 10; (h, w) = (10, 10), (10, 11), (11, 10), (11, 11). The largest index used is 11, so the next text token gets t = h = w = 12.</p>
</details>

<details class="check">
<summary>A ported model produces fluent but jumbled output, and the only change was a new attention kernel. What RoPE bug is a prime suspect?</summary>
<p>A pairing-convention mismatch: the checkpoint expects rotate-half (NeoX) pairing and the kernel applies interleaved (GPT-J) pairing or vice versa, so positions are encoded on the wrong dimension pairs.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Efficient Estimation of Word Representations in Vector Space | Mikolov et al., 2013 | [arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) | word2vec |
| Attention Is All You Need | Vaswani et al., 2017 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) | Sinusoidal positional encoding |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al., 2021 | [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864) | RoPE |
| Train Short, Test Long (ALiBi) | Press et al., 2022 | [arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409) | ALiBi |
| Extending Context Window of LLMs via Positional Interpolation | Chen et al., 2023 | [arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595) | Position Interpolation |
| YaRN: Efficient Context Window Extension of LLMs | Peng et al., 2023 | [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071) | NTK-by-parts and attention temperature |
| The Impact of Positional Encoding on Length Generalization in Transformers | Kazemnejad et al., 2023 | [arxiv.org/abs/2305.19466](https://arxiv.org/abs/2305.19466) | NoPE in causal decoders |
| Qwen2-VL | Wang et al., 2024 | [arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191) | M-RoPE |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | RoPE base 500K, 128K extension |
| Gemma 3 Technical Report | Gemma Team, 2025 | [arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786) | Local/global layers with different RoPE bases |

---

**Next:** [Chapter 3 — The Transformer](./03_the_transformer.md) — the block that turns these position-aware vectors into contextual representations.

[← Back to Table of Contents](./README.md)
