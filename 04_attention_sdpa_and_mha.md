---
title: "Chapter 4 — Attention Deep Dive: SDPA & Multi-Head Attention"
description: "Scaled dot-product attention with shapes, why 1/√d_k, causal masks (and PyTorch's inverted boolean convention), multi-head attention, FlashAttention 1–4, attention sinks and gated attention, and what heads learn."
---

[← Back to Table of Contents](./README.md)

# Chapter 4 — Attention Deep Dive: SDPA & Multi-Head Attention

> **The one-sentence version:** attention lets every token build its new representation as a weighted average of every earlier token's *value* vector, with the weights set by how well its *query* matches each of their *keys*.

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
    <span class="ts-label">Q (queries)</span>
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
    <span class="ts-label">K (keys)</span>
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
    <span class="ts-label">QK^T (scores)</span>
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
    <span class="ts-note">← This is O(T²) in memory!</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">V (values)</span>
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
    <span class="ts-label">Output</span>
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

If $q_i, k_i \sim \mathcal{N}(0, 1)$ independently, then $\text{Var}(q \cdot k) = d_k$. After scaling: $\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1$.

The scale is a convention, not a law: Gemma sets its own `query_pre_attn_scalar`, DeepSeek multiplies MLA's softmax scale by a YaRN correction at long context, and QK-norm (Chapter 3) bounds the logits more directly.

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

Below, a seven-token sentence with small hand-built query and key vectors. Pick a query token and watch its softmax row; raise the temperature (divide by more than √d_k) to flatten it, lower it to sharpen it, and toggle the causal mask.

<div class="lab" data-lab="attention"><noscript>Interactive lab: an attention heatmap for a short sentence with temperature and causal-mask controls. Enable JavaScript to use it.</noscript></div>

## Causal Masking

In **decoder-only** models (GPT, Llama, Qwen, DeepSeek), each token can only attend to itself and previous tokens — it must not see the future. This is enforced with a **causal mask**: an upper-triangular matrix of `−∞` values that, after softmax, zero out attention to future positions.

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

<div class="callout warn">
<span class="callout-title">Boolean masks mean opposite things</span>
<p>In the function above, <code>True</code> means "masked out". In PyTorch's <code>F.scaled_dot_product_attention</code>, a boolean <code>attn_mask</code> uses the opposite convention: <code>True</code> means "this position <em>may</em> be attended to". Passing one convention's mask to the other API attends only to the future. Prefer <code>is_causal=True</code> when you just need a causal mask.</p>
</div>

Training packs several documents into one sequence. A plain causal mask would let document 2 attend to document 1, so modern pipelines use a **block-diagonal causal mask** (document masking) built from the sequence boundaries — FlashAttention's variable-length API and PyTorch's FlexAttention both express it without materialising a T×T matrix.

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

For Llama 3.1 8B: d_model = 4096, H = 32 heads, d_k = 4096/32 = **128 per head**. The equality d_k = d_model / H is a convention, not a requirement: Gemma 3 27B has d_model = 5,376 but 32 heads of 128 (4,096 wide), and gpt-oss has d_model = 2,880 with 64 heads of 64. W_Q maps d_model to H·d_k and W_O maps back. Llama 3 also uses fewer key/value heads than query heads — the subject of [Chapter 5](./05_attention_gqa_mqa_mla.md).

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
      <li>Used in encoder-decoder models (T5, BART, Whisper)</li>
      <li>Also used to inject images into some decoders (Llama 3.2 Vision)</li>
    </ul>
  </div>
</div>
</div>

In self-attention, the query, key, and value all come from the same input. In cross-attention, Q comes from one sequence (the decoder) while K and V come from another (encoder output, or image features). Text-only decoder models use only self-attention.

## Flash Attention

The attention score matrix `[B, H, T, T]` is **O(T²)** in memory. For T = 128K tokens with 32 heads in FP16, that's 131,072² × 32 × 2 bytes ≈ **1.1 TB per layer for a single sequence** — impossible to materialise. **FlashAttention** (Dao et al., 2022) never builds it:

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
    <div class="card-desc">Processes T×T in tiles within SRAM. O(T) memory. Single fused kernel. Same FLOPs, far fewer bytes moved.</div>
  </div>
</div>
</div>

FlashAttention is *exact* — it computes the same result as the naive formula, still with O(T²) FLOPs. What it removes is the O(T²) **memory traffic**, which was the real bottleneck ([Chapter 25](./25_cuda_and_kernel_development.md) builds the kernel). Each generation tracks a GPU generation:

| Version | Year | Target | What changed |
|---|---|---|---|
| FlashAttention | 2022 | A100 | Tiling + online softmax + recomputation in backward |
| FlashAttention-2 | 2023 | A100/H100 | Better work partitioning across warps and sequence; ~2× FA1 |
| FlashAttention-3 | 2024 | H100 | Asynchronous TMA loads and warp specialisation, FP8; ~75% of H100 peak in BF16 |
| FlashAttention-4 | 2025 | B200 | Rewritten for Blackwell's tensor memory and new softmax tricks (software exp emulation, conditional rescaling) |

Around them sit **FlashInfer** (serving-oriented kernels for paged KV caches), **FlexAttention** (PyTorch: write a mask or score modification in Python and get a fused kernel) and cuDNN's attention.

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
| **Previous-token / positional** | Attend to a fixed offset | Each token looks one step back |
| **Syntactic** | Grammatical dependencies | "The cats … → are" (subject–verb agreement) |
| **Copying / retrieval** | Attend to an earlier identical or matching token | Repeating a name, quoting a variable |
| **Induction** | [A][B] … [A] → attend to [B] | The circuit behind much in-context learning |
| **Sink / no-op** | Dump almost all weight on the first token | What a head does when it has nothing to say |

Head specialisation is uneven — many heads can be pruned with little loss — which is part of why sharing keys and values across heads (GQA, [Chapter 5](./05_attention_gqa_mqa_mla.md)) costs so little quality.

## Attention Sinks and Gated Attention

Softmax weights must sum to 1, so a head with nothing relevant to read still has to put its weight *somewhere*. Trained models learn to park it on the first token, which becomes an **attention sink**: its key attracts huge weights while its value contributes almost nothing. Two consequences matter in practice:

- **Streaming and eviction.** If you evict the first tokens from a long KV cache, the sink disappears and quality collapses. StreamingLLM keeps the first few "sink" tokens plus a recent window and stays stable for millions of tokens ([Chapter 18](./18_kv_cache_optimization.md)).
- **Quantization.** Sinks come with massive activations in a few channels, one of the main sources of the outliers that make INT8/INT4 activation quantization hard ([Chapter 20](./20_quantization_fundamentals.md)).

Recent architectures give heads an explicit way to say "nothing here":

| Mechanism | How it works | Used by |
|---|---|---|
| **Learned sink logit** | Add one extra learned logit per head to the softmax denominator, so total weight on real tokens can be < 1 | gpt-oss |
| **Output gating** | Multiply each head's output by a sigmoid gate computed from the query token; a closed gate replaces the sink | Qwen3-Next, Qwen3.5 |
| **Off-by-one softmax** | Add 1 to the denominator (a fixed zero-logit sink) | Proposed 2023; similar in spirit |

Gated attention removes most of the sink behaviour and the giant activations that came with it, which also makes these models friendlier to low-bit quantization.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Attention sinks look like a bug — wasted weight on a meaningless token — yet every large model trained with softmax attention grows them. What does that tell you about what a head <em>needs</em> that softmax does not provide, and why might the fix change how well a model quantizes?</p>
</div>

## Key Takeaways

- SDPA = softmax(QKᵀ/√d_k)·V; the score tensor is `[B, H, T, T]`, quadratic in sequence length.
- Scaling by √d_k keeps logits at unit variance; causal masks hide the future, and packed training uses block-diagonal document masks.
- Multi-head attention splits the work into H heads of width d_k; d_k·H need not equal d_model.
- FlashAttention is exact attention with O(T) memory; versions 1–4 track A100 → H100 → B200.
- Attention sinks are softmax's way of doing nothing; learned sink logits and output gating (gpt-oss, Qwen3-Next/3.5) give heads an explicit "off" switch.

## Check Yourself

<details class="check">
<summary>For B = 1, H = 32, T = 32,768 in BF16, how large is one layer's attention-score tensor if materialised?</summary>
<p>32 × 32,768² × 2 bytes = 68.7 GB — for one layer of one sequence, which is why FlashAttention-style tiling is mandatory at long context.</p>
</details>

<details class="check">
<summary>Why does scaling by 1/√d_k rather than 1/d_k keep softmax well-behaved?</summary>
<p>For independent unit-variance entries the dot product has variance d_k, so standard deviation √d_k. Dividing by √d_k restores unit variance; dividing by d_k would shrink logits towards zero and make every attention row nearly uniform.</p>
</details>

<details class="check">
<summary>You pass <code>mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)</code> as <code>attn_mask</code> to <code>F.scaled_dot_product_attention</code>. What happens?</summary>
<p>PyTorch reads True as "allowed", so each token attends only to <em>future</em> positions (and not itself) — the first token sees everything after it and the last sees nothing. Use <code>is_causal=True</code> or invert the mask.</p>
</details>

<details class="check">
<summary>FlashAttention does the same number of FLOPs as naive attention. Where does its 2–4× speed-up come from?</summary>
<p>Memory traffic. Naive attention writes the T×T score and probability matrices to HBM and reads them back; FlashAttention keeps tiles in on-chip SRAM and fuses the matmuls and softmax, so attention becomes compute-bound instead of bandwidth-bound.</p>
</details>

<details class="check">
<summary>Why does evicting the oldest tokens from a KV cache break a model even when those tokens are irrelevant to the text?</summary>
<p>The first tokens act as attention sinks — many heads route their "unused" weight there. Remove them and that weight is redistributed over real tokens, corrupting every head's output. Keeping a few initial sink tokens (StreamingLLM) fixes it.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau et al., 2014 | [arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473) | The origin of attention |
| Attention Is All You Need | Vaswani et al., 2017 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) | SDPA, MHA |
| FlashAttention | Dao et al., 2022 | [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) | IO-aware exact attention |
| FlashAttention-2 | Dao, 2023 | [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691) | Work partitioning |
| FlashAttention-3 | Shah et al., 2024 | [arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608) | Hopper asynchrony and FP8 |
| In-context Learning and Induction Heads | Olsson et al. (Anthropic), 2022 | [transformer-circuits.pub/2022/in-context-learning-and-induction-heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Induction heads |
| Efficient Streaming Language Models with Attention Sinks | Xiao et al., 2023 | [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453) | Attention sinks, StreamingLLM |
| Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free | Qiu et al. (Qwen), 2025 | [arxiv.org/abs/2505.06708](https://arxiv.org/abs/2505.06708) | Output gating removes sinks |
| gpt-oss-120b & gpt-oss-20b Model Card | OpenAI, 2025 | [openai.com/index/gpt-oss-model-card](https://openai.com/index/gpt-oss-model-card/) | Learned attention-sink logits |
| PyTorch `scaled_dot_product_attention` docs | PyTorch, 2025 | [docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Boolean mask convention, backends |

---

**Next:** [Chapter 5 — GQA, MQA & MLA](./05_attention_gqa_mqa_mla.md) — sharing or compressing keys and values so the KV cache stops dominating inference memory.

[← Back to Table of Contents](./README.md)
