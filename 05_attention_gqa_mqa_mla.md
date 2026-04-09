---
title: "Chapter 5 — Attention Variants: GQA, MQA & MLA"
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — Attention Variants: GQA, MQA & MLA

> *"The memory bottleneck in LLM inference isn't compute — it's the KV-cache. Different attention variants offer different tradeoffs between quality, memory, and throughput."*

## The KV Bottleneck

In standard Multi-Head Attention (MHA), every head has its own Q, K, and V projections. During autoregressive generation, we cache K and V for every head at every layer for every past token. This **KV-cache** grows linearly with sequence length and is often the dominant memory cost at inference time (see [Chapter 17](./17_kv_cache_mechanics.md) for the full deep dive).

For LLaMA-3-8B (MHA, 32 heads, d_k=128, 32 layers, FP16):

$$\text{KV-cache} = 2 \times 32 \times 32 \times 128 \times T \times 2\text{B} = 524{,}288 \times T \text{ bytes}$$

At T = 8192 tokens: **~4 GB per request**, just for KV-cache.

The key insight: **not all heads need their own K and V.** Query heads are diverse and important, but key and value heads can be shared without much quality loss.

## Multi-Query Attention (MQA)

**MQA** (Shazeer, 2019) is the most aggressive sharing strategy: all query heads share a **single** K and V projection.

<div class="diagram">
<div class="diagram-title">MQA — Single KV Head Shared Across All Query Heads</div>
<div class="flow-h">
  <div class="flow-node accent">Q head 1</div>
  <div class="flow-node accent">Q head 2</div>
  <div class="flow-node accent">Q head 3</div>
  <div class="flow-node accent">…</div>
  <div class="flow-node accent">Q head H</div>
</div>
<div style="text-align: center; padding: 0.5rem 0; color: var(--text-muted);">↓ all attend using ↓</div>
<div class="flow-h">
  <div class="flow-node green" style="min-width: 300px;">K head 1 (shared) &nbsp; V head 1 (shared)</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">MQA Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 120px;">Q</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">H independent query heads</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 120px;">K, V</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">1</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">1 shared KV head → broadcast to H</span>
  </div>
</div>
</div>

KV-cache memory reduction: **H× smaller** (32× for 32-head models).

Trade-off: slightly lower quality than MHA, especially on harder reasoning tasks. Used in **PaLM, Falcon, StarCoder**.

## Grouped-Query Attention (GQA)

**GQA** (Ainslie et al., 2023) is the compromise: query heads are divided into G **groups**, and each group shares one K and V head. When G = 1, GQA = MQA. When G = H, GQA = MHA.

<div class="diagram">
<div class="diagram-title">GQA — Groups of Query Heads Share KV Heads</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Group 1</div>
    <div class="card-desc">Q heads 1–4 share KV head 1</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Group 2</div>
    <div class="card-desc">Q heads 5–8 share KV head 2</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Group 3</div>
    <div class="card-desc">Q heads 9–12 share KV head 3</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Group 4</div>
    <div class="card-desc">Q heads 13–16 share KV head 4</div>
  </div>
</div>
<div style="text-align: center; padding: 0.25rem 0; color: var(--text-muted); font-size: 0.75rem;">Example: H=16 query heads, G=4 KV heads → 4 queries per KV head</div>
</div>

<div class="diagram">
<div class="diagram-title">GQA Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 120px;">Q</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">H = 32 query heads</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 120px;">K, V</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">G = 8 KV heads → each shared by H/G = 4 queries</span>
  </div>
</div>
</div>

### GQA in Practice

| Model | Query Heads (H) | KV Heads (G) | Queries per KV | KV Reduction |
|-------|:---:|:---:|:---:|:---:|
| LLaMA-2-7B | 32 | 32 | 1 | 1× (MHA) |
| LLaMA-2-70B | 64 | 8 | 8 | 8× |
| LLaMA-3-8B | 32 | 8 | 4 | 4× |
| LLaMA-3-70B | 64 | 8 | 8 | 8× |
| Mistral 7B | 32 | 8 | 4 | 4× |
| Gemma 2 9B | 16 | 8 | 2 | 2× |
| Qwen 2.5 7B | 28 | 4 | 7 | 7× |

### GQA Implementation

The key change from MHA is in the projection sizes and a `repeat_interleave` or `expand` to broadcast KV heads to match Q heads:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # queries per KV head
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # smaller!
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # smaller!
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)     # [B, H, T, d_k]
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)  # [B, G, T, d_k]
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)  # [B, G, T, d_k]

        # Expand KV heads to match Q heads: [B, G, T, d_k] → [B, H, T, d_k]
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # Standard SDPA
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # [B, H, T, d_k]

        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

**Parameter savings**: W_k and W_v go from `[d_model, d_model]` to `[d_model, G × d_k]`. For LLaMA-3-8B (d=4096, H=32, G=8): each KV projection drops from 4096×4096 to 4096×1024 — **4× smaller**.

## Multi-Latent Attention (MLA)

**MLA** (DeepSeek-V2, 2024) takes a fundamentally different approach: instead of reducing the number of KV heads, it **compresses** the KV representations into a low-rank latent space.

<div class="diagram">
<div class="diagram-title">MLA — Low-Rank Compression of KV</div>
<div class="flow">
  <div class="flow-node accent wide">Input x: [B, T, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Down-project: W_dkv <small>[d_model, d_c] → compressed KV: [B, T, d_c]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Cache only compressed: [B, T, d_c] <small>d_c ≪ d_model (e.g., 512 vs 4096)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Up-project at decode time: [B, T, d_c] → K, V: [B, H, T, d_k]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Standard SDPA with reconstructed K, V</div>
</div>
</div>

The key advantage: at inference time, you only cache the **compressed** latent `[B, T, d_c]` instead of full K and V tensors. If d_c = 512 and d_model = 4096, this is an **8× compression** of the KV-cache — even better than GQA in some configs.

DeepSeek-V2 also combines this with a **decoupled RoPE** where a separate small projection handles positional encoding, allowing the main KV compression to be position-independent.

## Sliding Window Attention

**Sliding Window Attention** (Mistral, 2023) limits each token's attention to a local window of W previous tokens instead of the full sequence. Outside the window, attention scores are masked to −∞.

<div class="diagram">
<div class="diagram-title">Sliding Window vs Full Causal Attention (W=3)</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Full Causal (T=5)</div>
    <ul>
      <li>✓ · · · ·</li>
      <li>✓ ✓ · · ·</li>
      <li>✓ ✓ ✓ · ·</li>
      <li>✓ ✓ ✓ ✓ ·</li>
      <li>✓ ✓ ✓ ✓ ✓</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Sliding Window (W=3)</div>
    <ul>
      <li>✓ · · · ·</li>
      <li>✓ ✓ · · ·</li>
      <li>✓ ✓ ✓ · ·</li>
      <li>· ✓ ✓ ✓ ·</li>
      <li>· · ✓ ✓ ✓</li>
    </ul>
  </div>
</div>
</div>

KV-cache becomes fixed at W entries per head instead of growing with T. Information beyond the window can still flow through stacked layers (layer L attends to W, layer L+1's window sees those representations, effectively reaching 2W tokens back).

Mistral 7B uses W = 4096 with alternating sliding-window and full-attention layers.

## Comparison Summary

| Variant | KV per Head | KV Heads | Cache Size | Quality | Models |
|---------|:-----------:|:--------:|:----------:|:-------:|--------|
| **MHA** | Full | H | `2 × L × H × d_k × T` | Best | GPT, LLaMA-1 |
| **MQA** | Full | 1 | `2 × L × 1 × d_k × T` | Lowest | PaLM, Falcon |
| **GQA** | Full | G | `2 × L × G × d_k × T` | Near-MHA | LLaMA-2/3, Mistral |
| **MLA** | Compressed | — | `L × d_c × T` | Near-MHA | DeepSeek-V2/V3 |
| **SWA** | Full | H/G | `2 × L × H × d_k × W` | Good (local) | Mistral, Gemma |

<div class="diagram">
<div class="diagram-title">KV-Cache Memory vs Sequence Length</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">MHA (H=32)</div>
    <div class="card-desc">Linear growth: 512 KB/token for LLaMA-3-8B scale. 4 GB at 8K tokens.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">GQA (G=8)</div>
    <div class="card-desc">4× smaller: 128 KB/token. 1 GB at 8K tokens. Near-MHA quality.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">MQA (G=1)</div>
    <div class="card-desc">32× smaller: 16 KB/token. 128 MB at 8K. Some quality tradeoff.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">MLA</div>
    <div class="card-desc">Compressed latent: much smaller than GQA with comparable quality. DeepSeek-specific.</div>
  </div>
</div>
</div>

## What's Next

Now that we understand the attention mechanisms inside transformer blocks, we'll see how these components assemble into complete **decoder-only models** — the dominant architecture for modern LLMs.

[← Previous: Chapter 4 — SDPA & Multi-Head Attention](./04_attention_sdpa_and_mha.md) · **Next: [Chapter 6 — Decoder-Only Models →](./06_decoder_only_models.md)**

---

*Last updated: April 2026*
