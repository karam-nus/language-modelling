---
title: "Chapter 5 — Attention Variants: GQA, MQA & MLA"
description: "Why the KV cache is the bottleneck and how attention variants shrink it: MQA, GQA, DeepSeek's multi-head latent attention with weight absorption, sliding windows, and the 2025–26 wave of sparse and hybrid attention."
related:
  - { id: models, why: "Which attention variant each model generation chose." }
  - { id: inference, why: "What these KV-cache savings mean on real inference systems." }
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — Attention Variants: GQA, MQA & MLA

> **The one-sentence version:** every attention variant in this chapter is a different answer to one question — how few bytes per token can the KV cache hold before quality suffers?

## The KV Bottleneck

In standard Multi-Head Attention (MHA), every head has its own Q, K, and V projections. During autoregressive generation, we cache K and V for every head at every layer for every past token. This **KV-cache** grows linearly with sequence length and is often the dominant memory cost at inference time (see [Chapter 17](./17_kv_cache_mechanics.md) for the full deep dive).

Take Llama 3.1 8B's shape (32 layers, 32 heads, d_k = 128, BF16) and imagine it used plain MHA:

$$\text{KV-cache} = \underbrace{2}_{K,V} \times \underbrace{32}_{\text{layers}} \times \underbrace{32}_{\text{heads}} \times \underbrace{128}_{d_k} \times \underbrace{2\,\text{B}}_{\text{BF16}} \times T = 524{,}288 \times T \text{ bytes}$$

That is 512 KiB per token, so **4.3 GB per 8K-token request** — and 69 GB at 128K, more than the model's 16 GB of weights. The real Llama 3.1 8B uses 8 KV heads and needs a quarter of that.

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
<div class="flow-caption">↓ all attend using ↓</div>
<div class="flow-h">
  <div class="flow-node green wide">K head 1 (shared) &nbsp; V head 1 (shared)</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">MQA Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Q</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">H independent query heads</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">K, V</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">1</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">1 shared KV head → broadcast to H</span>
  </div>
</div>
</div>

KV-cache memory reduction: **H× smaller** (32× for 32-head models).

Trade-off: measurably lower quality than MHA, and training can be less stable. Used in **PaLM, Falcon-7B, StarCoder**; mostly superseded by GQA.

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
<div class="flow-caption">Example: H=16 query heads, G=4 KV heads → 4 queries per KV head</div>
</div>

<div class="diagram">
<div class="diagram-title">GQA Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Q</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">H = 32 query heads</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">K, V</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">G = 8 KV heads → each shared by H/G = 4 queries</span>
  </div>
</div>
</div>

### GQA in Practice

| Model | Query Heads (H) | KV Heads (G) | Queries per KV | KV Reduction |
|-------|:---:|:---:|:---:|:---:|
| Llama 2 7B | 32 | 32 | 1 | 1× (MHA) |
| Llama 2 70B | 64 | 8 | 8 | 8× |
| Llama 3.1 8B | 32 | 8 | 4 | 4× |
| Llama 3.1 70B / 405B | 64 / 128 | 8 | 8 / 16 | 8× / 16× |
| Mistral 7B | 32 | 8 | 4 | 4× |
| Gemma 3 27B | 32 | 16 | 2 | 2× |
| Qwen2.5-7B | 28 | 4 | 7 | 7× |
| Qwen3-8B | 32 | 8 | 4 | 4× |
| Qwen3-235B-A22B | 64 | 4 | 16 | 16× |
| gpt-oss-120b | 64 | 8 | 8 | 8× |

Eight KV heads became a near-universal default because it is the tensor-parallel degree of a typical 8-GPU node: with 8 KV heads, each GPU holds exactly one.

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
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)     # [B, H, T, d_k]
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)  # [B, G, T, d_k]
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)  # [B, G, T, d_k]

        # Expand KV heads to match Q heads: [B, G, T, d_k] → [B, H, T, d_k]
        K = K.repeat_interleave(self.n_rep, dim=1)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # Standard SDPA (PyTorch ≥ 2.5 can skip the copy: pass enable_gqa=True with G-head K, V)
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # [B, H, T, d_k]

        out = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)
```

**Parameter savings**: W_k and W_v go from `[d_model, d_model]` to `[d_model, G × d_k]`. For Llama 3.1 8B (d=4096, H=32, G=8): each KV projection drops from 4096×4096 to 4096×1024 — **4× smaller**. The repeat is only a view in good kernels: FlashAttention and FlashInfer read each KV head once and reuse it for its whole query group, which is exactly what makes GQA decode faster as well as smaller.

## Multi-Latent Attention (MLA)

**MLA** — *multi-head latent attention* (DeepSeek-V2, 2024) — takes a different route: instead of having fewer KV heads, it **compresses** keys and values for all heads into one low-rank latent vector per token, and caches only that.

<div class="diagram">
<div class="diagram-title">MLA — Low-Rank Compression of KV</div>
<div class="flow">
  <div class="flow-node accent wide">Input x: [B, T, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Down-project: W_dkv <small>[d_model, d_c] → compressed KV: [B, T, d_c]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Cache only compressed: [B, T, d_c] <small>DeepSeek-V3: d_c = 512 (+ a 64-dim RoPE key) vs d_model = 7,168</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Up-project at decode time: [B, T, d_c] → K, V: [B, H, T, d_k]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">Standard SDPA with reconstructed K, V</div>
</div>
</div>

At inference you cache only the latent instead of full per-head K and V. The numbers for DeepSeek-V3 (61 layers, 128 heads, head dim 128):

| | Per token per layer | Per token, all layers (BF16) |
|---|---:|---:|
| MHA equivalent (128 K heads of 192 + 128 V heads of 128) | 40,960 values | 5.0 MB |
| GQA with 8 KV heads of 128 | 2,048 values | 250 KB |
| **MLA: latent 512 + RoPE key 64** | **576 values** | **70 KB** |

MLA caches about **70× less than MHA** — as little as GQA with just 2.25 KV heads — while the DeepSeek-V2 paper reports quality *better* than its MHA baseline.

Two details make it work:

1. **Decoupled RoPE.** Rotating keys by position would break the low-rank trick (the rotation sits between the latent and the head). So the latent stays position-free and a separate small 64-dim key, shared across heads, carries RoPE.
2. **Weight absorption.** At decode time the up-projection for keys can be multiplied into the query projection, and the up-projection for values into the output projection. Attention then runs *directly on the cached latents*: every head reads the same 576-wide vector, like MQA with one very wide head. This raises arithmetic intensity — more FLOPs per byte of cache read — which suits bandwidth-bound decode on modern GPUs ([Chapter 24](./24_gpu_architecture.md)).

MLA spread beyond DeepSeek in 2025: Kimi K2 uses it, and conversion recipes (TransMLA, MHA2MLA) retrofit it onto GQA checkpoints with short fine-tunes.

## Sliding Window Attention

**Sliding Window Attention** (Longformer, 2020; popularised by Mistral 7B, 2023) limits each token's attention to a local window of W previous tokens instead of the full sequence. Outside the window, attention scores are masked to −∞.

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

KV-cache becomes fixed at W entries per head instead of growing with T. Information beyond the window can still flow through stacked layers: layer 2's window covers tokens that themselves summarised a window, so the receptive field after L layers is up to L × W.

Pure SWA (Mistral 7B v0.1 used W = 4,096 in every layer) struggles with precise long-range retrieval, so 2024–26 models **interleave** local and global layers:

| Model | Local : global layers | Window |
|---|---|---|
| Gemma 2 | 1 : 1 | 4,096 |
| Gemma 3 | 5 : 1 | 1,024 |
| gpt-oss | 1 : 1 | 128 |
| Command A (Cohere) | 3 : 1 | 4,096 |

Gemma 3's 5:1 mix cuts its KV cache at 32K context by roughly 5× versus all-global attention, with negligible perplexity cost.

## Comparison Summary

<div class="diagram">
<div class="diagram-title">Who Reads Which Keys and Values</div>
<div class="fig" data-fig="heads" data-variant="all"><noscript>Figure: query heads and the key/value heads they read under MHA (one each), GQA (groups share), MQA (all share one) and MLA (a cached latent expanded per head), with KV-cache bytes per token per layer for a 32-head, 128-dim layer in BF16: 16 KiB, 4 KiB, 512 B and 1,152 B.</noscript></div>
<div class="flow-caption">Drawn with 8 of 32 query heads. Cache sizes are per token per layer in BF16 for a 32-head, d_head = 128 layer (GQA with 8 KV heads; MLA with DeepSeek's 512 + 64 dims).</div>
</div>

<div class="img-caption">
  <img src="{{ '/assets/images/gqa_mha_mqa.svg' | relative_url }}" alt="Side-by-side comparison of MHA, GQA, and MQA showing Q, K, V head counts and KV cache sizes">
  <figcaption>Multi-Head Attention vs Grouped-Query Attention vs Multi-Query Attention — head configurations and cache implications</figcaption>
</div>

| Variant | KV per Head | KV Heads | Cache Size | Quality | Models |
|---------|:-----------:|:--------:|:----------:|:-------:|--------|
| **MHA** | Full | H | `2 × L × H × d_k × T` | Baseline | GPT-2/3, Llama 1, OLMo 2 7B |
| **MQA** | Full | 1 | `2 × L × 1 × d_k × T` | Lower | PaLM, Falcon-7B |
| **GQA** | Full | G | `2 × L × G × d_k × T` | Near-MHA | Llama 2 70B/3/4, Qwen, Mistral, gpt-oss |
| **MLA** | Compressed latent | — | `L × (d_c + d_R) × T` | ≥ MHA (DeepSeek's ablation) | DeepSeek-V2/V3, Kimi K2 |
| **SWA** | Full | G | `2 × L × G × d_k × min(T, W)` | Good locally | Gemma 2/3 local layers, gpt-oss |

Put real model shapes into the calculator and compare the variants at your own context length:

<div class="lab" data-lab="kvcache"><noscript>Interactive lab: KV-cache size calculator for MHA, GQA, MQA, MLA and sliding-window attention with real model presets. Enable JavaScript to use it.</noscript></div>

## Beyond Dense Attention: Sparse and Hybrid (2025–26)

GQA, MLA and windows shrink the cache, but every global layer still does O(T²) work over the whole context. The 2025–26 frontier attacks that directly:

<div class="diagram">
<div class="diagram-title">The Sparse Patterns, Side by Side</div>
<div class="fig" data-fig="masks" data-set="sliding,chunked,sink,dilated,nsa,topk" data-t="16"><noscript>Figure: 16×16 masks for sliding-window, chunked, sinks-plus-window, dilated, block-sparse (NSA-style) and top-k (DeepSeek DSA-style) attention, with pairs computed and keys read by the last query.</noscript></div>
<div class="flow-caption">Windows and chunks bound the cache; NSA and DSA keep the whole cache but read only a fraction of it per query.</div>
</div>

Change the sequence length, window, chunk size and k yourself:

<div class="lab" data-lab="masks"><noscript>Interactive lab: attention-mask explorer for causal, bidirectional, prefix-LM, sliding-window, chunked, sink, dilated, block-sparse and top-k patterns, with pairs computed and KV cache kept. Enable JavaScript to use it.</noscript></div>

<div class="diagram">
<div class="diagram-title">Attention in 2026 Is a Menu</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Learned sparse attention</div>
    <div class="card-desc">A cheap indexer scores all past tokens; full attention runs only on the top-k. DeepSeek NSA (2025), DeepSeek Sparse Attention in V3.2 (top-2,048 tokens), Kimi's MoBA (block-level selection).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Compressed + sparse</div>
    <div class="card-desc">DeepSeek-V4 (2026) compresses groups of KV entries, selects top-k compressed blocks plus a local window, and alternates with heavily compressed layers — about 10% of V3.2's KV cache at 1M tokens.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Linear-attention hybrids</div>
    <div class="card-desc">Replace most attention layers with fixed-state recurrent layers: Qwen3-Next and Qwen3.5 (3 Gated DeltaNet : 1 gated attention), Kimi Linear (3:1), MiniMax-M1 (7 lightning : 1 softmax). See Chapter 34.</div>
  </div>
</div>
</div>

The common thread: keep *some* exact softmax attention — it is what does precise retrieval — but spend it only where it pays.

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

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>MLA makes decode <em>more</em> compute-intensive per byte of cache read, and DeepSeek designed it on bandwidth-starved hardware. If memory bandwidth grows faster than FLOPs in the next GPU generation — or slower — which of the variants in this chapter wins, and why?</p>
</div>

## Key Takeaways

- KV cache per token = 2 × layers × KV heads × head dim × bytes; with MHA it can exceed the weights at long context.
- MQA shares one KV head; GQA shares G (8 is the de-facto default, matching 8-way tensor parallelism).
- MLA caches a 512-dim latent plus a 64-dim RoPE key per layer — ~70× less than MHA for DeepSeek-V3 — and uses weight absorption to attend directly on latents.
- Sliding windows cap cache size; modern models interleave local and global layers (Gemma 3 5:1, gpt-oss 1:1).
- 2025–26 designs go further with learned sparse selection (NSA, DSA, V4's compressed sparse attention) and linear-attention hybrids.

## Check Yourself

<details class="check">
<summary>Qwen3-235B-A22B has 94 layers, 4 KV heads of dimension 128. What is its BF16 KV cache per token, and at 128K tokens?</summary>
<p>2 × 94 × 4 × 128 × 2 B = 192,512 B ≈ 188 KB per token; × 131,072 ≈ 25 GB per 128K-token sequence.</p>
</details>

<details class="check">
<summary>Why can't MLA simply apply RoPE to its compressed keys?</summary>
<p>Weight absorption relies on the key up-projection being a fixed matrix that can be folded into the query projection. RoPE inserts a position-dependent rotation between the latent and the head, so the product is different for every position and cannot be precomputed. The fix is a small separate RoPE key that bypasses the latent.</p>
</details>

<details class="check">
<summary>A model has 48 layers in a 5:1 local:global pattern with a 1,024-token window. Roughly what fraction of a 64K-token KV cache does it need compared with all-global layers?</summary>
<p>8 global layers hold 64K tokens each; 40 local layers hold 1K each. (8 × 65,536 + 40 × 1,024) / (48 × 65,536) ≈ 0.18, about 5.5× smaller.</p>
</details>

<details class="check">
<summary>GQA with 8 KV heads and 64 query heads is served with 16-way tensor parallelism. What happens to the KV heads?</summary>
<p>There are fewer KV heads than GPUs, so each KV head must be replicated on two GPUs (each serving a different 4 of its 8 query heads). KV memory per GPU stops shrinking past TP = 8 — one reason 8 KV heads pairs naturally with 8-GPU nodes.</p>
</details>

<details class="check">
<summary>Why do sparse-attention designs like DSA still keep a dense "indexer" over all past tokens?</summary>
<p>To choose the top-k tokens you must score every candidate. The indexer is made very cheap (few heads, small dimension, FP8) so its O(T²) cost is a small fraction of full attention, while the expensive attention runs only over k ≪ T selected tokens.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Fast Transformer Decoding: One Write-Head is All You Need | Shazeer, 2019 | [arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150) | MQA |
| GQA: Training Generalized Multi-Query Transformer Models | Ainslie et al., 2023 | [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245) | GQA and uptraining |
| Longformer | Beltagy et al., 2020 | [arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150) | Sliding-window attention |
| Mistral 7B | Jiang et al., 2023 | [arxiv.org/abs/2310.06825](https://arxiv.org/abs/2310.06825) | SWA in an open LLM |
| DeepSeek-V2 | DeepSeek-AI, 2024 | [arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434) | MLA, decoupled RoPE, absorption |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | MLA dimensions |
| Native Sparse Attention | Yuan et al. (DeepSeek), 2025 | [arxiv.org/abs/2502.11089](https://arxiv.org/abs/2502.11089) | Hardware-aligned learned sparsity |
| MoBA: Mixture of Block Attention | Lu et al. (Moonshot), 2025 | [arxiv.org/abs/2502.13189](https://arxiv.org/abs/2502.13189) | Block-sparse attention |
| Gemma 3 Technical Report | Gemma Team, 2025 | [arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786) | 5:1 local/global |
| DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | DeepSeek-AI, 2026 | [huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | Compressed sparse attention, KV savings at 1M |

---

**Next:** [Chapter 6 — Decoder-Only Models](./06_decoder_only_models.md) — how these blocks assemble into the GPT, Llama, Qwen and DeepSeek families.

[← Back to Table of Contents](./README.md)
