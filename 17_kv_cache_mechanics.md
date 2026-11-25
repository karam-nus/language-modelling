---
title: "Chapter 17 — KV-Cache: Mechanics & Memory"
description: "Why the KV cache exists and what it actually saves, its shapes per layer, the memory formula checked on Llama, Qwen3, DeepSeek-V3 and gpt-oss, why decode is bandwidth-bound, how cache size sets batch size and throughput, and how to inspect the cache in transformers."
---

[← Back to Table of Contents](./README.md)

# Chapter 17 — KV-Cache: Mechanics & Memory

> **The one-sentence version:** the KV cache stores every past token's keys and values so each new token costs one forward pass instead of T — at the price of a memory footprint that grows with every token and, at long context, outweighs the model itself.

## Why KV-Cache Exists

During autoregressive generation, each new token must attend to all previous tokens. Without a cache, step t would rerun the *entire model* over all t tokens — every projection, every MLP — so generating T tokens costs O(T²) token-forward-passes.

Keys and values of past tokens never change (the causal mask means later tokens cannot affect earlier ones), so they can be computed once and stored. With a KV cache each decode step runs the model on **one** token: its Q, K, V projections, its MLPs, and an attention read over the cached keys and values. Total work for T tokens drops to O(T) forward passes; only the attention read still grows with context (O(T) per step).

<div class="diagram">
<div class="diagram-title">Without vs With KV-Cache</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Without Cache</div>
    <ul>
      <li>Step 1: compute K, V for tokens [1]</li>
      <li>Step 2: recompute K, V for tokens [1, 2]</li>
      <li>Step 3: recompute K, V for tokens [1, 2, 3]</li>
      <li>Step T: recompute all T tokens</li>
      <li>Total: O(T²) token-passes through every layer</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">With KV-Cache</div>
    <ul>
      <li>Step 1: compute & cache K₁, V₁</li>
      <li>Step 2: compute K₂, V₂, append to cache</li>
      <li>Step 3: compute K₃, V₃, append to cache</li>
      <li>Step T: compute Kₜ, Vₜ, attend to cached [1..T]</li>
      <li>Total: O(T) token-passes ✓ (attention reads still grow with T)</li>
    </ul>
  </div>
</div>
</div>

## Step-by-Step Visualization

<div class="img-caption">
  <img src="{{ '/assets/images/kv_cache_generation.svg' | relative_url }}" alt="KV cache generation showing token-by-token generation with cache reuse across steps, plus memory calculation formula">
  <figcaption>KV cache builds up one token at a time — each new token only runs through the model once, reusing cached K/V</figcaption>
</div>

<div class="diagram">
<div class="diagram-title">KV-Cache Growth During Generation</div>
<div class="flow">
  <div class="flow-node accent wide">Prefill: process prompt [1..T_p] in parallel <small>→ cache K, V: [B, H, T_p, d_k]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Decode step 1: new token T_p+1 <small>→ Q: [B, H, 1, d_k], attend to cache [B, H, T_p, d_k]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Append K_{T_p+1}, V_{T_p+1} <small>→ cache grows to [B, H, T_p+1, d_k]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Decode step 2: new token T_p+2 <small>→ Q: [B, H, 1, d_k], attend to cache [B, H, T_p+1, d_k]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">… continues until EOS or max_length <small>cache grows by [B, H, 1, d_k] per step per layer</small></div>
</div>
</div>

At each decode step, the computation is:
- **Query**: only the new token → `[B, H, 1, d_k]`
- **Key/Value**: read from cache → `[B, H, T_total, d_k]`
- **Attention**: `Q × K^T` → `[B, H, 1, T_total]` → softmax → `× V` → `[B, H, 1, d_k]`

For one sequence these are **matrix-vector products** (one query row against T cached rows), and the projections and MLPs likewise multiply each weight by just one token. Every weight byte and every cache byte is read from HBM to do about one multiply-add — so decode is **memory-bandwidth bound**. A useful ceiling for batch size 1:

$$\text{tokens/s} \lesssim \frac{\text{HBM bandwidth}}{\text{weight bytes} + \text{KV-cache bytes}}$$

An H100 (3.35 TB/s) serving Llama 3.1 8B in BF16 (16 GB) can therefore produce at most ~200 tokens/s for a single short sequence — no matter how many TFLOP/s it has. Batching amortises the weight reads across sequences; the KV reads cannot be shared, because each sequence has its own cache.

## Tensor Shapes in Detail

<div class="diagram">
<div class="diagram-title">KV-Cache Tensor Shapes (Per Layer)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">K cache (MHA)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">V cache (MHA)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim heads">H</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">K cache (GQA, G groups)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">G < H → smaller cache</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Per-step append</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">1</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">one new K, V per layer per step</span>
  </div>
</div>
</div>

## Memory Calculation

The KV-cache memory for the **entire model**:

$$\text{KV memory} = 2 \times L \times G \times d_k \times T \times B \times \text{bytes}$$

where: L = layers, G = KV heads, d_k = head dimension, T = sequence length, B = batch size, bytes = 2 (FP16/BF16).

### Worked Examples

| Model | L | G (KV heads) | d_k | Formula | Per token | At 8K | At 128K |
|-------|:-:|:---:|:---:|---------|:---:|:---:|:---:|
| Llama 2 7B (MHA) | 32 | 32 | 128 | 2×32×32×128×2 | 512 KiB | 4 GiB | 64 GiB |
| Llama 3.1 8B (GQA) | 32 | 8 | 128 | 2×32×8×128×2 | 128 KiB | 1 GiB | 16 GiB |
| Llama 3.1 70B (GQA) | 80 | 8 | 128 | 2×80×8×128×2 | 320 KiB | 2.5 GiB | 40 GiB |
| Llama 3.1 405B (GQA) | 126 | 8 | 128 | 2×126×8×128×2 | 504 KiB | 3.9 GiB | 63 GiB |
| Qwen3-235B-A22B (GQA) | 94 | 4 | 128 | 2×94×4×128×2 | 188 KiB | 1.5 GiB | 23.5 GiB |
| DeepSeek-V3 (MLA) | 61 | — | 512 + 64 | 61×576×2 | 68.6 KiB | 0.54 GiB | 8.6 GiB |
| gpt-oss-120b (half windowed) | 36 (18 full) | 8 | 64 | 2×18×8×64×2 (+ fixed 128-token windows) | 36 KiB | 0.28 GiB | 4.5 GiB |

> **Key insight**: Llama 3.1 405B needs 63 GiB of cache **per 128K-token request** in BF16 — on top of 810 GB of weights. Compare DeepSeek-V3 (MLA) at 8.6 GiB and gpt-oss (small heads, half the layers windowed) at 4.5 GiB: the attention design is the single biggest lever on long-context serving cost. [Chapter 18](./18_kv_cache_optimization.md) covers the rest.

Plug in any model's shape — or the presets — and a context length:

<div class="lab" data-lab="kvcache"><noscript>Interactive lab: KV-cache size calculator with model presets and attention variants. Enable JavaScript to use it.</noscript></div>

## GQA/MQA Impact on KV-Cache

The attention variant directly determines KV-cache size (see [Chapter 5](./05_attention_gqa_mqa_mla.md)):

<div class="diagram">
<div class="diagram-title">KV-Cache Size by Attention Variant (Llama 3.1 8B scale)</div>
<div class="diagram-grid cols-4">
  <div class="diagram-card accent">
    <div class="card-title">MHA (H=32)</div>
    <div class="card-desc">32 KV heads<br>512 KB/token<br>4 GB at 8K</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">GQA (G=8)</div>
    <div class="card-desc">8 KV heads<br>128 KB/token<br>1 GB at 8K<br><strong>4× savings</strong></div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">GQA (G=4)</div>
    <div class="card-desc">4 KV heads<br>64 KB/token<br>512 MB at 8K<br><strong>8× savings</strong></div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">MQA (G=1)</div>
    <div class="card-desc">1 KV head<br>16 KB/token<br>128 MB at 8K<br><strong>32× savings</strong></div>
  </div>
</div>
</div>

## Sliding Window KV-Cache

With Sliding Window Attention ([Chapter 5](./05_attention_gqa_mqa_mla.md)), the KV-cache is a fixed-size rolling buffer:

<div class="diagram">
<div class="diagram-title">Sliding Window Cache (W=4096)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Full KV-cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">grows with T — unbounded</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Sliding window cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">W</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">fixed at W — evicts oldest entry</span>
  </div>
</div>
</div>

## Inspecting KV-Cache in Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(model_id)
ids = tok("Hello, world!", return_tensors="pt").input_ids.to(model.device)   # [1, 5] incl. BOS

out = model(ids, use_cache=True)                    # prefill
cache = out.past_key_values                         # a DynamicCache object
layer0 = cache.layers[0]                            # recent transformers: one entry per layer
print(layer0.keys.shape, layer0.values.shape)       # [1, 8, 5, 128] each: 8 KV heads, 5 tokens
print(cache.get_seq_length())                       # 5

nbytes = sum(l.keys.nbytes + l.values.nbytes for l in cache.layers)
print(f"{nbytes / 1024:.0f} KiB")                   # 640 KiB = 5 tokens × 128 KiB

nxt = out.logits[:, -1:].argmax(-1)                 # [1, 1]
out = model(nxt, past_key_values=cache, use_cache=True)   # decode: feed only the new token
print(cache.get_seq_length())                       # 6 — the same cache object grew in place
```

(Older transformers versions return a tuple of per-layer `(k, v)` pairs or expose `cache.key_cache[i]`; the shapes are the same.)

## KV-Cache vs Model Weights Memory

At long context lengths, KV-cache can exceed the model weights:

| Context | Model Weights (8B, BF16) | KV-Cache (8B, GQA-8) | KV-Cache % |
|:---:|:---:|:---:|:---:|
| 2K | 16 GB | 0.25 GB | 1.5% |
| 8K | 16 GB | 1 GB | 6% |
| 32K | 16 GB | 4 GB | 20% |
| 128K | 16 GB | 16 GB | 50% |
| 512K | 16 GB | 64 GB | 80% |

At 128K context, the KV-cache equals the model size. At 512K, it's 4× the model. This is why context extension ([Chapter 12](./12_mid_training.md)) and KV-cache optimization ([Chapter 18](./18_kv_cache_optimization.md)) go hand in hand.

## Batching and KV-Cache

When serving many requests, the cache — not the weights — decides how many fit at once, and batch size decides throughput:

| Llama 3.1 8B, BF16 on one 80 GB GPU | |
|---|---|
| Weights | 16 GB |
| Left for KV cache (after ~4 GB activations/workspace) | ~60 GB |
| Cache per 32K-token request | 32,768 × 128 KiB = 4 GiB |
| Concurrent 32K requests | **~14** |
| Concurrent 2K requests | **~220** |

Each request's cache grows at its own rate and ends at an unpredictable length, so naive allocation (reserve the maximum per request) wastes most of that memory. Managing it like an operating system manages RAM is the job of **PagedAttention** and the rest of [Chapter 18](./18_kv_cache_optimization.md).

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A reasoning model may write 30,000 thinking tokens to answer a 200-token question. Its KV cache is then dominated by text no user will ever read. Which of the ideas in Chapter 18 — compression, eviction, offloading, sharing — fits a cache that is mostly the model's own scratch work?</p>
</div>

## Key Takeaways

- The cache stores per-layer K and V for every past token; decode then runs the model on one token per step.
- KV bytes per token = 2 × layers × KV heads × head dim × bytes per value (MLA: layers × (latent + RoPE dim) × bytes).
- Decode is bandwidth-bound: tokens/s ≲ bandwidth / (weights + KV bytes) at batch 1; batching shares weight reads but not cache reads.
- At long context the cache exceeds the weights, and cache capacity sets the batch size — and therefore throughput and cost.
- Attention design (GQA, MLA, windows, hybrids) changes the cache by 10–100×; it is a serving decision as much as an architecture one.

## Check Yourself

<details class="check">
<summary>Llama 3.1 70B serves a 64K-token conversation in BF16. How large is the cache, and what fraction of an 80 GB GPU is that?</summary>
<p>65,536 × 320 KiB = 20 GiB ≈ 21.5 GB — over a quarter of one GPU, for a single request (and the 140 GB of weights need two GPUs anyway).</p>
</details>

<details class="check">
<summary>Why does batching raise decode throughput almost linearly at first, and why does it eventually stop?</summary>
<p>At batch 1 each step reads all weights to produce one token; with batch B the same weight read produces B tokens, so throughput grows ~B× while still bandwidth-bound. It stops when either compute becomes the bottleneck (arithmetic intensity reaches the GPU's ridge point) or the KV caches of B sequences no longer fit in memory.</p>
</details>

<details class="check">
<summary>Why can the cache store K and V but not Q?</summary>
<p>Past queries are never needed again: each query is used only at its own step to read the cache. Keys and values of past tokens are read by every future query, so they are worth keeping.</p>
</details>

<details class="check">
<summary>What is the upper bound on single-sequence decode speed for a 30 GB model on a 4.8 TB/s H200 with a 6 GB cache?</summary>
<p>4.8 TB/s / 36 GB ≈ 133 tokens/s — before kernel overheads, which typically leave you at 60–80% of the bound.</p>
</details>

<details class="check">
<summary>gpt-oss-120b has 36 layers but the table counts only 18 for its growing cache. Why?</summary>
<p>Half its layers use a 128-token sliding window, so their cache is capped at 128 tokens regardless of context length; only the 18 full-attention layers grow with T.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Efficiently Scaling Transformer Inference | Pope et al. (Google), 2022 | [arxiv.org/abs/2211.05102](https://arxiv.org/abs/2211.05102) | KV-cache memory and decode bandwidth analysis |
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al., 2023 | [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) | Cache fragmentation, batching |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | Llama 3.1 shapes |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | MLA cache dimensions |
| gpt-oss model card | OpenAI, 2025 | [openai.com/index/gpt-oss-model-card](https://openai.com/index/gpt-oss-model-card/) | Windowed/full layer pattern |
| Transformers cache documentation | Hugging Face, 2025 | [huggingface.co/docs/transformers/cache_explanation](https://huggingface.co/docs/transformers/cache_explanation) | DynamicCache API |

---

**Next:** [Chapter 18 — KV-Cache: Optimization Strategies](./18_kv_cache_optimization.md) — paging, sharing, compressing, evicting and offloading the cache.

[← Back to Table of Contents](./README.md)
