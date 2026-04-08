[← Back to Table of Contents](./README.md)

# Chapter 14 — KV-Cache: Mechanics & Memory

> *"The KV-cache is the single most important optimization in LLM inference — without it, generating each token would require recomputing attention over the entire sequence from scratch."*

## Why KV-Cache Exists

During autoregressive generation, each new token needs to attend to all previous tokens. Without caching, we'd recompute Q, K, V for every previous token at every step — O(T²) total computation to generate T tokens.

The KV-cache stores the Key and Value projections from all previous tokens, so each decode step only computes Q, K, V for the **new** token and reuses everything else.

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
      <li>Total compute: O(T²) per layer</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">With KV-Cache</div>
    <ul>
      <li>Step 1: compute & cache K₁, V₁</li>
      <li>Step 2: compute K₂, V₂, append to cache</li>
      <li>Step 3: compute K₃, V₃, append to cache</li>
      <li>Step T: compute Kₜ, Vₜ, attend to cached [1..T]</li>
      <li>Total compute: O(T) per layer ✓</li>
    </ul>
  </div>
</div>
</div>

## Step-by-Step Visualization

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

This is a **matrix-vector product** (not matrix-matrix), making decode steps **memory-bandwidth bound** rather than compute-bound.

## Tensor Shapes in Detail

<div class="diagram">
<div class="diagram-title">KV-Cache Tensor Shapes (Per Layer)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">K cache (MHA)</span>
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
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">V cache (MHA)</span>
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
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">K cache (GQA, G groups)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">G < H → smaller cache</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Per-step append</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">1</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">one new K, V per layer per step</span>
  </div>
</div>
</div>

## Memory Calculation

The KV-cache memory for the **entire model**:

$$\text{KV memory} = 2 \times L \times G \times d_k \times T \times B \times \text{bytes}$$

where: L = layers, G = KV heads, d_k = head dimension, T = sequence length, B = batch size, bytes = 2 (FP16/BF16).

### Worked Examples

| Model | L | G (KV heads) | d_k | Formula | Per Token | At 8K | At 128K |
|-------|:-:|:---:|:---:|---------|:---:|:---:|:---:|
| LLaMA-2 7B (MHA) | 32 | 32 | 128 | 2×32×32×128×2 | 512 KB | 4 GB | 64 GB |
| LLaMA-3 8B (GQA) | 32 | 8 | 128 | 2×32×8×128×2 | 128 KB | 1 GB | 16 GB |
| LLaMA-3 70B (GQA) | 80 | 8 | 128 | 2×80×8×128×2 | 320 KB | 2.5 GB | 40 GB |
| Mistral 7B (GQA) | 32 | 8 | 128 | 2×32×8×128×2 | 128 KB | 1 GB | 16 GB |
| LLaMA-3.1 405B | 126 | 8 | 128 | 2×126×8×128×2 | 504 KB | 3.9 GB | 63 GB |

> **Key insight**: The KV-cache for LLaMA-3.1 405B at 128K context exceeds 63 GB **per request** in FP16. This is why KV-cache optimization ([Chapter 15](./15_kv_cache_optimization.md)) is critical.

## GQA/MQA Impact on KV-Cache

The attention variant directly determines KV-cache size (see [Chapter 5](./05_attention_gqa_mqa_mla.md)):

<div class="diagram">
<div class="diagram-title">KV-Cache Size by Attention Variant (LLaMA-3-8B Scale)</div>
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
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Full KV-cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">grows with T — unbounded</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Sliding window cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">W</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">fixed at W — evicts oldest entry</span>
  </div>
</div>
</div>

## Inspecting KV-Cache in Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(model.device)

# First forward pass (prefill)
outputs = model(input_ids, use_cache=True)
past_kv = outputs.past_key_values

# Inspect shapes
for layer_idx, (k, v) in enumerate(past_kv):
    if layer_idx == 0:
        print(f"Layer {layer_idx}:")
        print(f"  K shape: {k.shape}")  # [1, 8, T, 128] for GQA with 8 KV heads
        print(f"  V shape: {v.shape}")  # [1, 8, T, 128]

# Total KV-cache memory
total_bytes = sum(k.nbytes + v.nbytes for k, v in past_kv)
print(f"Total KV-cache: {total_bytes / 1024**2:.1f} MB")

# Decode one more token — only feed the last token
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
outputs2 = model(next_token, past_key_values=past_kv, use_cache=True)
past_kv2 = outputs2.past_key_values

# Cache grew by 1 token
k0_old, k0_new = past_kv[0][0], past_kv2[0][0]
print(f"Cache grew: {k0_old.shape[2]} → {k0_new.shape[2]} tokens")
```

## KV-Cache vs Model Weights Memory

At long context lengths, KV-cache can exceed the model weights:

| Context | Model Weights (8B, BF16) | KV-Cache (8B, GQA-8) | KV-Cache % |
|:---:|:---:|:---:|:---:|
| 2K | 16 GB | 0.25 GB | 1.5% |
| 8K | 16 GB | 1 GB | 6% |
| 32K | 16 GB | 4 GB | 20% |
| 128K | 16 GB | 16 GB | 50% |
| 512K | 16 GB | 64 GB | 80% |

At 128K context, the KV-cache equals the model size. At 512K, it's 4× the model. This is why context extension ([Chapter 10](./10_mid_training.md)) and KV-cache optimization ([Chapter 15](./15_kv_cache_optimization.md)) go hand in hand.

## Batching and KV-Cache

When serving multiple requests, each request has its own KV-cache of potentially different length. Managing memory across a dynamic batch is non-trivial — this is the core problem that **PagedAttention** and other optimizations solve (see [Chapter 15](./15_kv_cache_optimization.md)).

## What's Next

The KV-cache memory problem is the central challenge of LLM serving. The next chapter covers the full landscape of **KV-cache optimization** — from PagedAttention to quantized KV-cache to token eviction strategies.

[← Previous: Chapter 13 — Inference & Sampling](./13_inference_and_sampling.md) · **Next: [Chapter 15 — KV-Cache Optimization →](./15_kv_cache_optimization.md)**

---

*Last updated: April 2026*
