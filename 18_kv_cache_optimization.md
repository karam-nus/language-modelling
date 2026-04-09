---
title: "Chapter 18 — KV-Cache Optimization Strategies"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — KV-Cache Optimization Strategies

> *"The art of LLM serving is the art of KV-cache management — fitting more requests, longer contexts, and higher throughput into the same GPU memory."*

## The Problem Space

The KV-cache is the dominant memory consumer in LLM inference (see [Chapter 17](./17_kv_cache_mechanics.md)). Optimization strategies fall into several categories:

<div class="diagram">
<div class="diagram-title">KV-Cache Optimization Landscape</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Memory Management</div>
    <div class="card-desc">PagedAttention, continuous batching, prefix caching. How we allocate and reuse KV-cache memory.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Compression</div>
    <div class="card-desc">Quantized KV-cache (INT4/INT8), token eviction, low-rank compression. Reduce bits per cached token.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Architectural</div>
    <div class="card-desc">GQA/MQA (fewer heads), MLA (latent compression), Sliding Window (bounded cache). Design-time choices.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Offloading</div>
    <div class="card-desc">CPU/disk offload of older KV blocks. Trade latency for capacity.</div>
  </div>
</div>
</div>

## PagedAttention (vLLM)

**PagedAttention** (Kwon et al., 2023) applies virtual memory concepts to KV-cache management, eliminating fragmentation and enabling efficient memory sharing.

### The Fragmentation Problem

Without PagedAttention, each request pre-allocates a contiguous KV-cache for its maximum possible length. This wastes memory:

<div class="diagram">
<div class="diagram-title">Contiguous vs Paged KV-Cache</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Contiguous (Naive)</div>
    <ul>
      <li>Pre-allocate max_seq_len per request</li>
      <li>Request uses 2K tokens but allocated 8K</li>
      <li>75% memory wasted (internal fragmentation)</li>
      <li>Can't reclaim until request finishes</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Paged (vLLM)</div>
    <ul>
      <li>Allocate in fixed-size blocks (e.g., 16 tokens)</li>
      <li>Blocks can be non-contiguous in physical memory</li>
      <li>Allocate new blocks on demand</li>
      <li>Near-zero waste, ~4× more requests per GPU</li>
    </ul>
  </div>
</div>
</div>

### How PagedAttention Works

<div class="diagram">
<div class="diagram-title">PagedAttention — Block-Level KV-Cache</div>
<div class="flow">
  <div class="flow-node accent wide">KV-cache divided into fixed blocks: [block_size, n_kv_heads, d_k] <small>e.g., block_size=16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Page table maps logical positions → physical block addresses <small>like OS virtual memory</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Attention kernel reads blocks via page table <small>non-contiguous memory access</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">New tokens fill current block; allocate new block when full <small>on-demand allocation</small></div>
</div>
</div>

Each physical block stores KV entries for `block_size` tokens:

<div class="diagram">
<div class="diagram-title">Physical KV Block</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">K block</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim seq">block_size</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">e.g., [16, 8, 128]</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">V block</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim seq">block_size</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Block memory</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim generic">2 × 16 × 8 × 128 × 2B = 64 KB</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">per block per layer</span>
  </div>
</div>
</div>

## Continuous Batching

Traditional **static batching** waits until a batch is full, processes all requests, and waits until all finish. **Continuous batching** inserts new requests as soon as any slot frees up:

<div class="diagram">
<div class="diagram-title">Static vs Continuous Batching</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Static Batching</div>
    <ul>
      <li>Batch of 8 requests processed together</li>
      <li>Request 3 finishes early → its slot is idle</li>
      <li>Entire batch waits for the longest request</li>
      <li>Poor GPU utilization</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Continuous Batching</div>
    <ul>
      <li>Request 3 finishes → immediately replaced</li>
      <li>No idle slots, maximum GPU utilization</li>
      <li>Iteration-level scheduling (per-token)</li>
      <li>2–3× higher throughput</li>
    </ul>
  </div>
</div>
</div>

## Prefix Caching

Many requests share the same system prompt (e.g., "You are a helpful assistant..."). **Prefix caching** computes and stores this shared KV-cache once, then reuses it for all requests with the same prefix:

<div class="diagram">
<div class="diagram-title">Prefix Caching with PagedAttention</div>
<div class="flow">
  <div class="flow-node accent wide">Shared prefix: "You are a helpful assistant..." <small>KV blocks cached once</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Request 1: [shared prefix blocks] → [unique blocks for "What is ML?"]</div>
  <div class="flow-node green wide">Request 2: [shared prefix blocks] → [unique blocks for "Write a poem"]</div>
  <div class="flow-node orange wide">Request 3: [shared prefix blocks] → [unique blocks for "Explain RoPE"]</div>
</div>
</div>

In vLLM, shared prefix blocks use **copy-on-write**: all requests point to the same physical blocks. Only the unique suffix blocks are allocated per request.

## KV-Cache Quantization

Storing the KV-cache in lower precision (INT8, INT4) provides direct memory savings:

<div class="diagram">
<div class="diagram-title">KV-Cache Quantization Impact</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">FP16 KV-cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ 2 bytes</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">INT8 KV-cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ 1 byte → 2× savings</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">INT4 KV-cache</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim kv">G</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_k</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ 0.5 byte → 4× savings</span>
    </div>
  </div>
</div>
</div>

### Methods

| Method | Approach | Key Insight |
|--------|----------|-------------|
| **KIVI** | Per-channel K, per-token V quantization to 2-bit | Keys have per-channel outliers; values have per-token outliers |
| **KVQuant** | Non-uniform quantization with sensitivity weighting | Different heads have different quantization sensitivity |
| **QServe KV4** | Joint W4A8KV4 optimization | Co-design kernel for weights + activations + KV |
| **Gear** | Low-rank + sparse residual KV compression | Separate low-rank approximation from sparse outliers |

## Token Eviction and Pruning

Not all cached tokens are equally important. **Token eviction** strategies selectively remove less-important cached entries:

<div class="diagram">
<div class="diagram-title">Token Eviction Strategies</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">H2O (Heavy-Hitter Oracle)</div>
    <div class="card-desc">Keep tokens that accumulate the most attention weight over time. "Heavy hitter" tokens are consistently attended to. Budget: keep top-k important + recent window.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">StreamingLLM (Attention Sinks)</div>
    <div class="card-desc">Keep initial "attention sink" tokens (first 4) + recent sliding window. The first tokens absorb disproportionate attention mass regardless of content.</div>
  </div>
</div>
</div>

StreamingLLM insight: attention scores for the first few tokens are always high (attention sinks). Keeping these + a recent window enables infinite-length generation with fixed memory:

```
Cache layout: [sink tokens (4)] + [recent window (4092)] = fixed 4096 entries
             ↑ always kept        ↑ sliding window
```

## Comparison of Optimization Strategies

| Strategy | Memory Savings | Quality Impact | Latency Impact | Complexity |
|----------|:---:|:---:|:---:|:---:|
| **GQA (architectural)** | 4–8× | Minimal | None | Design-time |
| **MLA (architectural)** | 8–16× | Minimal | Slight overhead | Design-time |
| **Sliding Window** | Bounded at W | Good for local tasks | None | Design-time |
| **PagedAttention** | ~0% (reduces waste) | None | Slight overhead | Framework |
| **Continuous batching** | Higher utilization | None | Better | Framework |
| **Prefix caching** | Proportional to sharing | None | Faster prefill | Framework |
| **KV INT8** | 2× | Negligible (<0.1 PPL) | Slight overhead | Kernel |
| **KV INT4** | 4× | Small (~0.2 PPL) | Slight overhead | Kernel |
| **H2O eviction** | ~2–4× (fixed budget) | Small | None | Runtime |
| **StreamingLLM** | Bounded at window | Good for streaming | None | Runtime |
| **CPU offload** | Extends to system RAM | None | Higher latency | System |

## Using These Optimizations

### vLLM with PagedAttention + KV Quantization

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    dtype="bfloat16",
    kv_cache_dtype="fp8_e5m2",   # FP8 KV-cache — 2× savings
    max_model_len=32768,
    enable_prefix_caching=True,   # Prefix caching enabled
    gpu_memory_utilization=0.90,  # Use 90% of GPU for KV blocks
)

# PagedAttention and continuous batching are automatic
outputs = llm.generate(
    ["What is PagedAttention?", "Explain KV-cache quantization."],
    SamplingParams(temperature=0.7, max_tokens=512),
)
```

### TGI with KV-Cache Options

```bash
docker run --gpus all \
  -e MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  -e QUANTIZE=bitsandbytes-fp4 \
  -e MAX_INPUT_LENGTH=4096 \
  -e MAX_TOTAL_TOKENS=8192 \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference
```

## What's Next

KV-cache precision is just one aspect of numerical representation. The next chapter provides a comprehensive guide to **data types and numerical precision** — the foundation for understanding quantization.

[← Previous: Chapter 17 — KV-Cache — Mechanics & Memory](./17_kv_cache_mechanics.md) · **Next: [Chapter 19 — Data Types & Numerical Precision →](./19_data_types_and_precision.md)**

---

*Last updated: April 2026*
