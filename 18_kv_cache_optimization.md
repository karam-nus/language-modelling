---
title: "Chapter 18 — KV-Cache Optimization Strategies"
description: "Managing the KV cache like virtual memory: PagedAttention, continuous batching and chunked prefill, prefix caching and RadixAttention, FP8 and low-bit cache quantization, eviction and query-aware sparsity, offloading to CPU and SSD tiers, and disaggregated prefill/decode with KV transfer."
related:
  - { id: inference, why: "Serving-side optimizations in practice." }
  - { id: chips, why: "Why memory capacity and bandwidth dominate decode." }
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — KV-Cache Optimization Strategies

Chapter 17 established that the KV cache, not the weights, decides how many requests a GPU can serve and how long they can be. This chapter is the toolbox for that problem — most of it now built into vLLM, SGLang, TensorRT-LLM and NVIDIA Dynamo.

> **The one-sentence version:** serve more and longer requests by wasting less cache (paging), sharing it (prefix caching), shrinking it (quantization, eviction, better attention), moving it (offloading, disaggregation), or reading less of it (sparse selection).

## The Problem Space

<div class="diagram">
<div class="diagram-title">KV-Cache Optimization Landscape</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Allocate</div>
    <div class="card-desc">PagedAttention, continuous batching, chunked prefill — waste nothing, keep the GPU busy.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Share</div>
    <div class="card-desc">Prefix caching, RadixAttention, cache-aware routing — compute a shared prefix once.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Shrink</div>
    <div class="card-desc">FP8/FP4 cache, KIVI-style low-bit, eviction (StreamingLLM, H2O, SnapKV); architecture (GQA, MLA, windows).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Move</div>
    <div class="card-desc">Offload cold blocks to CPU RAM and SSD (LMCache, Mooncake); ship caches from prefill to decode workers.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Read less</div>
    <div class="card-desc">Query-aware sparse attention (Quest, DSA, V4's compressed sparse attention): keep the cache, read a fraction of it.</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-title">Schedule</div>
    <div class="card-desc">Admission control, preemption and recomputation when memory runs out mid-request.</div>
  </div>
</div>
</div>

## PagedAttention (vLLM)

**PagedAttention** (Kwon et al., 2023) manages the cache the way an operating system manages RAM.

### The Fragmentation Problem

Without paging, each request reserves a contiguous region big enough for its maximum length:

<div class="diagram">
<div class="diagram-title">Contiguous vs Paged KV-Cache</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Contiguous (naive)</div>
    <ul>
      <li>Reserve max_seq_len per request</li>
      <li>A request that uses 2K of a reserved 8K wastes 75%</li>
      <li>Free gaps between requests are too small to reuse</li>
      <li>The vLLM paper measured only 20–38% of cache memory holding real tokens in earlier systems</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Paged (vLLM)</div>
    <ul>
      <li>Fixed-size blocks (e.g., 16 tokens)</li>
      <li>Blocks need not be contiguous</li>
      <li>Allocate a new block only when the last one fills</li>
      <li>Waste &lt; one block per request → 2–4× throughput in the paper</li>
    </ul>
  </div>
</div>
</div>

Watch both allocators handle the same stream of requests:

<div class="lab" data-lab="paged"><noscript>Interactive lab: compare contiguous and paged KV-cache allocation on a stream of requests with random lengths. Enable JavaScript to use it.</noscript></div>

### How PagedAttention Works

<div class="diagram">
<div class="diagram-title">PagedAttention — Block-Level KV-Cache</div>
<div class="flow">
  <div class="flow-node accent wide">Cache split into physical blocks: [block_size, n_kv_heads, d_k] per layer <small>e.g., block_size = 16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Per-request block table maps logical blocks → physical blocks <small>like a page table</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Attention kernel gathers K, V through the block table <small>FlashAttention / FlashInfer paged kernels</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">New tokens fill the current block; allocate a new one when full <small>free blocks return to a global pool when a request ends</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Physical KV Block (Llama 3.1 8B, one layer)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">K block</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim seq">16</span><span class="ts-sep">,</span><span class="ts-dim kv">8</span><span class="ts-sep">,</span><span class="ts-dim feature">128</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">tokens × KV heads × head dim</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">V block</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim seq">16</span><span class="ts-sep">,</span><span class="ts-dim kv">8</span><span class="ts-sep">,</span><span class="ts-dim feature">128</span><span class="ts-bracket">]</span></div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Block memory</span>
    <div class="tensor-shape"><span class="ts-dim generic">2 × 16 × 8 × 128 × 2 B = 64 KiB</span></div>
    <span class="ts-note">per block per layer, BF16</span>
  </div>
</div>
</div>

## Continuous Batching and Chunked Prefill

**Static batching** runs a fixed batch until its longest request finishes. **Continuous (in-flight) batching** (Orca, 2022) schedules at every decode iteration: finished sequences leave, waiting ones join immediately.

<div class="diagram">
<div class="diagram-title">Static vs Continuous Batching</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Static batching</div>
    <ul>
      <li>Batch of 8 processed together</li>
      <li>Request 3 finishes early → its slot idles</li>
      <li>Everyone waits for the longest request</li>
      <li>Poor utilisation with variable lengths</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Continuous batching</div>
    <ul>
      <li>Request 3 finishes → a new request takes its slot next step</li>
      <li>Iteration-level scheduling</li>
      <li>Batch stays full</li>
      <li>Several-fold throughput on real traffic</li>
    </ul>
  </div>
</div>
</div>

A new request's prefill is a large burst of compute; dropped into a running batch whole, it stalls every decoding sequence for that step. **Chunked prefill** splits long prompts into chunks (e.g., 512–8K tokens) and mixes them with decode steps, keeping inter-token latency smooth. It is on by default in modern vLLM.

## Prefix Caching

Requests often share a prefix — a system prompt, tool definitions, a long document, the earlier turns of a conversation, an agent's growing transcript. **Prefix caching** keeps the KV blocks of computed prefixes and reuses them:

<div class="diagram">
<div class="diagram-title">Prefix Caching with Paged Blocks</div>
<div class="flow">
  <div class="flow-node accent wide">Shared prefix: system prompt + tool schemas <small>KV blocks computed once, hashed by token content</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Request 1: [shared blocks] → [own blocks for "What is ML?"]</div>
  <div class="flow-node green wide">Request 2: [shared blocks] → [own blocks for "Write a poem"]</div>
  <div class="flow-node orange wide">Request 3: [shared blocks] → [own blocks for "Explain RoPE"]</div>
</div>
</div>

vLLM identifies each full block by a hash of its tokens (and all tokens before it); any request whose prefix produces the same hashes simply points at the existing blocks, with reference counts deciding when they can be freed. SGLang's **RadixAttention** keeps prefixes in a radix tree, which suits branching workloads (agents, tree search, few-shot variants). Hit rates matter enormously for agents: each turn re-sends the whole transcript, so without prefix caching an hour-long agent session re-prefills millions of tokens.

At fleet scale, **cache-aware routing** sends a request to the replica that already holds its prefix (SGLang router, NVIDIA Dynamo, llm-d). API providers expose the same idea as **prompt caching** — cached input tokens are billed at a fraction of the normal price — and recommend putting stable content at the start of prompts.

## KV-Cache Quantization

Storing K and V in fewer bits saves memory *and* bandwidth, since decode reads the whole cache every step:

<div class="diagram">
<div class="diagram-title">KV-Cache Quantization Impact (per token, Llama 3.1 8B)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row"><span class="ts-label">BF16 cache</span><div class="tensor-shape"><span class="ts-dim generic">128 KiB</span></div><span class="ts-note">2 bytes per value</span></div>
  <div class="tensor-flow-row"><span class="ts-label">FP8 cache</span><div class="tensor-shape"><span class="ts-dim generic">64 KiB</span></div><span class="ts-note">1 byte → 2× — the 2025–26 serving default</span></div>
  <div class="tensor-flow-row"><span class="ts-label">4-bit cache</span><div class="tensor-shape"><span class="ts-dim generic">~34 KiB</span></div><span class="ts-note">0.5 byte + scales → ~3.8×</span></div>
  <div class="tensor-flow-row"><span class="ts-label">2-bit (KIVI)</span><div class="tensor-shape"><span class="ts-dim generic">~18 KiB</span></div><span class="ts-note">research; group-wise with a full-precision recent window</span></div>
</div>
</div>

| Method | Approach | Key insight |
|--------|----------|-------------|
| **FP8 KV** (E4M3 + per-tensor or per-head scales) | Native in vLLM, SGLang, TensorRT-LLM | Near-lossless for most models; halves cache size and read bandwidth |
| **KIVI** (2024) | 2-bit: keys per-channel, values per-token; recent tokens kept in full precision | Keys have channel-wise outliers (from RoPE and massive activations); values do not |
| **KVQuant** (2024) | Pre-RoPE key quantization, non-uniform levels, outlier isolation | Quantize keys *before* rotation to keep channel structure |
| **QServe** (2024) | W4A8KV4 co-designed kernels | System–algorithm co-design for 4-bit serving |
| **NVFP4 KV** (2025–26) | Blackwell's 4-bit float with fine-grained block scales | Hardware-supported 4-bit cache |

## Eviction and Sparse Reading

Not every cached token matters equally. Two families of ideas:

**Evict** — permanently drop entries to fit a budget:

<div class="diagram">
<div class="diagram-title">Eviction Strategies</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">StreamingLLM</div>
    <div class="card-desc">Keep the first few "attention sink" tokens plus a recent window. Stable generation over millions of tokens with a fixed cache — but anything in the middle is forgotten.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">H2O / SnapKV / PyramidKV</div>
    <div class="card-desc">Keep tokens that received the most attention (H2O), or that the last part of the prompt attends to (SnapKV); give lower layers bigger budgets (PyramidKV).</div>
  </div>
</div>
</div>

```
StreamingLLM cache layout: [4 sink tokens] + [recent window of 4,092] = fixed 4,096 entries
                            ↑ always kept      ↑ slides forward every step
```

Eviction is dangerous for multi-turn chat and agents: a token that looked unimportant now may be exactly what a later question needs, and it is gone.

**Read sparsely** — keep the full cache (possibly offloaded) but read only the relevant part each step: Quest selects pages by estimating each page's maximum attention score for the current query; DuoAttention gives full caches to "retrieval heads" and short windows to "streaming heads"; DeepSeek Sparse Attention and DeepSeek-V4 train the selection into the model ([Chapter 5](./05_attention_gqa_mqa_mla.md)). Because nothing is thrown away, these are safer for long, multi-turn use.

## Offloading and Tiered Caches

GPU memory is the most expensive tier. Cold cache blocks — finished conversation turns, idle agent sessions, shared documents — can move down a hierarchy:

| Tier | Capacity per server | Bandwidth to GPU | Use |
|---|---|---|---|
| HBM | 80–288 GB per GPU | 3–8 TB/s | Active decoding |
| CPU DRAM (via PCIe / NVLink-C2C) | 1–2 TB | 64–900 GB/s | Recently idle sessions, shared prefixes |
| Local NVMe SSD | tens of TB | a few GB/s per drive | Long-lived prefixes, multi-turn history |
| Remote / pooled storage | cluster-wide | RDMA network | Cross-replica sharing (Mooncake store, LMCache, Dynamo KV block manager) |

Reloading a cached prefix from CPU or SSD is usually far faster than recomputing it with a fresh prefill, especially for long contexts. Moonshot's **Mooncake** (Kimi's serving system) was built around this "KV-centric" idea; DeepSeek's API serves cached input tokens from a disk cache.

## Disaggregated Prefill and Decode

Prefill is compute-bound; decode is bandwidth-bound. Running both on the same GPUs makes them interfere (a big prefill stalls everyone's decode) and forces one hardware configuration on both. **Disaggregation** (DistServe, Splitwise, Mooncake, 2024) runs them on separate pools:

<div class="diagram">
<div class="diagram-title">Disaggregated Serving</div>
<div class="flow-h">
  <div class="flow-node accent">Router<br><small>cache-aware</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple">Prefill pool<br><small>compute-heavy, high TP</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange">KV transfer<br><small>RDMA / NVLink (NIXL)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Decode pool<br><small>large batches, wide expert parallelism</small></div>
</div>
</div>

The price is moving the cache: a 32K-token Llama 3.1 70B request carries 10 GiB of BF16 KV from the prefill GPU to the decode GPU. With 400 Gb/s RDMA that is ~0.2 s — acceptable against a multi-second generation, and overlappable layer by layer. By 2025–26 disaggregation is standard in large deployments: NVIDIA Dynamo, vLLM and SGLang support it, and DeepSeek described serving V3/R1 with separate prefill and decode clusters and wide expert parallelism.

## Comparison of Optimization Strategies

| Strategy | Memory effect | Quality impact | Latency impact | Where |
|----------|:---:|:---:|:---:|:---:|
| GQA / MLA / windows | 4× (GQA-8) to ~70× (MLA vs MHA) | Minimal if trained in | None | Architecture |
| PagedAttention | Removes fragmentation waste | None | Small kernel overhead | Engine |
| Continuous batching + chunked prefill | Higher utilisation | None | Smoother ITL | Engine |
| Prefix caching | Proportional to sharing | None | Much faster prefill on hits | Engine / fleet |
| FP8 KV | 2× | Negligible for most models | Faster decode (fewer bytes) | Kernel |
| 4-bit / 2-bit KV | ~4–7× | Small to noticeable; model-dependent | Dequant overhead | Kernel |
| Eviction (StreamingLLM, SnapKV) | Fixed budget | Loses information; risky for multi-turn | None | Runtime |
| Sparse reading (Quest, DSA) | Reads, not storage | Small if well-trained | Faster long-context decode | Kernel / architecture |
| Offloading to CPU / SSD | Capacity ×10–100 | None | Reload latency on hit | System |
| Prefill/decode disaggregation | Separate pools | None | Better tail latency; KV transfer | Cluster |

## Using These Optimizations

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    kv_cache_dtype="fp8",            # FP8 (E4M3) cache — half the memory and read bandwidth
    max_model_len=32768,
    enable_prefix_caching=True,      # on by default in recent vLLM
    gpu_memory_utilization=0.90,     # fraction of GPU memory vLLM may use; the rest after weights becomes KV blocks
)
outs = llm.generate(["What is PagedAttention?", "Explain KV-cache quantization."],
                    SamplingParams(temperature=0.7, max_tokens=512))
```

The same options on the OpenAI-compatible server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 --max-model-len 32768 \
  --enable-chunked-prefill --gpu-memory-utilization 0.90
```

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>An agent re-sends a growing transcript every turn, so its prefix-cache hit rate can exceed 95% — until the cache evicts its blocks during a lull. If cached input tokens cost a tenth of uncached ones, how should a serving system decide which sessions' caches to keep in HBM, which to push to SSD, and which to drop?</p>
</div>

## Key Takeaways

- Paging (fixed blocks + block tables) eliminates fragmentation; continuous batching and chunked prefill keep the GPU busy without latency spikes.
- Prefix caching (hashed blocks, radix trees) and cache-aware routing turn repeated prompts and agent transcripts into near-free prefill.
- FP8 KV is the default 2× win; 4-bit and 2-bit caches need care with key outliers.
- Eviction saves memory by forgetting; sparse reading saves bandwidth without forgetting — safer for multi-turn and agentic use.
- Tiered offloading and prefill/decode disaggregation treat the KV cache as a first-class distributed data structure.

## Check Yourself

<details class="check">
<summary>With 16-token blocks, what is the worst-case waste per request, and how does that compare with reserving 8K tokens for a request that uses 1,000?</summary>
<p>At most 15 token slots in the last block (under 2% of a 1,000-token request). The contiguous reservation wastes 7,192 of 8,192 slots — 88%.</p>
</details>

<details class="check">
<summary>Why does chunked prefill improve inter-token latency for users who are already generating?</summary>
<p>Without chunking, a 30K-token prefill occupies an entire scheduler step, so every decoding sequence waits for it. Splitting it into chunks interleaved with decode steps bounds each step's work, so ongoing generations keep producing tokens.</p>
</details>

<details class="check">
<summary>Why are keys harder to quantize to 2 bits than values?</summary>
<p>Keys have a few channels with consistently large magnitudes (outlier channels, amplified by RoPE), so a per-token scale would be dominated by them. KIVI quantizes keys per channel and values per token to match where each tensor's outliers live.</p>
</details>

<details class="check">
<summary>How long does it take to move a 32K-token Llama 3.1 70B BF16 cache over a 400 Gb/s link?</summary>
<p>32,768 × 320 KiB = 10 GiB ≈ 10.7 GB. 400 Gb/s = 50 GB/s, so ≈ 0.21 s — and it can be streamed layer by layer while prefill is still running.</p>
</details>

<details class="check">
<summary>An eviction policy scores well on single-turn long-document QA but fails on follow-up questions. Why?</summary>
<p>It chose which tokens to keep based on the first question's attention pattern. The follow-up needs different parts of the document, which were already evicted. Sparse-reading methods keep the full cache and select per query, avoiding this.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu et al., 2022 | [usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration-level (continuous) batching |
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al., 2023 | [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180) | PagedAttention, fragmentation measurements |
| SGLang: Efficient Execution of Structured LM Programs | Zheng et al., 2023 | [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104) | RadixAttention |
| Efficient Streaming Language Models with Attention Sinks | Xiao et al., 2023 | [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453) | StreamingLLM |
| H2O: Heavy-Hitter Oracle | Zhang et al., 2023 | [arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048) | Attention-score eviction |
| SnapKV | Li et al., 2024 | [arxiv.org/abs/2404.14469](https://arxiv.org/abs/2404.14469) | Prompt-observation eviction |
| KIVI | Liu et al., 2024 | [arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750) | 2-bit asymmetric KV quantization |
| KVQuant | Hooper et al., 2024 | [arxiv.org/abs/2401.18079](https://arxiv.org/abs/2401.18079) | Pre-RoPE key quantization |
| Quest: Query-Aware Sparsity for Long-Context Inference | Tang et al., 2024 | [arxiv.org/abs/2406.10774](https://arxiv.org/abs/2406.10774) | Query-aware page selection |
| DistServe | Zhong et al., 2024 | [arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670) | Prefill/decode disaggregation |
| Mooncake: A KVCache-centric Disaggregated Architecture | Qin et al. (Moonshot), 2024 | [arxiv.org/abs/2407.00079](https://arxiv.org/abs/2407.00079) | KV-centric serving, tiered cache |
| vLLM documentation | vLLM project, 2025 | [docs.vllm.ai](https://docs.vllm.ai/) | Prefix caching, FP8 KV, chunked prefill |

---

**Next:** [Chapter 19 — Data Types & Numerical Precision](./19_data_types_and_precision.md) — the number formats underneath every quantization trick.

[← Back to Table of Contents](./README.md)
