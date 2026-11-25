---
title: "Chapter 22 — Quantization Benchmarks & Selection Guide"
description: "How to choose a quantization scheme: what large published studies found (FP8 ≈ lossless, INT4 weight-only competitive, reasoning models more sensitive), bandwidth-bound speed ceilings by GPU, what fits where, a decision guide for datacenter, local and fine-tuning use, and how to benchmark properly."
---

[← Back to Table of Contents](./README.md)

# Chapter 22 — Quantization Benchmarks & Selection Guide

There is no single best quantization method — only the best one for a given model, task, hardware and traffic pattern. This chapter gives you the evidence to choose and the method to check your choice.

> **The one-sentence version:** FP8 is free, 4-bit weights are cheap, below 4 bits you pay — and the price depends on model size, task (reasoning and long context pay most) and whether your serving is bandwidth- or compute-bound.

## What Large Studies Found

Treat single perplexity tables on the internet with suspicion: numbers depend on context length, evaluation code, calibration data and kernel. The most useful evidence comes from large, systematic studies:

| Finding | Evidence |
|---|---|
| **FP8 W8A8 is effectively lossless** for Llama 3.1 8B, 70B and 405B across academic and real-world benchmarks | Kurtic et al. (Red Hat / Neural Magic), *"Give Me BF16 or Give Me Death?"*, 2024 — over 500K evaluations |
| **INT8 W8A8**, properly tuned, loses only ~1–3% | Same study |
| **INT4 weight-only (GPTQ/AWQ) is competitive with INT8 W8A8**, and the best choice for low-batch (latency-oriented) serving on mid-range GPUs; W8A8 wins for high-throughput serving on H100-class hardware | Same study |
| **For a fixed memory budget, 4-bit is the sweet spot**: a bigger model at 4 bits beats a smaller one at 8 or 16 | Dettmers & Zettlemoyer, *k-bit inference scaling laws*, 2022 |
| **Llama 3 is more sensitive to low-bit quantization than Llama 2**, especially at ≤ 3 bits — models trained on more tokens pack more information per weight | Huang et al., *How Good Are Low-bit Quantized Llama 3 Models?*, 2024 |
| **Reasoning models degrade more**, especially on hard problems and below 4 bits, and quantized models can think *longer* | *Quantization Hurts Reasoning?*, 2025 |
| **Larger models tolerate quantization better**; MoE experts tolerate aggressive quantization well | Consistent across GPTQ/AWQ papers; community GGUF results on DeepSeek/Kimi MoEs |

The practical picture in 2026:

| Bits | Typical quality impact | Comment |
|:---:|---|---|
| 8 (FP8, INT8, Q8_0) | Indistinguishable from BF16 for most tasks | Default for datacenter serving |
| 5–6 (Q5_K_M, Q6_K, MXFP6) | Very close to BF16 | Safe local default when memory allows |
| 4 (AWQ, GPTQ, Q4_K_M, NVFP4) | Small loss; larger on small models, maths and long-context retrieval | The cost/quality sweet spot |
| 3 | Noticeable loss for ≤ 30B dense; acceptable for very large MoE | Use imatrix / i-quants; evaluate carefully |
| ≤ 2 | Large loss unless codebook methods or QAT | Research, or to make a giant MoE fit at all |

## Speed: The Bandwidth Ceiling

At batch size 1, decode speed is bounded by how fast the GPU can read the weights ([Chapter 17](./17_kv_cache_mechanics.md)). That makes speed predictable from first principles: **tokens/s ≲ memory bandwidth ÷ bytes read per token**.

Theoretical single-stream ceilings (all weights read once per token; INT4 includes group scales ≈ 4.25 bits; KV cache ignored; "—" = does not fit):

| GPU (bandwidth, memory) | 8B BF16 (16 GB) | 8B FP8 (8 GB) | 8B INT4 (4.3 GB) | 70B FP8 (71 GB) | 70B INT4 (37 GB) |
|---|:---:|:---:|:---:|:---:|:---:|
| RTX 4090 (1.0 TB/s, 24 GB) | 63 | 126 | 237 | — | — |
| A100 (2.0 TB/s, 80 GB) | 127 | 254 | 479 | tight | 54 |
| H100 SXM (3.35 TB/s, 80 GB) | 209 | 417 | 787 | tight | 90 |
| B200 (8 TB/s, 192 GB) | 498 | 996 | 1,880 | 113 | 214 |
| Apple M4 Max (546 GB/s, 128 GB unified) | 34 | 68 | 128 | 8 | 15 |

Good kernels reach roughly 50–80% of these ceilings. Three consequences:

1. **Halving bytes nearly doubles single-stream speed** — weight-only 4-bit can be ~3× faster than BF16 at batch 1.
2. **At large batches the ceiling changes**: decode becomes compute-bound, and only formats with low-precision *activations* (FP8, FP4) speed up the matmul itself. That is why datacenters serve FP8/FP4 while laptops run 4-bit weight-only GGUFs.
3. **Long contexts add KV bytes** to every step; quantizing weights alone cannot fix a cache-dominated workload ([Chapter 18](./18_kv_cache_optimization.md)).

Check fit and the ceiling for your own model, format and hardware:

<div class="lab" data-lab="fit"><noscript>Interactive lab: does a model fit on a GPU at a given precision and context, and what is its bandwidth-bound decode speed? Enable JavaScript to use it.</noscript></div>

## Decision Guide

<div class="diagram">
<div class="diagram-title">Which Quantization Should I Use?</div>
<div class="flow">
  <div class="flow-node accent wide">What is the deployment target?</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Datacenter GPUs serving many users → FP8 first, FP4 on Blackwell</div>
  <div class="flow-node green wide">Single GPU / workstation, low batch → 4-bit weight-only (AWQ / GPTQ), or FP8 if it fits</div>
  <div class="flow-node cyan wide">Laptop, Mac, CPU, edge → GGUF (Q4_K_M → Q6_K) or MLX 4–8-bit</div>
  <div class="flow-node orange wide">Fine-tuning on limited memory → NF4 base + LoRA (QLoRA), then re-quantize the merged model</div>
</div>
</div>

### Datacenter Serving

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Default</div>
    <div class="card-desc">FP8 W8A8 (dynamic per-token activations) + FP8 KV cache on Hopper or newer. Use the lab's own FP8 checkpoint if one exists.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Maximum throughput (Blackwell)</div>
    <div class="card-desc">NVFP4 weights and activations via ModelOpt or LLM Compressor; validate reasoning and long-context tasks before rollout.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Latency-sensitive, small batch</div>
    <div class="card-desc">INT4 weight-only (AWQ/GPTQ with Marlin/Machete kernels), plus speculative decoding.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Long-context heavy</div>
    <div class="card-desc">Quantize the KV cache (FP8) before squeezing weights further; consider architecture (MLA, hybrid) when choosing the model.</div>
  </div>
</div>
</div>

### Local and Edge

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">llama.cpp / Ollama / LM Studio</div>
    <div class="card-desc">Best balance: Q4_K_M or IQ4_XS. Higher quality: Q5_K_M / Q6_K. Giant MoE on a workstation: mixed 2–3-bit expert quants with an imatrix.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Rough memory guide (weights + modest context)</div>
    <div class="card-desc">8 GB: 7–8B at Q4. 16 GB: 8B at Q8 or 14B at Q4. 32 GB: 30B-class at Q4–Q6. 64 GB: 70B at Q4. 128 GB+: 120B-class MoE (e.g., gpt-oss-120b in MXFP4) or 70B at Q8.</div>
  </div>
</div>
</div>

## Practical Tips

1. **Prefer the lab's own low-precision release** (FP8, MXFP4, QAT INT4) — it was calibrated or trained by people who know the model.
2. **FP8 before 4-bit** in the datacenter; it is simple and nearly free.
3. **Keep sensitive pieces at higher precision**: LM head, embeddings, first/last layers, MoE router — most tools do this by default.
4. **Calibrate on representative data**: chat-formatted text for chat models, code for coding models, the languages you serve.
5. **Evaluate the tasks you care about**, not just perplexity: maths and code, long-context retrieval, tool-calling format adherence, and — for reasoning models — accuracy *and* output length.
6. **Measure speed on your traffic**: batch size, prompt/output lengths and concurrency decide whether weight-only or W8A8 wins.

## Benchmarking Properly

Two layers: a quick windowed **perplexity** check to catch broken quantization, then **task evaluations**.

```python
"""Windowed perplexity over the concatenated test set (non-overlapping windows; a stride < ctx
with overlapping labels masked to -100 gives the slower sliding-window variant)."""
import math, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(model_id, ctx=2048, stride=2048):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto").eval()
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids                 # [1, N]
    nll, count = 0.0, 0
    for start in range(0, ids.size(1) - 1, stride):
        chunk = ids[:, start:start + ctx].to(model.device)
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss                 # mean over chunk.size(1) - 1 predictions
        nll += loss.item() * (chunk.size(1) - 1)
        count += chunk.size(1) - 1
    return math.exp(nll / count)

for m in ["meta-llama/Llama-3.1-8B-Instruct", "./Llama-3.1-8B-Instruct-W4A16"]:
    print(m, round(perplexity(m), 3))
```

Compare perplexities only between variants of the *same* model and tokenizer, with the same context length. Then run task suites — `lm-evaluation-harness` or `lighteval` for academic benchmarks, plus your own held-out prompts — on the original and quantized model side by side. A perplexity increase of a few hundredths is typical for good 4-bit; a jump of 1+ means something went wrong (bad calibration, quantized LM head, wrong kernel).

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Quantized reasoning models sometimes compensate for lost precision by thinking for longer — so a 4-bit model that is 2× faster per token but writes 30% more tokens is less of a win than it looks. What metric would you put on a dashboard to capture the real cost per <em>solved</em> task?</p>
</div>

## Key Takeaways

- Trust systematic studies over one-off tables: FP8 ≈ lossless; tuned INT8 ~1–3% loss; INT4 weight-only competitive, especially for low-batch serving.
- Bigger models, and MoE experts, tolerate quantization better; heavily trained small models and reasoning models are more sensitive.
- Single-stream speed is bandwidth ÷ bytes per token; at high batch only low-precision activations (FP8/FP4) speed up compute.
- Datacenter: FP8 (FP4 on Blackwell). Single GPU: AWQ/GPTQ. Local: GGUF Q4_K_M–Q6_K or MLX. Fine-tuning: QLoRA.
- Benchmark with windowed perplexity as a sanity check, then task evals — including output length for reasoning models.

## Check Yourself

<details class="check">
<summary>What is the bandwidth-bound decode ceiling for a 70B INT4 model on an RTX 5090 with 1.79 TB/s and 32 GB?</summary>
<p>It does not fit: 70.6B × 4.25 bits ≈ 37.5 GB of weights exceeds 32 GB. You would need ~3.5 bits per weight (and some room for KV cache), or two GPUs.</p>
</details>

<details class="check">
<summary>Why might INT4 weight-only beat FP8 for a chatbot with one user, but lose for a service with 200 concurrent users?</summary>
<p>With one user, decode is bandwidth-bound and INT4 reads half FP8's bytes. With 200 concurrent sequences, weight reads are amortised and the matmul becomes compute-bound; FP8 W8A8 uses FP8 tensor cores while W4A16 dequantizes to BF16 and multiplies at BF16 speed.</p>
</details>

<details class="check">
<summary>Two papers report WikiText-2 perplexity 6.1 and 5.8 for the same FP16 model. Name three things that could explain the difference.</summary>
<p>Context length / stride (longer windows give lower perplexity), BOS handling and text joining, and the evaluation split or preprocessing; also tokenizer version and whether the loss is averaged per window or per token.</p>
</details>

<details class="check">
<summary>Per Dettmers & Zettlemoyer, should you run a 14B model at 8 bits or a 30B model at ~4 bits in the same memory?</summary>
<p>The 30B at 4 bits: for a fixed total bit budget, more parameters at 4-bit precision generally beat fewer parameters at higher precision (down to about 4 bits, below which quality falls off quickly).</p>
</details>

<details class="check">
<summary>What should you check besides accuracy when quantizing a reasoning model?</summary>
<p>Average output length (thinking tokens), since a quantized model may reason longer and erase the speed gain; accuracy on the hardest problems (where degradation concentrates); and format adherence for tool calls.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization | Kurtic et al., 2024 | [arxiv.org/abs/2411.02355](https://arxiv.org/abs/2411.02355) | Large-scale FP8/INT8/INT4 evaluation |
| The case for 4-bit precision: k-bit Inference Scaling Laws | Dettmers & Zettlemoyer, 2022 | [arxiv.org/abs/2212.09720](https://arxiv.org/abs/2212.09720) | 4-bit as the memory-optimal precision |
| How Good Are Low-bit Quantized LLaMA3 Models? | Huang et al., 2024 | [arxiv.org/abs/2404.14047](https://arxiv.org/abs/2404.14047) | Llama 3's sensitivity to low bits |
| Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models | Liu et al., 2025 | [arxiv.org/abs/2504.04823](https://arxiv.org/abs/2504.04823) | Reasoning-model sensitivity, longer outputs |
| lm-evaluation-harness | EleutherAI | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Task evaluation |
| Perplexity of fixed-length models | Hugging Face docs | [huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity) | Sliding-window protocol |
| NVIDIA H100 / B200 datasheets | NVIDIA, 2023–25 | [nvidia.com/en-us/data-center](https://www.nvidia.com/en-us/data-center/h100/) | Memory bandwidth figures |

---

**Next:** [Chapter 23 — Knowledge Distillation & QAD](./23_knowledge_distillation.md) — making smaller (and lower-precision) models learn from bigger ones.

[← Back to Table of Contents](./README.md)
