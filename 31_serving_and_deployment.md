---
title: "Chapter 31 — Serving & Deployment"
description: "Running LLMs for real users in 2026: the engine landscape (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX, on-device runtimes), orchestration (Dynamo, llm-d), latency and throughput metrics and SLOs, cost per million tokens from first principles, MoE serving with wide expert parallelism, multi-LoRA, routing, benchmarking and a deployment checklist."
related:
  - { id: inference, why: "Serving and inference in practice." }
  - { id: chips, why: "Which accelerators serve tokens cheapest." }
---

[← Back to Table of Contents](./README.md)

# Chapter 31 — Serving & Deployment

Serving is where everything in Chapters 16–24 meets a budget: a target latency, a traffic pattern and a price per GPU-hour. In 2026 serving cost — not training cost — dominates most organisations' LLM spend, and reasoning models that think for thousands of tokens have made it more important still.

> **The one-sentence version:** a serving system turns GPU-hours into tokens; your job is to maximise tokens per GPU-hour subject to latency targets, by choosing the engine, precision, parallelism, batching and caching that fit your traffic.

## The Engine Landscape

<div class="diagram">
<div class="diagram-title">LLM Serving Stack (2026)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">vLLM</div>
    <div class="card-desc">The most widely used open engine. V1 architecture, PagedAttention, prefix caching, chunked prefill, speculative decoding, FP8/FP4, multi-LoRA; NVIDIA, AMD, TPU, CPU and more.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">SGLang</div>
    <div class="card-desc">High-performance engine with RadixAttention, fast structured output, strong MoE and expert-parallel support; used for DeepSeek-style large deployments.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">TensorRT-LLM</div>
    <div class="card-desc">NVIDIA's engine: hand-tuned kernels, FP8/NVFP4, in-flight batching; a PyTorch-based workflow and <code>trtllm-serve</code> replaced most engine-building steps.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">llama.cpp · Ollama · LM Studio</div>
    <div class="card-desc">GGUF on CPUs, Apple Silicon and consumer GPUs. The local-inference standard.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">MLX-LM</div>
    <div class="card-desc">Apple's array framework; fast quantized inference and fine-tuning on Macs.</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-title">On-device</div>
    <div class="card-desc">ExecuTorch, MLC LLM, ONNX Runtime GenAI, LiteRT; OS-level models such as Apple's on-device foundation model and Gemini Nano.</div>
  </div>
</div>
</div>

Above single engines sits **orchestration**: **NVIDIA Dynamo** (2025) and **llm-d** (Kubernetes-native, 2025) coordinate many engine instances — disaggregated prefill and decode pools, KV-cache-aware routing, KV offloading to CPU and SSD, autoscaling ([Chapter 18](./18_kv_cache_optimization.md)).

## vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          tensor_parallel_size=2,           # shard across 2 GPUs
          max_model_len=16384,              # don't reserve cache for lengths you won't serve
          gpu_memory_utilization=0.90,
          kv_cache_dtype="fp8")
outs = llm.chat([[{"role": "user", "content": "Explain transformers in one paragraph."}]],
                SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512))
print(outs[0].outputs[0].text)
```

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 --max-model-len 16384 --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser llama3_json   # OpenAI-style tool calling

curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128 }'
```

Reasoning models need a **reasoning parser** (e.g., `--reasoning-parser deepseek_r1` or `qwen3`) so the API returns thinking and final answer in separate fields.

## SGLang, TensorRT-LLM and Local Engines

```bash
# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --tp 2 --port 30000

# TensorRT-LLM (PyTorch workflow, OpenAI-compatible)
trtllm-serve meta-llama/Llama-3.1-8B-Instruct --tp_size 1 --port 8000

# Ollama
ollama run llama3.1:8b-instruct-q4_K_M

# llama.cpp server (OpenAI-compatible), all layers on GPU/Metal
./llama-server -m llama-3.1-8b-instruct-Q4_K_M.gguf -ngl 99 -c 16384 --port 8080
```

## Framework Comparison

| Feature | vLLM | SGLang | TensorRT-LLM | llama.cpp | MLX-LM |
|---------|------|-----|---------------|-----------|--------|
| Paged KV / continuous batching | ✅ | ✅ | ✅ | Partial (server slots) | Basic |
| Prefix caching | ✅ hashed blocks | ✅ radix tree | ✅ | ✅ (per slot) | ✅ |
| Tensor / expert parallelism | ✅ / ✅ | ✅ / ✅ (wide EP) | ✅ / ✅ | Layer split across GPUs | Distributed (MLX) |
| Precision | FP8, NVFP4/MXFP4, GPTQ, AWQ, GGUF (limited) | FP8, FP4, GPTQ, AWQ | FP8, NVFP4, INT4/INT8 | GGUF 1.5–8-bit | 3–8-bit MLX quants |
| Speculative decoding | EAGLE-3, MTP, n-gram | EAGLE-3, MTP | EAGLE, MTP, Medusa | Draft models, n-gram | Draft models |
| Structured output | xgrammar / guidance | xgrammar (native) | xgrammar | GBNF grammars | Via libraries |
| Hardware | NVIDIA, AMD, TPU, CPU, others | NVIDIA, AMD | NVIDIA | Everything | Apple Silicon |
| Best for | General GPU serving | High-throughput, MoE, agents | Peak NVIDIA performance | Local / edge | Macs |

## Metrics and SLOs

- **TTFT** (time to first token) — dominated by queueing and prefill.
- **TPOT / ITL** (time per output token / inter-token latency) — dominated by decode batch size and bandwidth.
- **End-to-end latency** = TTFT + output tokens × TPOT. A reasoning model writing 5,000 tokens at 20 ms each takes 100 s, whatever its TTFT.
- **Throughput** — output tokens per second per GPU (or per dollar).
- **Goodput** — throughput of requests that *meet* their latency SLOs; the honest metric when tails matter.
- Always report **percentiles** (p50, p95, p99), not just means.

<div class="diagram">
<div class="diagram-title">Latency vs Throughput</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Optimise for latency</div>
    <ul>
      <li>Small batches; high tensor parallelism</li>
      <li>Speculative decoding (EAGLE-3, MTP)</li>
      <li>Low-bit weights to cut bytes per token</li>
      <li>Interactive chat, coding assistants, voice</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Optimise for throughput</div>
    <ul>
      <li>Large batches; continuous batching</li>
      <li>FP8/FP4 activations (compute-bound regime)</li>
      <li>Prefix caching, disaggregation, wide EP</li>
      <li>Batch jobs, RL rollouts, synthetic data, cheap APIs</li>
    </ul>
  </div>
</div>
</div>

## What a Million Tokens Costs

Serving cost follows directly from throughput:

$$\text{\$ per million output tokens} = \frac{\text{GPU \$/hour} \times 10^6}{\text{output tokens/s per GPU} \times 3600}$$

At $2.50 per H100-hour, a deployment sustaining 2,500 output tokens/s per GPU costs about **$0.28 per million tokens**; at batch 1 and 100 tokens/s it costs **$6.94**. The 25× gap is batching. This is why API prices for the same model vary so much between providers, why batch APIs are discounted, and why prefix-cache hits are billed at a fraction of normal input tokens. Explore the trade-off:

<div class="lab" data-lab="servecost"><noscript>Interactive lab: cost per million tokens from GPU price, batch size and per-sequence speed, with the latency each choice implies. Enable JavaScript to use it.</noscript></div>

## Serving Large MoE Models

A 671B–2.8T MoE has few active parameters per token but enormous weights. The 2025 pattern, described in DeepSeek's account of serving V3/R1:

- **Disaggregate** prefill and decode ([Chapter 18](./18_kv_cache_optimization.md)).
- **Wide expert parallelism**: spread experts across many GPUs (DeepSeek reported 32-GPU prefill units and 144–320-GPU decode units), so each GPU holds few experts and serves a large effective batch; all-to-all kernels (DeepEP) move tokens between them.
- **Load-balance experts** at runtime by replicating hot experts.
- **Overlap** communication with computation using two micro-batches.

DeepSeek reported that this let it serve both models at a theoretical cost far below its API prices — the economics of sparse models at scale.

## Multi-Tenancy, Routing and Caching

- **Multi-LoRA**: serve hundreds of fine-tunes on one base model with batched adapter kernels ([Chapter 14](./14_peft_lora_and_variants.md)).
- **Model routing**: send easy requests to small, cheap models and hard ones to large or reasoning models; GPT-5's router between fast and thinking modes is a product-scale example.
- **Prompt caching**: structure prompts so stable content (system prompt, tools, documents) comes first and can be reused across requests.
- **Semantic caching**: return a stored answer for near-duplicate questions (with care — staleness and personalisation).

## Benchmark Before You Trust

Use load generators that replay realistic prompt/output length distributions and arrival rates: `vllm bench serve`, SGLang's `bench_serving`, NVIDIA GenAI-Perf, GuideLLM. Public continuous benchmarks such as SemiAnalysis's InferenceMAX compare hardware and engines on standard models, but your traffic mix decides your numbers.

## Deployment Checklist

1. **Pick the precision**: FP8 by default on Hopper+, NVFP4/MXFP4 on Blackwell after validation; AWQ/GPTQ for single-GPU latency; GGUF/MLX locally ([Chapter 22](./22_quantization_benchmarks.md)).
2. **Pick the engine**: vLLM or SGLang for GPUs; TensorRT-LLM for peak NVIDIA throughput; llama.cpp/Ollama/MLX locally.
3. **Size memory for the cache**: set `max_model_len` to what you serve; use FP8 KV; measure concurrency.
4. **Turn on prefix caching and chunked prefill**; add speculative decoding for latency-sensitive traffic.
5. **Match parallelism to the model**: TP within a node for dense models; EP (and disaggregation) for large MoE.
6. **Use the model's chat template, reasoning parser and tool parser** — most "the model is bad" reports are template bugs.
7. **Load-test with real traffic shapes**, set SLOs, monitor TTFT/TPOT percentiles, queue depth, KV-cache utilisation and prefix-hit rate.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Reasoning models made output tokens 10–100× more numerous per request, but providers price input and output tokens separately and often discount cached input heavily. If you were pricing an API, how would you charge for thinking tokens the user never sees — and how would that change what users ask for?</p>
</div>

## Key Takeaways

- vLLM and SGLang dominate open GPU serving; TensorRT-LLM leads on NVIDIA; llama.cpp, Ollama and MLX own local inference; Dynamo and llm-d orchestrate fleets.
- Measure TTFT, TPOT, throughput and goodput at p95/p99; reasoning models make TPOT × output length the dominant latency term.
- $ per million tokens = GPU $/hour ÷ (tokens/s × 3,600) × 10⁶; batching is the biggest lever.
- Large MoE serving relies on disaggregation, wide expert parallelism and communication overlap.
- Templates, parsers and cache settings are where deployments silently go wrong.

## Check Yourself

<details class="check">
<summary>A GPU costs $3/hour and sustains 1,200 output tokens/s. What is the cost per million output tokens?</summary>
<p>$3 × 10⁶ / (1,200 × 3,600) ≈ $0.69 per million tokens.</p>
</details>

<details class="check">
<summary>Why can a request with a fast TTFT still feel slow for a reasoning model?</summary>
<p>End-to-end latency is TTFT plus output tokens times TPOT; a model that thinks for 8,000 tokens at 25 ms each takes 200 s regardless of how quickly the first token arrives.</p>
</details>

<details class="check">
<summary>Why does setting <code>max_model_len</code> far above real usage hurt a vLLM deployment?</summary>
<p>It does not reserve memory per request (paging avoids that), but it caps how many long requests can be admitted, raises worst-case memory per request, can force smaller batch sizes and longer prefill chunks, and may enable expensive RoPE-scaling paths — set it to what you actually serve.</p>
</details>

<details class="check">
<summary>What is goodput, and why can it fall while throughput rises?</summary>
<p>The rate of completed requests that meet their latency SLOs. Increasing batch size raises raw tokens/s but also raises TPOT and queueing delay; past a point, more requests miss their SLOs, so useful (good) throughput drops.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| vLLM documentation | vLLM project | [docs.vllm.ai](https://docs.vllm.ai/) | Engine features, serving flags |
| SGLang | Zheng et al., 2023; docs | [docs.sglang.ai](https://docs.sglang.ai/) | RadixAttention, MoE serving |
| TensorRT-LLM | NVIDIA | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | trtllm-serve, PyTorch workflow |
| NVIDIA Dynamo | NVIDIA, 2025 | [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) | Disaggregated orchestration |
| llm-d | llm-d project, 2025 | [llm-d.ai](https://llm-d.ai/) | Kubernetes-native distributed inference |
| DeepSeek-V3/R1 Inference System Overview | DeepSeek, 2025 | [github.com/deepseek-ai/open-infra-index](https://github.com/deepseek-ai/open-infra-index) | Wide EP, disaggregation, cost analysis |
| DistServe | Zhong et al., 2024 | [arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670) | Goodput, prefill/decode separation |
| llama.cpp | ggml-org | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Local serving |
| MLX-LM | Apple | [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple Silicon inference |

---

**Next:** [Chapter 32 — Scaling Laws & Emergent Abilities](./32_scaling_laws.md) — the empirical laws that decided how big all of these models are.

[← Back to Table of Contents](./README.md)
