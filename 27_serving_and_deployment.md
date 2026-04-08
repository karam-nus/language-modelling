[← Back to Table of Contents](./README.md)

# Chapter 27 — Serving & Deployment

> *"Training a great model is half the battle. Serving it efficiently — low latency, high throughput, at reasonable cost — is the other half."*

## Inference Frameworks Landscape

<div class="diagram">
<div class="diagram-title">LLM Serving Stack</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">vLLM</div>
    <div class="card-desc">PagedAttention, continuous batching, OpenAI-compatible API. The default choice for GPU serving.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">TGI</div>
    <div class="card-desc">HuggingFace's production inference server. Rust-based, watermark support, HF Hub integration.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">TensorRT-LLM</div>
    <div class="card-desc">NVIDIA's optimized backend. Custom CUDA kernels, FP8, in-flight batching. Fastest on NVIDIA hardware.</div>
  </div>
</div>
<div class="diagram-grid cols-3" style="margin-top: 0.5rem;">
  <div class="diagram-card orange">
    <div class="card-title">llama.cpp / Ollama</div>
    <div class="card-desc">CPU + Apple Silicon inference via GGUF. Run models locally on laptops and desktops.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">SGLang</div>
    <div class="card-desc">Structured generation + RadixAttention. Fast for constrained output (JSON, regex).</div>
  </div>
  <div class="diagram-card red">
    <div class="card-title">ONNX Runtime</div>
    <div class="card-desc">Cross-platform inference. Export PyTorch → ONNX → optimized runtime. CPU/GPU/NPU.</div>
  </div>
</div>
</div>

## vLLM

The most popular open-source LLM serving engine. Key innovations from [Chapter 15](./15_kv_cache_optimization.md):

- **PagedAttention**: Non-contiguous KV-cache with virtual memory → no fragmentation
- **Continuous batching**: New requests join mid-batch instead of waiting
- **Prefix caching**: Reuse KV-cache for shared prefixes (system prompts)
- **Speculative decoding**: Use a small draft model for faster generation
- **Quantization**: GPTQ, AWQ, FP8, bitsandbytes support

```python
# vLLM — Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=2,         # split across 2 GPUs
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(["Explain transformers in one paragraph."], params)
print(outputs[0].outputs[0].text)
```

```bash
# vLLM — OpenAI-compatible server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --port 8000

# Use with any OpenAI SDK client
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
        "max_tokens": 256
    }'
```

## Text Generation Inference (TGI)

HuggingFace's production server, used by the Inference API:

```bash
# TGI via Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --quantize awq \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 4096

# Query
curl http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{"inputs": "What is attention?", "parameters": {"max_new_tokens": 200}}'
```

## TensorRT-LLM

NVIDIA's highest-performance option, using custom CUDA kernels:

```python
# Build TensorRT-LLM engine
# Step 1: Convert HF model to TensorRT-LLM checkpoint
python convert_checkpoint.py \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./trt_ckpt \
    --dtype bfloat16 \
    --tp_size 2

# Step 2: Build engine with optimizations
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin bfloat16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --use_fp8_context_fmha enable

# Step 3: Serve via Triton Inference Server
```

## llama.cpp & Ollama

For local deployment on CPU and Apple Silicon:

```bash
# Ollama — simplest way to run models locally
ollama pull llama3.1:8b-instruct-q4_K_M
ollama run llama3.1:8b-instruct-q4_K_M

# Python client
import ollama
response = ollama.chat(model='llama3.1:8b-instruct-q4_K_M', messages=[
    {'role': 'user', 'content': 'Explain attention in one sentence.'}
])
```

```bash
# llama.cpp — direct usage with GGUF files
./llama-cli \
    -m models/llama-3.1-8b-instruct-Q4_K_M.gguf \
    -p "Explain attention:" \
    -n 256 \
    -ngl 99          # offload all layers to GPU (Metal on macOS)
```

## Framework Comparison

| Feature | vLLM | TGI | TensorRT-LLM | llama.cpp | SGLang |
|---------|------|-----|---------------|-----------|--------|
| **PagedAttention** | ✅ | ✅ | ✅ (custom) | ✗ | ✅ |
| **Continuous batching** | ✅ | ✅ | ✅ | ✗ | ✅ |
| **Tensor parallelism** | ✅ | ✅ | ✅ | ✗ | ✅ |
| **Quantization** | GPTQ, AWQ, FP8, bnb | GPTQ, AWQ, EETQ | FP8, INT8, INT4 | GGUF (2-8 bit) | GPTQ, AWQ, FP8 |
| **Speculative decode** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Structured output** | Via outlines | ✅ (grammar) | ✗ | ✅ (grammar) | ✅ (native) |
| **OpenAI API** | ✅ | ✅ | Via Triton | ✅ | ✅ |
| **CPU inference** | ✗ | ✗ | ✗ | ✅ | ✗ |
| **Apple Silicon** | ✗ | ✗ | ✗ | ✅ (Metal) | ✗ |
| **Best for** | General GPU serving | HF ecosystem prod | Maximum NVIDIA perf | Local / edge | Structured output |

## Latency vs Throughput

These are often conflicting goals:

<div class="diagram">
<div class="diagram-title">Latency vs Throughput Tradeoffs</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Optimize for Latency</div>
    <ul>
      <li>Small batch size (1–4)</li>
      <li>Tensor parallelism across GPUs</li>
      <li>Speculative decoding</li>
      <li>FP8 or INT4 quantization</li>
      <li>Use case: chatbot, real-time</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Optimize for Throughput</div>
    <ul>
      <li>Large batch size (64–256)</li>
      <li>Continuous batching</li>
      <li>Prefix caching for shared prompts</li>
      <li>Maximize GPU utilization</li>
      <li>Use case: batch processing, APIs</li>
    </ul>
  </div>
</div>
</div>

Key metrics:
- **Time to First Token (TTFT)**: How long until the first token appears (prefill latency)
- **Time per Output Token (TPOT)**: Average decode latency per token
- **Tokens per Second (TPS)**: Total throughput across all concurrent requests

## Deployment Checklist

1. **Choose quantization**: W4A16 (AWQ/GPTQ) if quality-sensitive, FP8 if hardware supports it, GGUF for CPU/Apple
2. **Choose framework**: vLLM for most GPU deployments, Ollama for local
3. **Set max_model_len**: Don't set higher than needed — it reserves KV-cache memory
4. **Enable prefix caching**: If system prompt is shared across requests
5. **Monitor**: Track TTFT, TPOT, queue depth, GPU utilization, KV-cache usage
6. **Load test**: Use benchmarking tools to find the throughput/latency sweet spot

## What's Next

From infrastructure, we now move to the **science of scale**. The next chapter explores **scaling laws** — the mathematical relationships between compute, data, parameters, and performance.

[← Previous: Chapter 26 — Evaluation & Benchmarks](./26_evaluation_and_benchmarks.md) · **Next: [Chapter 28 — Scaling Laws & Emergent Abilities →](./28_scaling_laws.md)**

---

*Last updated: April 2026*
