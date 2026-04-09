---
title: "Chapter 19 — Quantization Benchmarks & Selection Guide"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Quantization Benchmarks & Selection Guide

> *"The best quantization method depends on your constraints — there is no single winner. This chapter helps you choose."*

## Perplexity vs Bit-Width

Perplexity (PPL) on WikiText-2 is the standard measure of quantization quality loss. Lower is better.

### LLaMA-3-8B Quantization Comparison

| Method | Bits | PPL (↓) | Δ PPL | Model Size | Tokens/sec (A100) |
|--------|:----:|:-------:|:-----:|:----------:|:-----------------:|
| FP16 (baseline) | 16 | 6.14 | — | 16.0 GB | 42 |
| GPTQ | 8 | 6.15 | +0.01 | 8.5 GB | 68 |
| AWQ | 4 | 6.24 | +0.10 | 4.7 GB | 105 |
| GPTQ | 4 | 6.27 | +0.13 | 4.7 GB | 100 |
| HQQ | 4 | 6.31 | +0.17 | 4.7 GB | 98 |
| GGUF Q4_K_M | ~4.6 | 6.22 | +0.08 | 5.0 GB | 85 |
| GGUF Q5_K_M | ~5.5 | 6.16 | +0.02 | 5.7 GB | 75 |
| bitsandbytes NF4 | 4 | 6.35 | +0.21 | 4.5 GB | 55 |
| QuIP# | 2 | 7.15 | +1.01 | 2.8 GB | 60 |
| AQLM | 2 | 6.93 | +0.79 | 2.8 GB | 48 |
| GPTQ | 3 | 6.62 | +0.48 | 3.6 GB | 88 |

### Scaling with Model Size

Larger models tolerate quantization much better:

| Model | FP16 PPL | INT4 PPL | Δ PPL |
|-------|:--------:|:--------:|:-----:|
| LLaMA-3 8B | 6.14 | 6.24 | +0.10 |
| LLaMA-3 70B | 3.12 | 3.15 | +0.03 |
| Mixtral 8×7B | 3.84 | 3.88 | +0.04 |

**Rule of thumb**: INT4 quantization is nearly lossless for 70B+ models, and acceptably lossy for 7–13B models.

## Speed Benchmarks by GPU

Tokens per second (batch_size=1, seq_len=512, generating 128 tokens):

| Method | RTX 4090 | A100 80GB | H100 |
|--------|:--------:|:---------:|:----:|
| FP16 | 38 | 42 | 58 |
| GPTQ INT4 | 92 | 105 | 140 |
| AWQ INT4 | 95 | 110 | 145 |
| FP8 (native) | — | — | 115 |
| GGUF Q4_K_M (GPU) | 88 | 85 | 110 |
| bitsandbytes NF4 | 48 | 55 | 70 |

For llama.cpp on CPU (Apple M2 Ultra, 192 GB):

| GGUF Type | LLaMA-3-8B tok/s | LLaMA-3-70B tok/s |
|-----------|:---:|:---:|
| Q4_K_M | 45 | 12 |
| Q5_K_M | 38 | 10 |
| Q8_0 | 28 | 6 |
| F16 | 15 | OOM |

## Decision Flowchart

<div class="diagram">
<div class="diagram-title">Which Quantization Method Should I Use?</div>
<div class="flow">
  <div class="flow-node accent wide">What's your deployment target?</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">GPU server (NVIDIA) → Go to GPU section</div>
  <div class="flow-node green wide">CPU / Apple Silicon / local → Go to CPU section</div>
  <div class="flow-node orange wide">Fine-tuning (not inference) → bitsandbytes NF4 + QLoRA</div>
</div>
</div>

### GPU Deployment

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Maximum Throughput</div>
    <div class="card-desc">
      H100: FP8 native<br>
      A100/4090: AWQ INT4 + vLLM<br>
      Latency-sensitive: QServe W4A8KV4
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Maximum Quality</div>
    <div class="card-desc">
      GPTQ/AWQ INT4 (g=128)<br>
      GGUF Q5_K_M or Q6_K<br>
      FP8 on H100 (near-lossless)
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Fit Large Model on Small GPU</div>
    <div class="card-desc">
      70B on 24GB VRAM: AWQ INT4<br>
      8B on 8GB VRAM: GGUF Q4_K_M<br>
      Extreme: QuIP# or AQLM 2-bit
    </div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Long Context Serving</div>
    <div class="card-desc">
      AWQ/GPTQ INT4 + KV INT8 (vLLM)<br>
      QServe W4A8KV4<br>
      PagedAttention + prefix caching
    </div>
  </div>
</div>
</div>

### CPU / Local Deployment

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Ollama / llama.cpp</div>
    <div class="card-desc">
      Best balance: Q4_K_M<br>
      Higher quality: Q5_K_M<br>
      Fastest: Q4_0 (slightly worse quality)
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">GGUF Selection Guide</div>
    <div class="card-desc">
      8GB RAM: 7B @ Q4_K_M<br>
      16GB RAM: 7B @ Q8_0 or 13B @ Q4_K_M<br>
      32GB RAM: 70B @ Q4_K_M<br>
      64GB+ RAM: 70B @ Q6_K
    </div>
  </div>
</div>
</div>

## Practical Tips

1. **Start with AWQ INT4 or GPTQ INT4** for GPU inference — they have the best ecosystem support (vLLM, TGI, transformers)
2. **Use Q4_K_M for llama.cpp/Ollama** — best quality-per-bit for local deployment  
3. **NF4 + bitsandbytes for QLoRA** — it's the standard for fine-tuning on consumer GPUs
4. **FP8 on H100** if available — simplest with near-zero quality loss
5. **INT4 is the sweet spot** — below 4 bits, quality degrades noticeably; above 4 bits, the size savings don't justify the cost
6. **Benchmark on YOUR tasks** — perplexity doesn't perfectly predict downstream task performance. Always evaluate on your actual use case.

## Benchmark Script

```python
"""Minimal script to compare quantization methods on perplexity."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def compute_perplexity(model, tokenizer, dataset, max_samples=100, max_length=2048):
    total_loss = 0
    total_tokens = 0

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        inputs = tokenizer(
            sample["text"], return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()

# Usage
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
model = AutoModelForCausalLM.from_pretrained("your-quantized-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("your-quantized-model")
ppl = compute_perplexity(model, tokenizer, dataset)
print(f"Perplexity: {ppl:.2f}")
```

## What's Next

With quantization covered comprehensively, the next part of the guide shifts to the hardware that runs these models. The next chapter covers **GPU architecture** — SMs, Tensor Cores, memory hierarchy, and the roofline model.

[← Previous: Chapter 18 — Quantization Techniques](./18_quantization_techniques.md) · **Next: [Chapter 20 — GPU Architecture →](./20_gpu_architecture.md)**

---

*Last updated: April 2026*
