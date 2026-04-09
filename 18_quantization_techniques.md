---
title: "Chapter 18 — Quantization Techniques: The Full Landscape"
---

[← Back to Table of Contents](./README.md)

# Chapter 18 — Quantization Techniques: The Full Landscape

> *"Every quantization method is a different answer to the same question: how do you preserve accuracy while dramatically reducing precision?"*

## Overview

This chapter covers every major quantization technique for LLMs, organized by what they quantize.

<div class="diagram">
<div class="diagram-title">Quantization Technique Map</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Weight-Only Quantization</div>
    <div class="card-desc">GPTQ, AWQ, GGUF, bitsandbytes, QuIP#, AQLM, HQQ, EXL2</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Weight + Activation Quantization</div>
    <div class="card-desc">SmoothQuant, QServe/QoQ, Atom, SpinQuant, FP8</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">QAT (Train with Quantization)</div>
    <div class="card-desc">LLM-QAT, BitNet, OneBit</div>
  </div>
</div>
</div>

---

## Weight-Only Quantization

### GPTQ (Frantar et al., 2022)

GPTQ applies **Optimal Brain Quantization (OBQ)** layer by layer — it quantizes each weight column while compensating the error in remaining columns using the inverse Hessian of the layer's output.

**Key ideas:**
- Quantize weights one column at a time using second-order (Hessian) information
- After quantizing column j, update remaining columns to minimize total output error
- Uses a calibration dataset (~128 samples) to estimate the Hessian

**Typical config**: INT4, group_size=128, symmetric quantization.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",            # calibration dataset
    desc_act=True,           # order columns by activation magnitude
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=quantization_config,
    device_map="auto",
)
model.save_pretrained("Llama-3.1-8B-GPTQ-INT4")
```

### AWQ (Lin et al., 2023)

**Activation-Aware Weight Quantization**: not all weight channels are equally important. Channels that correspond to large activations (salient channels) should be preserved with higher precision.

**Key ideas:**
- Identify salient weight channels by observing activation magnitudes on calibration data
- Scale salient channels up before quantization, scale down after (equivalent to multiplying activations by inverse)
- This moves quantization difficulty from sensitive channels to insensitive ones
- No Hessian computation — faster than GPTQ

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

model.quantize(
    tokenizer,
    quant_config={"w_bit": 4, "q_group_size": 128, "version": "gemm"},
)
model.save_quantized("Llama-3.1-8B-AWQ")
```

### GGUF / llama.cpp Quantization

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp and Ollama. It supports many quantization variants with different quality/size tradeoffs:

| GGUF Type | Bits (effective) | Description |
|-----------|:---:|-------------|
| **Q2_K** | ~2.6 | Extreme compression, significant quality loss |
| **Q3_K_M** | ~3.4 | Low quality but very small |
| **Q4_0** | 4.0 | Simple 4-bit, no group scaling |
| **Q4_K_M** | ~4.6 | 4-bit with group scales and mins. **Best quality/size.** |
| **Q5_K_M** | ~5.5 | Near-FP16 quality at ~60% size |
| **Q6_K** | ~6.6 | Near-lossless |
| **Q8_0** | 8.0 | Essentially lossless |

```bash
# Convert and quantize with llama.cpp
python convert_hf_to_gguf.py meta-llama/Llama-3.1-8B --outtype f16
./llama-quantize Llama-3.1-8B-F16.gguf Llama-3.1-8B-Q4_K_M.gguf Q4_K_M
```

### bitsandbytes

The library behind QLoRA. Two main modes:

| Mode | Method | Bits | Use Case |
|------|--------|:----:|----------|
| **LLM.int8()** | Mixed decomposition — FP16 for outlier channels, INT8 for rest | 8 | Inference without quality loss |
| **NF4** | Normal Float 4-bit with double quantization | 4 | QLoRA fine-tuning, inference |

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit NF4 with double quantization (QLoRA config)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", quantization_config=bnb_config, device_map="auto"
)
```

### Other Weight-Only Methods

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card cyan">
    <div class="card-title">QuIP# (Tseng et al., 2024)</div>
    <div class="card-desc">Incoherence processing + lattice codebook quantization. State-of-the-art at 2-bit. Randomized Hadamard transforms make weights "incoherent" (uniform magnitude).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">AQLM (Egiazarian et al., 2024)</div>
    <div class="card-desc">Additive quantization with learned codebooks. Multiple codebooks are summed to approximate each weight group. Best at extreme compression (2-bit).</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">HQQ (Badri & Shaji, 2023)</div>
    <div class="card-desc">Half-Quadratic Quantization. Zero-shot — no calibration data needed. Very fast quantization. Good quality at 4-bit.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">EXL2 (turboderp, 2023)</div>
    <div class="card-desc">ExLlamaV2 format. Dynamic per-layer bit allocation — important layers get more bits. Optimal Pareto frontier for quality vs size.</div>
  </div>
</div>
</div>

---

## Weight + Activation Quantization

### SmoothQuant (Xiao et al., 2022)

The key insight: **weights are easy to quantize; activations are hard** (due to outliers). SmoothQuant migrates difficulty from activations to weights by applying per-channel scaling:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

The scaling factor $s$ balances the quantization difficulty between X and W per channel.

<div class="diagram">
<div class="diagram-title">SmoothQuant — Migrating Outlier Difficulty</div>
<div class="flow">
  <div class="flow-node accent wide">Original: X has outliers, W is smooth <small>activations hard to quantize</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Per-channel scaling: divide X by s, multiply W by s <small>mathematically equivalent</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Smoothed: X̂ is smooth, Ŵ slightly harder <small>both are now quantizable</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">W8A8: INT8 weights × INT8 activations → INT8 matmul <small>2× speedup over FP16</small></div>
</div>
</div>

### QServe / QoQ (W4A8KV4)

**QServe** (Lin et al., 2024) achieves the holy grail: 4-bit weights, 8-bit activations, AND 4-bit KV-cache in a single framework. Key technique: **QoQ** (Quattuor-Octo-Quattuor) progressive quantization.

| Component | Precision | Method |
|-----------|:---------:|--------|
| Weights | W4 | Group-quantized INT4 |
| Activations | A8 | Per-token dynamic INT8 |
| KV-Cache | KV4 | Per-channel key + per-token value INT4 |

### FP8 Inference

On H100+ GPUs, FP8 inference is native and requires minimal changes:

```python
from transformers import AutoModelForCausalLM

# FP8 inference with transformers
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float8_e4m3fn,  # FP8 weights
    device_map="auto",
)

# Or with vLLM
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B", dtype="float16", quantization="fp8")
```

---

## Quantization-Aware Training (QAT)

### BitNet (Wang et al., 2023)

Trains 1-bit (ternary: -1, 0, +1) transformers from scratch. No floating-point weights at all.

<div class="diagram">
<div class="diagram-title">BitNet — Ternary Weight Transformer</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Standard Transformer</div>
    <ul>
      <li>Weights: FP16 (16 bits per param)</li>
      <li>Matmul: FP16 × FP16 → FP16</li>
      <li>Memory: 2 bytes per parameter</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">BitNet b1.58</div>
    <ul>
      <li>Weights: {-1, 0, +1} (1.58 bits)</li>
      <li>Matmul: replaces multiply with add/subtract</li>
      <li>Memory: ~0.2 bytes per parameter</li>
    </ul>
  </div>
</div>
</div>

BitNet can match FP16 quality at 3B+ params while being dramatically more efficient. The challenge: requires training from scratch.

---

## Master Comparison Table

| Method | Target | Bits | PTQ/QAT | Calibration | Speed vs FP16 | Quality (vs FP16) | GPU Required | Best For |
|--------|--------|:----:|:-------:|:-----------:|:---:|:---:|:---:|----------|
| **GPTQ** | Weights | 4 | PTQ | ~128 samples | ~3–4× | -0.1–0.5 PPL | NVIDIA | GPU inference (AutoGPTQ, vLLM) |
| **AWQ** | Weights | 4 | PTQ | ~128 samples | ~3–4× | -0.1–0.3 PPL | NVIDIA | GPU inference (vLLM, TGI) |
| **GGUF Q4_K_M** | Weights | ~4.6 | PTQ | None | ~2–3× (CPU) | -0.2–0.5 PPL | CPU/GPU | llama.cpp, Ollama |
| **bitsandbytes NF4** | Weights | 4 | PTQ | None | ~2× | -0.3–0.5 PPL | NVIDIA | QLoRA fine-tuning |
| **QuIP#** | Weights | 2 | PTQ | ~128 samples | ~2–3× | -0.5–1.0 PPL | NVIDIA | Extreme compression |
| **AQLM** | Weights | 2 | PTQ | Learnable | ~2× | -0.3–0.8 PPL | NVIDIA | 2-bit with good quality |
| **HQQ** | Weights | 4 | PTQ | None (zero-shot) | ~3× | -0.2–0.5 PPL | NVIDIA | Fast quantization |
| **EXL2** | Weights | 2–8 | PTQ | ~128 samples | ~3–4× | Dynamic (optimal) | NVIDIA | ExLlamaV2, Pareto optimal |
| **SmoothQuant** | W+A | 8 | PTQ | ~512 samples | ~2× | -0.0–0.1 PPL | NVIDIA | W8A8 server inference |
| **QServe** | W+A+KV | 4/8/4 | PTQ | Calibration | ~3–4× | -0.1–0.3 PPL | NVIDIA | Full-stack quantization |
| **FP8** | W+A | 8 | PTQ | Per-tensor | ~2× | Negligible | H100+ | Native H100 inference |
| **BitNet** | Weights | 1.58 | QAT | Full training | ~10×+ | Comparable at 3B+ | Any | From-scratch training |

## Dequantization at Runtime

When a layer performs its computation, INT4 weights are dequantized on the fly:

<div class="diagram">
<div class="diagram-title">INT4 Dequantization Flow</div>
<div class="flow">
  <div class="flow-node accent wide">Packed INT4 weights: [d_out, d_in/8] <small>8 values per int32</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Unpack: extract 4-bit values → [d_out, d_in] <small>INT4 integers</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Dequantize: (int4 - zero) × scale → FP16 <small>[d_out, d_in] @ FP16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Matmul: FP16 weights × FP16 activations → FP16 output</div>
</div>
</div>

Efficient implementations (GPTQ, AWQ kernels) fuse unpacking + dequantization + matmul into a single GPU kernel, avoiding materializing the full FP16 weight matrix.

## What's Next

With all techniques covered, the next chapter provides **benchmarks and a selection guide** — empirical comparisons of quality, speed, and memory across methods, plus a decision flowchart for choosing the right approach.

[← Previous: Chapter 17 — Quantization Fundamentals](./17_quantization_fundamentals.md) · **Next: [Chapter 19 — Quantization Benchmarks →](./19_quantization_benchmarks.md)**

---

*Last updated: April 2026*
