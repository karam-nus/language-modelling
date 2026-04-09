---
title: "Chapter 19 — Data Types & Numerical Precision"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Data Types & Numerical Precision

> *"The choice of data type determines how much memory your model consumes, how fast it runs, and how accurate its computations are — it's the most fundamental optimization lever."*

## IEEE 754 Floating Point

All floating-point numbers are represented as: $(-1)^s \times 2^{e - \text{bias}} \times (1 + m)$

where **s** = sign bit, **e** = exponent bits, **m** = mantissa (fraction) bits.

## Data Type Comparison

| Type | Bits | Sign | Exponent | Mantissa | Range | Precision | Use Case |
|------|:----:|:----:|:--------:|:--------:|-------|-----------|----------|
| **FP32** | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits | Master weights, loss |
| **TF32** | 19 | 1 | 8 | 10 | ±3.4×10³⁸ | ~3 digits | A100+ Tensor Core matmul |
| **FP16** | 16 | 1 | 5 | 10 | ±65504 | ~3 digits | Inference, mixed precision |
| **BF16** | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | ~2 digits | Training, inference |
| **FP8 E4M3** | 8 | 1 | 4 | 3 | ±448 | ~1 digit | H100+ inference |
| **FP8 E5M2** | 8 | 1 | 5 | 2 | ±57344 | <1 digit | Gradients, KV-cache |
| **INT8** | 8 | 1 | — | 7 | -128 to 127 | Exact integers | Weight/activation quant |
| **INT4** | 4 | 1 | — | 3 | -8 to 7 | 16 levels | Weight quantization |
| **NF4** | 4 | — | — | — | Fixed codebook | 16 optimal levels | QLoRA (normally distr.) |

## Bit Layouts

<div class="diagram">
<div class="diagram-title">Floating Point Bit Layouts</div>
<div class="layer-stack">
  <div class="layer accent">FP32: [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM] — 1+8+23</div>
  <div class="layer purple">FP16: [S|EEEEE|MMMMMMMMMM] — 1+5+10</div>
  <div class="layer green">BF16: [S|EEEEEEEE|MMMMMMM] — 1+8+7 (same exponent as FP32)</div>
  <div class="layer orange">FP8 E4M3: [S|EEEE|MMM] — 1+4+3</div>
  <div class="layer cyan">FP8 E5M2: [S|EEEEE|MM] — 1+5+2 (same exponent as FP16)</div>
</div>
</div>

## Why BF16 > FP16 for Training

The critical difference: **exponent bits determine range, mantissa bits determine precision**.

<div class="diagram">
<div class="diagram-title">BF16 vs FP16</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">FP16 (5 exponent bits)</div>
    <ul>
      <li>Max value: 65,504</li>
      <li>Higher precision (10 mantissa bits)</li>
      <li>Prone to overflow in training</li>
      <li>Requires loss scaling</li>
      <li>Fine for inference</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">BF16 (8 exponent bits)</div>
    <ul>
      <li>Max value: 3.4×10³⁸ (same as FP32)</li>
      <li>Lower precision (7 mantissa bits)</li>
      <li>Rarely overflows — huge dynamic range</li>
      <li>No loss scaling needed</li>
      <li>Standard for modern training</li>
    </ul>
  </div>
</div>
</div>

BF16 is essentially a truncated FP32 — you can convert between them by simply dropping the lower 16 mantissa bits. This makes mixed-precision training trivial.

## Mixed Precision Training

Modern training keeps **master weights in FP32** but uses BF16 for forward/backward passes:

<div class="diagram">
<div class="diagram-title">Mixed Precision Training Flow</div>
<div class="flow">
  <div class="flow-node accent wide">Master weights (FP32) <small>stored in optimizer</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Cast to BF16 for forward pass <small>activations in BF16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Compute loss and backward pass in BF16 <small>gradients in BF16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Update FP32 master weights with BF16 gradients <small>small updates preserved</small></div>
</div>
</div>

Why FP32 master weights? Small gradient updates (e.g., 1e-7) would underflow in BF16 but are preserved in FP32.

```python
import torch

# PyTorch automatic mixed precision
scaler = torch.amp.GradScaler("cuda")  # Only needed for FP16, not BF16

for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(batch["input_ids"])
        loss = criterion(outputs, batch["labels"])
    loss.backward()  # BF16 doesn't need loss scaling
    optimizer.step()
```

## FP8 Inference

NVIDIA H100 and newer GPUs support native FP8 compute with Tensor Cores:

| FP8 Variant | Best For | Range | Precision |
|-------------|----------|-------|-----------|
| **E4M3** | Weights, activations | ±448 | Higher (3 mantissa bits) |
| **E5M2** | Gradients, KV-cache | ±57344 | Lower (2 mantissa bits) |

FP8 inference provides **2× speedup** over FP16 on H100 with minimal quality loss, using per-tensor dynamic scaling.

## Memory Savings at Each Precision

| Precision | Bytes per Param | 8B Model Size | 70B Model Size |
|-----------|:---:|:---:|:---:|
| FP32 | 4 | 32 GB | 280 GB |
| BF16/FP16 | 2 | 16 GB | 140 GB |
| FP8 | 1 | 8 GB | 70 GB |
| INT8 | 1 | 8 GB | 70 GB |
| INT4 | 0.5 | 4 GB | 35 GB |

```python
# Quick memory estimation
def model_memory(params_billions, bytes_per_param):
    return params_billions * 1e9 * bytes_per_param / (1024**3)  # in GB

print(f"8B @ FP16: {model_memory(8, 2):.1f} GB")     # 14.9 GB
print(f"8B @ INT4: {model_memory(8, 0.5):.1f} GB")    # 3.7 GB
print(f"70B @ INT4: {model_memory(70, 0.5):.1f} GB")  # 32.6 GB
```

## Practical Dtype Usage

```python
import torch

# Check available dtypes
x = torch.randn(1000)

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    t = x.to(dtype)
    print(f"{str(dtype):20s} | min: {t.min():.4f} | max: {t.max():.4f} | "
          f"nbytes: {t.nbytes} | finfo: range [{torch.finfo(dtype).min:.0e}, {torch.finfo(dtype).max:.0e}]")

# Overflow in FP16 but not BF16
big = torch.tensor(100000.0)
print(f"FP16: {big.half()}")      # inf (overflow!)
print(f"BF16: {big.bfloat16()}")  # 99840.0 (representable)
```

## What's Next

Understanding data types is the foundation for quantization — the process of reducing model precision from FP16/BF16 to INT8/INT4 for cheaper, faster inference. The next chapter covers **quantization fundamentals**.

[← Previous: Chapter 18 — KV-Cache — Optimization](./18_kv_cache_optimization.md) · **Next: [Chapter 20 — Quantization Fundamentals →](./20_quantization_fundamentals.md)**

---

*Last updated: April 2026*
