---
title: "Chapter 20 — Quantization Fundamentals"
---

[← Back to Table of Contents](./README.md)

# Chapter 20 — Quantization Fundamentals

> *"Quantization is the art of throwing away precision strategically — keeping what matters and discarding what doesn't."*

## What Is Quantization?

Quantization maps high-precision floating-point values to lower-precision integers. For LLMs, this typically means FP16 → INT8 or INT4, reducing memory and enabling faster inference.

$$x_q = \text{round}\!\left(\frac{x}{s}\right) + z$$

where $s$ = scale factor, $z$ = zero point, and $x_q$ is the quantized integer value.

Dequantization (at inference time): $\hat{x} = s \cdot (x_q - z)$

## Symmetric vs Asymmetric Quantization

<div class="img-caption">
  <img src="{{ '/assets/images/quantization_overview.svg' | relative_url }}" alt="Symmetric versus asymmetric INT8 quantization showing number lines, formulas, and worked examples">
  <figcaption>Symmetric quantization maps to [−127, 127] around zero; asymmetric maps to [0, 255] with a zero-point offset</figcaption>
</div>

<div class="diagram">
<div class="diagram-title">Symmetric vs Asymmetric Quantization</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Symmetric</div>
    <ul>
      <li>Zero point z = 0 (no offset)</li>
      <li>Scale: s = max(|x|) / (2^(b-1) - 1)</li>
      <li>Range: [-127, 127] for INT8</li>
      <li>Simpler, faster dequantization</li>
      <li>Wastes range if distribution is skewed</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Asymmetric</div>
    <ul>
      <li>Zero point z ≠ 0 (shifted)</li>
      <li>Scale: s = (max(x) - min(x)) / (2^b - 1)</li>
      <li>Zero point: z = round(-min(x) / s)</li>
      <li>Full range utilization</li>
      <li>Better for skewed distributions</li>
    </ul>
  </div>
</div>
</div>

```python
import torch

def symmetric_quantize(x, bits=8):
    """Symmetric quantization — zero-centered."""
    qmax = 2 ** (bits - 1) - 1                    # 127 for INT8
    scale = x.abs().max() / qmax
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale

def symmetric_dequantize(x_q, scale):
    """Dequantize back to float."""
    return x_q.float() * scale

# Example
w = torch.randn(4096, 4096)                       # FP32 weight matrix
w_q, scale = symmetric_quantize(w, bits=8)
w_hat = symmetric_dequantize(w_q, scale)           # Reconstructed
print(f"Original: {w.nbytes / 1e6:.1f} MB")       # 67.1 MB
print(f"Quantized: {w_q.nbytes / 1e6:.1f} MB")    # 16.8 MB (4× smaller)
print(f"Max error: {(w - w_hat).abs().max():.6f}") # Small reconstruction error
```

## Quantization Granularity

**Where** you compute scales dramatically affects quality:

<div class="diagram">
<div class="diagram-title">Quantization Granularity Levels</div>
<div class="flow">
  <div class="flow-node accent wide">Per-Tensor: 1 scale for entire weight matrix <small>fastest, lowest quality</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Per-Channel: 1 scale per output channel (row) <small>good balance</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Per-Group: 1 scale per group of G values (e.g., G=128) <small>best quality for INT4</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Per-Element: 1 scale per value <small>no compression — pointless</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Per-Group Quantization Tensor Layout</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Original weight</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_in</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ FP16: 2 bytes each</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Quantized weight</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_in/pack</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ INT4: 8 values packed per int32</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Scales</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">d_in/G</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ FP16: 1 scale per group of G</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Zero points</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">d_in/G</span>
      <span class="ts-bracket">]</span>
      <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">@ INT4 or FP16</span>
    </div>
  </div>
</div>
</div>

Effective bits per value with group size 128: INT4 + scales + zeros ≈ **4.25 bits** (the overhead from storing scales is small).

## PTQ vs QAT

<div class="diagram">
<div class="diagram-title">Post-Training Quantization vs Quantization-Aware Training</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">PTQ (Post-Training)</div>
    <ul>
      <li>Quantize after training is complete</li>
      <li>Uses calibration dataset (~128–512 samples)</li>
      <li>No additional training needed</li>
      <li>Fast: minutes to hours</li>
      <li>Lower quality at extreme compression (INT3/2)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">QAT (Aware Training)</div>
    <ul>
      <li>Simulate quantization during training</li>
      <li>Model adapts to quantization noise</li>
      <li>Requires training infrastructure</li>
      <li>Slow: full training cost</li>
      <li>Best quality at any bit-width</li>
    </ul>
  </div>
</div>
</div>

For LLMs, **PTQ dominates** because models are expensive to retrain. QAT methods like BitNet train from scratch.

## Calibration Strategies

PTQ methods need a calibration dataset to determine quantization parameters (scales, zero points). The choice of calibration strategy matters:

| Strategy | Description | Quality |
|----------|-------------|:---:|
| **Min-Max** | Use observed min/max of activations | Baseline |
| **Percentile** | Clip to 99.9th percentile (ignore outliers) | Better |
| **MSE** | Minimize reconstruction error (MSE) per layer | Best |
| **Entropy** | Minimize KL divergence between original and quantized | Good |

## What Gets Quantized

<div class="diagram">
<div class="diagram-title">Quantization Targets</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Weight-Only</div>
    <div class="card-desc">Quantize weights to INT4/8. Activations stay in FP16. Simplest, most common. <strong>Memory-bound speedup.</strong></div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Weight + Activation</div>
    <div class="card-desc">Quantize both weights and activations (W8A8, W4A8). Enables INT8 matmul. <strong>Compute-bound speedup.</strong></div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">KV-Cache</div>
    <div class="card-desc">Quantize cached K, V tensors. Reduces memory for long sequences. See Ch 15. <strong>Serving-bound savings.</strong></div>
  </div>
</div>
</div>

When to use each:
- **Weight-only**: most LLM inference (decode is memory-bandwidth bound — smaller weights = faster reads)
- **Weight + activation**: latency-sensitive serving, batch inference (prefill is compute-bound)
- **KV-cache**: long-context serving, high-throughput scenarios

## The Outlier Problem

LLM activations have massive **outliers** — a few channels with values 100× larger than average. These outliers make naive quantization fail because the scale is dominated by outliers, leaving most values underrepresented.

```python
# Typical activation distribution in a transformer
# 99% of values: [-3, 3]
# 0.1% outlier channels: [-100, 100]
# If we quantize to [-127, 127] with max=100:
#   scale = 100/127 ≈ 0.79
#   A value of 1.0 maps to round(1.0/0.79) = 1 → dequant = 0.79
#   Quantization error of 0.21 — over 20% for typical values!
```

This is why methods like SmoothQuant, GPTQ, and AWQ were invented — they're all different strategies for handling outliers. See [Chapter 21](./21_quantization_techniques.md).

## What's Next

With the fundamentals in place, the next chapter surveys the full landscape of **quantization techniques** — from GPTQ and AWQ to SmoothQuant and BitNet, with a master comparison table.

[← Previous: Chapter 19 — Data Types & Numerical Precision](./19_data_types_and_precision.md) · **Next: [Chapter 21 — Quantization Techniques →](./21_quantization_techniques.md)**

---

*Last updated: April 2026*
