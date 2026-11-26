---
title: "Chapter 20 — Quantization Fundamentals"
description: "The affine quantization map, symmetric vs asymmetric, granularity from per-tensor to per-group and microscaling blocks, a verified outlier experiment, PTQ vs QAT (now common for release checkpoints), calibration, weight-only vs weight-activation vs KV quantization, and why LLM activations have massive outliers."
related:
  - { id: quantization, why: "The companion guide to quantization." }
  - { id: weight-analysis, why: "Per-channel weight plots of models with quantization trouble." }
---

[← Back to Table of Contents](./README.md)

# Chapter 20 — Quantization Fundamentals

> **The one-sentence version:** quantization stores each block of numbers as small integers (or tiny floats) plus a scale, and nearly all of its difficulty comes from a few outlier values that force the scale to be large.

## What Is Quantization?

Quantization maps high-precision values onto a small set of levels — integers (INT8, INT4) or tiny floats (FP8, FP4) — plus the scale needed to map back. For LLMs it means BF16 → 8 or 4 bits (sometimes fewer), cutting memory and, because decode is bandwidth-bound, speeding up generation.

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
import numpy as np

def quantize_sym(x, bits=8, axis=None):
    """Symmetric round-to-nearest. axis=None → one scale per tensor; axis=1 → one per row."""
    qmax = 2 ** (bits - 1) - 1                                  # 127 for INT8, 7 for INT4
    scale = np.abs(x).max(axis=axis, keepdims=axis is not None) / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
    return q, scale                                             # dequantize: q * scale

def quantize_group(x, bits=4, g=128):
    """One scale per group of g consecutive values along the input dimension."""
    qmax = 2 ** (bits - 1) - 1
    xg = x.reshape(x.shape[0], -1, g)
    s = np.abs(xg).max(-1, keepdims=True) / qmax
    return (np.clip(np.round(xg / s), -qmax, qmax) * s).reshape(x.shape)

rng = np.random.default_rng(0)
w = rng.standard_normal((4096, 4096)).astype(np.float32) * 0.02   # weight-like values
w[:, 7] *= 50                                                       # one outlier input channel
rel = lambda w_hat: np.linalg.norm(w - w_hat) / np.linalg.norm(w)
for bits in (8, 4):
    for axis, name in ((None, "per-tensor"), (1, "per-row")):
        q, s = quantize_sym(w, bits, axis)
        print(f"INT{bits} {name:10s} relative error {rel(q * s):.4f}")
print(f"INT4 group-128  relative error {rel(quantize_group(w)):.4f}")
# INT8 per-tensor relative error 0.3677
# INT8 per-row    relative error 0.0899
# INT4 per-tensor relative error 0.7931
# INT4 per-row    relative error 0.6642
# INT4 group-128  relative error 0.1482
```

One channel 50× larger than the rest wrecks per-tensor quantization even at 8 bits. Per-row scales help at INT8, but at INT4 the outlier *column* appears in every row, so every row's scale is still dominated by it. Groups of 128 along the input dimension confine the outlier to one group per row and cut the error fivefold. Try it with your own outliers, bit-widths and group sizes:

<div class="lab" data-lab="quant"><noscript>Interactive lab: quantize a weight vector with outliers at different bit-widths and granularities, and see the reconstruction error. Enable JavaScript to use it.</noscript></div>

## Quantization Granularity

**Where** you compute scales dramatically affects quality:

<div class="diagram">
<div class="diagram-title">Quantization Granularity Levels</div>
<div class="flow">
  <div class="flow-node accent wide">Per-Tensor: 1 scale for entire weight matrix <small>fastest, lowest quality</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Per-Channel: 1 scale per output channel (row) <small>good balance</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Per-Group / per-block: 1 scale per group of G values (G = 128 for INT4 weights; 16–32 in MXFP4/NVFP4) <small>the 4-bit standard</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Per-Element: 1 scale per value <small>no compression — pointless</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Per-Group Quantization Tensor Layout</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Original weight</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_in</span>
      <span class="ts-bracket">]</span>
      <span class="ts-note">@ FP16: 2 bytes each</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Quantized weight</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d_in/pack</span>
      <span class="ts-bracket">]</span>
      <span class="ts-note">@ INT4: 8 values packed per int32</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Scales</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">d_in/G</span>
      <span class="ts-bracket">]</span>
      <span class="ts-note">@ FP16: 1 scale per group of G</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Zero points</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim feature">d_out</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">d_in/G</span>
      <span class="ts-bracket">]</span>
      <span class="ts-note">@ INT4 or FP16</span>
    </div>
  </div>
</div>
</div>

Effective bits per value with group size 128 and FP16 scale and zero point: 4 + 32/128 = **4.25 bits**. Microscaling formats make the same trade with hardware support: MXFP4 = 4.25 bits (block 32), NVFP4 = 4.5 bits (block 16) — see [Chapter 19](./19_data_types_and_precision.md).

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
      <li>Costs a (usually short) training phase</li>
      <li>Best quality at low bit-widths</li>
    </ul>
  </div>
</div>
</div>

For community quantization of released models, **PTQ dominates** — nobody but the lab has the training pipeline. But by 2025–26 labs increasingly ship **QAT checkpoints** themselves: Google released QAT versions of Gemma 3 for 4-bit use, OpenAI post-trained gpt-oss with its MoE weights in MXFP4, and Moonshot released Kimi K2 Thinking with native INT4 weights from QAT. At the extreme, BitNet b1.58 trains with ternary weights from scratch ([Chapter 21](./21_quantization_techniques.md)). QAT works by inserting *fake quantization* (quantize-dequantize) in the forward pass and passing gradients straight through the rounding ([Chapter 23](./23_knowledge_distillation.md)).

## Calibration Strategies

PTQ methods need a calibration dataset to determine quantization parameters (scales, zero points). The choice of calibration strategy matters:

| Strategy | Description | Notes |
|----------|-------------|:---:|
| **Min-max** | Scale from observed min/max | Simple; one outlier sets the scale |
| **Percentile** | Clip at e.g. the 99.99th percentile | Trades clipping error for resolution |
| **MSE search** | Choose the clipping range that minimises reconstruction error | Common default for weights |
| **Entropy (KL)** | Minimise KL between original and quantized histograms | Classic TensorRT activation calibration |
| **Output-aware (GPTQ, AWQ)** | Minimise error in each layer's *output* on calibration inputs, not in the weights themselves | The basis of modern 4-bit PTQ — Chapter 21 |

Calibration data matters: a few hundred samples representative of real use (chat, code, the target languages) beat random web text, especially for instruction-tuned models.

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
    <div class="card-desc">Quantize cached K, V tensors. Reduces memory and read bandwidth for long sequences. See Ch 18. <strong>Serving-bound savings.</strong></div>
  </div>
</div>
</div>

When to use each:
- **Weight-only (W4A16, W8A16)**: small-batch and local inference — decode is bandwidth-bound, so smaller weights mean faster reads even though the matmul itself runs in BF16.
- **Weight + activation (W8A8 FP8/INT8, W4A4 FP4)**: high-throughput serving and prefill, which are compute-bound — only low-precision *inputs to the matmul* unlock faster tensor cores.
- **KV cache (FP8, 4-bit)**: long-context and high-concurrency serving.

In 2026 datacenter serving, **FP8 W8A8 is the near-lossless default**, NVFP4/MXFP4 W4A4 is the Blackwell-era push, and INT4 weight-only (AWQ/GPTQ, GGUF) dominates local inference.

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

Where do they come from? LLMs develop a handful of **massive activations** — specific hidden dimensions, often at the first token and at delimiter tokens, with values thousands of times the median — that act as fixed biases and are tied to attention sinks ([Chapter 4](./04_attention_sdpa_and_mha.md)). They live in a few *channels*, consistently across tokens, which is why per-channel tricks work. Weights have outliers too, but milder: roughly 1% of weights matter disproportionately (AWQ's "salient weights").

Methods such as SmoothQuant, GPTQ, AWQ, and rotation-based QuaRot/SpinQuant are different strategies for exactly this problem — see [Chapter 21](./21_quantization_techniques.md). Architectures that remove attention sinks (gated attention, [Chapter 4](./04_attention_sdpa_and_mha.md)) also remove many massive activations, making models easier to quantize.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Massive activations look like a nuisance for quantization, yet the model learned them on purpose — removing them hurts quality. If they act as built-in biases and "no-op" switches for attention, could an architecture provide those functions explicitly and leave every channel well-behaved?</p>
</div>

## Key Takeaways

- Affine quantization: $x_q = \mathrm{round}(x/s) + z$, $\hat x = s(x_q - z)$; symmetric sets z = 0, asymmetric uses the full integer range.
- Granularity is the main quality lever: per-tensor → per-channel → per-group (128 for INT4) → micro-blocks (16–32 for MXFP4/NVFP4).
- A few outlier channels dominate per-tensor scales; groups along the input dimension contain them.
- PTQ is the community default; labs increasingly ship QAT checkpoints (Gemma 3 QAT, gpt-oss MXFP4, Kimi K2 Thinking INT4).
- Weight-only quantization speeds bandwidth-bound decode; weight-activation quantization speeds compute-bound prefill and batched serving; FP8 W8A8 is the near-lossless datacenter default.

## Check Yourself

<details class="check">
<summary>Symmetric INT8 with max |x| = 6.35: what is the scale, and what does 0.1 quantize to?</summary>
<p>s = 6.35/127 = 0.05. 0.1/0.05 = 2 → stored as 2, dequantized to exactly 0.1. A value of 0.12 → round(2.4) = 2 → 0.1, an error of 0.02.</p>
</details>

<details class="check">
<summary>Why didn't per-row scaling fix INT4 in the experiment above?</summary>
<p>The outlier is an input channel — a column — so it appears in every row. Each row's max is dominated by its element in that column, so every row gets a large scale and the ordinary weights collapse onto a few levels. Grouping along the input dimension isolates the outlier in one group per row.</p>
</details>

<details class="check">
<summary>Why does weight-only INT4 speed up decode but not a large prefill?</summary>
<p>Decode at small batch is limited by reading weights from memory; 4× fewer bytes means up to ~4× faster reads. Prefill is limited by matmul compute, and weight-only kernels dequantize to BF16 before multiplying, so the arithmetic is unchanged (and dequantization adds work).</p>
</details>

<details class="check">
<summary>What is the storage overhead of asymmetric INT4 with group size 64 and FP16 scale and zero point?</summary>
<p>32 bits per 64 values = 0.5 bits per value, so 4.5 bits per value in total — the same as NVFP4's 4.5.</p>
</details>

<details class="check">
<summary>What does fake quantization do in QAT, and how do gradients pass through rounding?</summary>
<p>It quantizes and immediately dequantizes weights (and possibly activations) in the forward pass, so the loss sees quantization error. Rounding has zero gradient almost everywhere, so the straight-through estimator treats it as identity in the backward pass (often only within the clipping range).</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| A Survey of Quantization Methods for Efficient Neural Network Inference | Gholami et al., 2021 | [arxiv.org/abs/2103.13630](https://arxiv.org/abs/2103.13630) | Affine quantization, granularity, PTQ vs QAT |
| Integer Quantization for Deep Learning Inference | Wu et al. (NVIDIA), 2020 | [arxiv.org/abs/2004.09602](https://arxiv.org/abs/2004.09602) | Calibration methods |
| LLM.int8() | Dettmers et al., 2022 | [arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339) | Emergent outlier features |
| Massive Activations in Large Language Models | Sun et al., 2024 | [arxiv.org/abs/2402.17762](https://arxiv.org/abs/2402.17762) | Where outliers come from |
| SmoothQuant | Xiao et al., 2022 | [arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438) | Activation outliers in W8A8 |
| AWQ | Lin et al., 2023 | [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978) | Salient weights |
| Gemma 3 QAT models | Google, 2025 | [developers.googleblog.com](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/) | QAT checkpoints from a lab |

---

**Next:** [Chapter 21 — Quantization Techniques: The Full Landscape](./21_quantization_techniques.md) — GPTQ, AWQ, SmoothQuant, rotations, low-bit formats and the methods that handle outliers.

[← Back to Table of Contents](./README.md)
