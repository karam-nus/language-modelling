---
title: "Chapter 19 — Data Types & Numerical Precision"
description: "How floating-point formats trade range for precision — FP32, TF32, BF16, FP16, FP8 (E4M3, E5M2), FP6 and FP4 — plus integer formats, block-scaled microscaling (MXFP8/MXFP4, NVFP4), which hardware supports what, mixed-precision training, and verified examples of rounding and overflow."
related:
  - { id: quantization, why: "The companion guide to quantization." }
  - { id: chips, why: "How number formats are implemented in silicon." }
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Data Types & Numerical Precision

Every number in a model — weight, activation, gradient, cache entry — is stored in some format, and the format decides how much memory it takes, how fast hardware can multiply it, and how much information it loses. Between 2020 and 2026 the default dropped from 32 bits to 16 for training and towards 8 and 4 for both training and serving.

> **The one-sentence version:** exponent bits buy range, mantissa bits buy precision, and modern low-bit formats recover what 4–8 bits lose by attaching a shared scale to every small block of values.

## Floating Point in One Line

A floating-point value is $(-1)^s \times 2^{e - \text{bias}} \times (1.m)$ — a sign bit, an exponent that sets the scale, and a mantissa (fraction) that sets the detail within that scale. With $M$ mantissa bits, neighbouring values near 1.0 are $2^{-M}$ apart (the format's **epsilon**); with $E$ exponent bits, the representable range spans about $2^{2^{E-1}}$ in each direction.

## Data Type Comparison

| Type | Bits | Sign | Exponent | Mantissa | Max | Epsilon (spacing at 1.0) | Typical use |
|------|:----:|:----:|:--------:|:--------:|-------|-----------|----------|
| **FP32** | 32 | 1 | 8 | 23 | 3.4 × 10³⁸ | 1.2 × 10⁻⁷ | Master weights, optimizer state, reductions |
| **TF32** | 19 (stored as 32) | 1 | 8 | 10 | 3.4 × 10³⁸ | 9.8 × 10⁻⁴ | Tensor-core matmul input format for FP32 code |
| **FP16** | 16 | 1 | 5 | 10 | 65,504 | 9.8 × 10⁻⁴ | Inference; older mixed-precision training |
| **BF16** | 16 | 1 | 8 | 7 | 3.4 × 10³⁸ | 7.8 × 10⁻³ | Training and inference default |
| **FP8 E4M3** | 8 | 1 | 4 | 3 | 448 | 0.125 | Weights and activations (with scales) |
| **FP8 E5M2** | 8 | 1 | 5 | 2 | 57,344 | 0.25 | Gradients (needs range) |
| **FP6 E2M3 / E3M2** | 6 | 1 | 2 / 3 | 3 / 2 | 7.5 / 28 | — | MX block formats |
| **FP4 E2M1** | 4 | 1 | 2 | 1 | 6 | 0.5 | NVFP4 / MXFP4 weights and activations |
| **INT8** | 8 | two's complement | — | — | −128…127 | 1 (uniform) | Weight/activation quantization |
| **INT4** | 4 | two's complement | — | — | −8…7 | 1 (uniform) | Weight-only quantization |
| **NF4** | 4 | lookup table | — | — | ±1 × block scale | non-uniform | QLoRA base weights |

FP4 E2M1 can represent exactly sixteen values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Everything in a 4-bit float model is one of those numbers times a scale.

## Bit Layouts

<div class="diagram">
<div class="diagram-title">Floating-Point Bit Layouts</div>
<div class="layer-stack">
  <div class="layer accent">FP32: [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM] — 1+8+23</div>
  <div class="layer purple">FP16: [S|EEEEE|MMMMMMMMMM] — 1+5+10</div>
  <div class="layer green">BF16: [S|EEEEEEEE|MMMMMMM] — 1+8+7 (FP32's exponent, truncated mantissa)</div>
  <div class="layer orange">FP8 E4M3: [S|EEEE|MMM] — 1+4+3 ("fn": no infinities, max 448)</div>
  <div class="layer cyan">FP8 E5M2: [S|EEEEE|MM] — 1+5+2 (FP16's exponent)</div>
  <div class="layer pink">FP4 E2M1: [S|EE|M] — 1+2+1</div>
</div>
</div>

Type any number into the explorer and see exactly what each format stores — the bits, the decoded value and the rounding error:

<div class="lab" data-lab="bits"><noscript>Interactive lab: encode a number in FP32, BF16, FP16, FP8, FP6 and FP4 and see the bits and rounding error. Enable JavaScript to use it.</noscript></div>

## Why BF16 Beat FP16 for Training

**Exponent bits decide range; mantissa bits decide precision.**

<div class="diagram">
<div class="diagram-title">BF16 vs FP16</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">FP16 (5 exponent bits)</div>
    <ul>
      <li>Max value 65,504; smallest normal 6.1 × 10⁻⁵</li>
      <li>More precision (10 mantissa bits)</li>
      <li>Large activations overflow; small gradients underflow</li>
      <li>Training needs dynamic loss scaling</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">BF16 (8 exponent bits)</div>
    <ul>
      <li>Same range as FP32</li>
      <li>Less precision (7 mantissa bits)</li>
      <li>Practically never overflows</li>
      <li>No loss scaling; the default since A100/TPU</li>
    </ul>
  </div>
</div>
</div>

Neural networks tolerate coarse values far better than they tolerate infinities. BF16 is FP32 with the bottom 16 mantissa bits rounded away, so converting between them is cheap and lossless in range.

Verified with NumPy and `ml_dtypes` (the same rounding PyTorch uses):

```python
import numpy as np
import ml_dtypes as mld

x = np.float32(100000.0)
print(np.float16(x), float(mld.bfloat16(x)))            # inf 99840.0 — FP16 overflows, BF16 rounds
y = np.float32(1.001)
print(float(np.float16(y)), float(mld.bfloat16(y)))     # 1.0009765625 1.0 — BF16 can't see the 0.001
w = np.float32(0.3)
for t in (np.float16, mld.bfloat16, mld.float8_e4m3fn, mld.float8_e5m2, mld.float4_e2m1fn):
    print(t.__name__, float(t(w)))
# float16 0.300048828125
# bfloat16 0.30078125
# float8_e4m3fn 0.3125
# float8_e5m2 0.3125
# float4_e2m1fn 0.5
```

The last line is the whole low-precision story: 0.3 survives 16 bits, gets visibly rounded in 8, and in raw FP4 becomes 0.5. Without a scale, 4 bits cannot represent a typical weight.

## Microscaling: Block-Scaled Formats

The fix for tiny formats is a **shared scale per small block**. Each block of 16 or 32 values stores one scale factor; each element stores only its value relative to that scale. Outliers in one block no longer ruin the resolution of every other block.

| Format | Element | Block size | Scale format | Effective bits per value | Hardware |
|---|---|:---:|---|:---:|---|
| **MXFP8** (OCP MX, 2023) | FP8 E4M3/E5M2 | 32 | E8M0 (power of two) | 8.25 | Blackwell, MI355X |
| **MXFP6** | FP6 E2M3/E3M2 | 32 | E8M0 | 6.25 | Blackwell, MI355X |
| **MXFP4** | FP4 E2M1 | 32 | E8M0 | 4.25 | Blackwell, MI355X; gpt-oss weights |
| **NVFP4** (NVIDIA, 2025) | FP4 E2M1 | 16 | FP8 E4M3 + one FP32 per tensor | 4.5 | Blackwell |
| DeepSeek-V3 FP8 recipe | FP8 E4M3 | 1×128 activations, 128×128 weights | FP32 | ~8 | Hopper (software scaling) |

NVFP4's smaller blocks and finer-grained (non-power-of-two) scales typically lose less accuracy than MXFP4 at the same element width, at the cost of a quarter-bit more per value. The [Chips guide's Chapter 23](https://karam-nus.github.io/chips/23_number_formats_deep_dive.html) works through the encodings bit by bit.

## Mixed-Precision Training

Training keeps a **master copy of the weights in FP32** and does the heavy maths in BF16 (or lower):

<div class="diagram">
<div class="diagram-title">Mixed-Precision Training Flow</div>
<div class="flow">
  <div class="flow-node accent wide">Master weights (FP32) <small>held by the optimizer, with FP32 Adam moments</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Cast to BF16 (or FP8 with scales) for the forward pass <small>matmuls on tensor cores, FP32 accumulation</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Backward pass in BF16 / FP8 <small>gradients in BF16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Update the FP32 master weights <small>tiny updates are preserved</small></div>
</div>
</div>

Why FP32 master weights? A weight of 1.0 receiving an update of 10⁻⁴ stays exactly 1.0 in BF16 — the update is below half of BF16's spacing of 0.0078 at 1.0 — so in pure BF16 most small updates would vanish. (Stochastic rounding is the alternative some FP8/FP4 recipes use.)

```python
import torch

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):    # matmuls in BF16, sensitive ops in FP32
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
    loss.backward()                                        # no GradScaler needed for BF16 (it is for FP16)
    optimizer.step()                                       # parameters and optimizer state stay FP32
```

## Precision Across the Stack (2026)

| Stage | Weights | Activations | Gradients / state | Notes |
|---|---|---|---|---|
| Pre-training (frontier) | FP32 master, BF16/FP8 compute | BF16 / FP8 | BF16 grads, FP32 Adam | FP8 standard since DeepSeek-V3; NVFP4 demonstrated |
| Fine-tuning | BF16 (or NF4 base for QLoRA) | BF16 | FP32 adapter state | |
| Serving (datacenter) | FP8 or 4-bit (MXFP4, NVFP4, INT4) | BF16 / FP8 / FP4 | — | KV cache FP8 |
| Serving (local) | 4–6-bit integer/k-quants (GGUF), AWQ/GPTQ INT4 | FP16 | — | Chapter 22 |

And what the silicon accelerates (dense tensor-core formats):

| Hardware | FP16/BF16 | FP8 | FP6 / FP4 | INT8 |
|---|:---:|:---:|:---:|:---:|
| NVIDIA A100 (2020) | ✓ | – | – | ✓ |
| NVIDIA H100/H200 (2022–24) | ✓ | ✓ | – | ✓ |
| NVIDIA B200/GB300 (2025) | ✓ | ✓ (+MXFP8) | ✓ MXFP6/MXFP4/NVFP4 | ✓ |
| AMD MI300X / MI355X | ✓ | ✓ | MI355X: ✓ | ✓ |
| Google TPU v5p / Ironwood | ✓ | Ironwood: ✓ | – | ✓ |

Each halving of bits roughly doubles peak tensor throughput and halves bytes moved. End-to-end inference speed-ups are smaller — typically 1.3–1.8× per halving — because attention, communication and non-matmul operations do not all shrink with it.

## Memory at Each Precision

| Precision | Bytes per param | 8B model | 70B model | 671B model |
|-----------|:---:|:---:|:---:|:---:|
| FP32 | 4 | 32 GB | 280 GB | 2.7 TB |
| BF16 / FP16 | 2 | 16 GB | 140 GB | 1.34 TB |
| FP8 / INT8 | 1 | 8 GB | 70 GB | 671 GB |
| NVFP4 (incl. scales) | ~0.56 | ~4.5 GB | ~39 GB | ~376 GB |
| INT4 (group 128, incl. scales) | ~0.53 | ~4.2 GB | ~37 GB | ~356 GB |

(Decimal GB; real checkpoints keep embeddings, norms and sometimes the LM head at higher precision, so add a few percent.)

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>FP4 E2M1 has only fifteen distinct non-zero magnitudes per sign, yet NVFP4 models lose little accuracy. What does that say about how much information each individual weight carries — and where is the information actually stored: in the elements, in the scales, or in the pattern across them?</p>
</div>

## Key Takeaways

- Range comes from exponent bits, precision from mantissa bits; BF16 won training because range matters more than precision.
- FP8 E4M3 (weights/activations) and E5M2 (gradients) are standard; FP4 E2M1 has just eight magnitudes and needs scales.
- Microscaling formats (MXFP8/6/4, NVFP4) attach a scale to every 16–32 values; Blackwell and MI355X execute them natively.
- Mixed precision keeps FP32 master weights because small updates vanish in BF16.
- Halving bits halves memory and peak-FLOP cost, but end-to-end speed-ups are smaller.

## Check Yourself

<details class="check">
<summary>What is BF16's spacing between representable values near 1,000, and what does that mean for adding 1 to a BF16 value of 1,000?</summary>
<p>1,000 lies in [512, 1024), where the spacing is 2⁹ × 2⁻⁷ = 4. Adding 1 rounds back to 1,000 — the increment is lost. This is why accumulations and optimizer state stay in FP32.</p>
</details>

<details class="check">
<summary>Why does FP8 E4M3 drop infinities ("fn" = finite)?</summary>
<p>With only 4 exponent bits, reserving the top exponent for inf/NaN would waste a large share of the 256 codes and cut the maximum to 240. E4M3fn keeps a single NaN pattern and uses the rest for finite values, raising the max to 448.</p>
</details>

<details class="check">
<summary>How many bits per value does MXFP4 use including its scale, and why are scales power-of-two?</summary>
<p>4 bits per element + one 8-bit E8M0 scale per 32 elements = 4.25 bits. A power-of-two scale only shifts the exponent, so applying it is exact and cheap in hardware; NVFP4 trades that simplicity for a finer FP8 scale per 16 values.</p>
</details>

<details class="check">
<summary>A gradient value of 3 × 10⁻⁶ is cast to FP16. What happens, and why does BF16 avoid it?</summary>
<p>FP16's smallest normal is 6.1 × 10⁻⁵; 3 × 10⁻⁶ is subnormal and keeps only a few bits of precision (and values below ~6 × 10⁻⁸ flush to zero). Loss scaling multiplies the loss to lift gradients into range. BF16's exponent reaches 10⁻³⁸, so no scaling is needed.</p>
</details>

<details class="check">
<summary>Why is FP4 inference not simply 4× faster than BF16 inference?</summary>
<p>Only the weight-and-activation matmuls get the 4× peak; attention over a BF16/FP8 KV cache, normalisations, sampling, communication and kernel overheads do not shrink, and dequantization or scale handling adds work. Amdahl's law caps the end-to-end gain.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Mixed Precision Training | Micikevicius et al., 2017 | [arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740) | FP32 master weights, loss scaling |
| A Study of BFLOAT16 for Deep Learning Training | Kalamkar et al., 2019 | [arxiv.org/abs/1905.12322](https://arxiv.org/abs/1905.12322) | BF16 |
| FP8 Formats for Deep Learning | Micikevicius et al., 2022 | [arxiv.org/abs/2209.05433](https://arxiv.org/abs/2209.05433) | E4M3 / E5M2 |
| OCP Microscaling Formats (MX) v1.0 Specification | Open Compute Project, 2023 | [opencompute.org](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | MXFP8/6/4, E8M0 scales |
| Microscaling Data Formats for Deep Learning | Rouhani et al., 2023 | [arxiv.org/abs/2310.10537](https://arxiv.org/abs/2310.10537) | MX formats evaluated |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | Fine-grained FP8 training |
| Introducing NVFP4 for Efficient and Accurate Low-Precision Inference | NVIDIA, 2025 | [developer.nvidia.com/blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | NVFP4 block format |
| ml_dtypes | JAX team, 2025 | [github.com/jax-ml/ml_dtypes](https://github.com/jax-ml/ml_dtypes) | Reference implementations used above |

---

**Next:** [Chapter 20 — Quantization Fundamentals](./20_quantization_fundamentals.md) — mapping real weights and activations onto these formats with as little error as possible.

[← Back to Table of Contents](./README.md)
