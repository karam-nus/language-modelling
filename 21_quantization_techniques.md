---
title: "Chapter 21 — Quantization Techniques: The Full Landscape"
description: "Every major LLM quantization method and what it is for: GPTQ and AWQ, GGUF k-quants and i-quants, bitsandbytes, codebook methods (QuIP#, AQLM), SmoothQuant and rotation methods (QuaRot, SpinQuant), FP8, MXFP4 and NVFP4, QServe, BitNet b1.58 — plus the 2025–26 tooling (llm-compressor, ModelOpt, GPTQModel) and a master comparison."
related:
  - { id: quantization, why: "The companion guide to quantization." }
  - { id: model-optimization, why: "Pruning, distillation and other ways to shrink models." }
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — Quantization Techniques: The Full Landscape

Chapter 20 showed why naive rounding fails: outliers inflate scales, and every bit removed doubles the rounding step. Each method below is a different answer to the same question — *where can the error go so that the model's outputs barely change?*

> **The one-sentence version:** good quantization minimises error in each layer's *output*, not in its weights — by compensating errors across weights (GPTQ), protecting the weights that big activations touch (AWQ), moving outliers between activations and weights (SmoothQuant), or rotating them away (QuaRot, SpinQuant).

## Overview

<div class="diagram">
<div class="diagram-title">Quantization Technique Map</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Weight-only (W4A16, W3A16, W2A16)</div>
    <div class="card-desc">GPTQ, AWQ, GGUF k-/i-quants, bitsandbytes NF4, HQQ, EXL2/EXL3, QuIP#, AQLM</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Weight + activation (W8A8, W4A8, W4A4)</div>
    <div class="card-desc">FP8, SmoothQuant, QuaRot, SpinQuant, QServe, MXFP4/NVFP4</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Trained with quantization</div>
    <div class="card-desc">QAT checkpoints (Gemma 3 QAT, gpt-oss MXFP4, Kimi K2 Thinking INT4), BitNet b1.58, LLM-QAT</div>
  </div>
</div>
</div>

## Weight-Only Quantization

### GPTQ (Frantar et al., 2022)

GPTQ quantizes one layer at a time, minimising the error of the layer's *output* on calibration inputs, $\lVert WX - \hat{W}X \rVert^2$. It quantizes columns in order and, after each one, adjusts the not-yet-quantized columns to cancel the error just introduced, using the inverse Hessian $H = 2XX^\top$ of that objective (the Optimal Brain Quantization update, made fast with lazy batched updates and a Cholesky factorisation).

- Calibration: ~128–512 sequences of real text.
- Typical config: INT4, group size 128, symmetric, `desc_act` (process columns in order of activation magnitude) on or off depending on kernel support.
- Quantizes a 70B model in a few GPU-hours.

### AWQ (Lin et al., 2023)

**Activation-aware weight quantization** notices that about 1% of weight channels — those multiplied by large activations — matter disproportionately. Rather than keep them in higher precision (awkward for kernels), AWQ multiplies each input channel of W by a per-channel scale s and divides the activations by s (folded into the previous layer). Scaling salient channels up makes their *relative* rounding error smaller. The scales are found by a fast grid search on calibration data — no Hessian, no weight updates — so AWQ is quick and robust to calibration-set choice.

### Producing GPTQ/AWQ/FP8 Checkpoints in 2026

The original AutoGPTQ and AutoAWQ projects were deprecated in 2025; their successors are **GPTQModel** and the vLLM project's **LLM Compressor**, which writes checkpoints in the `compressed-tensors` format that vLLM and SGLang load directly. NVIDIA's **TensorRT Model Optimizer (ModelOpt)** plays the same role for FP8 and NVFP4 on TensorRT-LLM.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tok = AutoTokenizer.from_pretrained(model_id)

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])   # INT4 weights, group 128
oneshot(model=model, dataset="open_platypus", recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained("Llama-3.1-8B-Instruct-W4A16", save_compressed=True)
tok.save_pretrained("Llama-3.1-8B-Instruct-W4A16")
# then: vllm serve ./Llama-3.1-8B-Instruct-W4A16
```

Swap the recipe for `QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])` to produce an FP8 W8A8 checkpoint (no calibration data needed), or an `NVFP4` scheme for Blackwell. AWQ is a recipe in the same tool:

```python
from llmcompressor.modifiers.awq import AWQModifier

recipe = [AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])]   # activation-aware scales, then INT4
oneshot(model=model, dataset="open_platypus", recipe=recipe,
        max_seq_length=2048, num_calibration_samples=256)
```

### GGUF / llama.cpp

**GGUF** is the single-file format of llama.cpp, Ollama and LM Studio — the backbone of local inference on CPUs, Apple Silicon and consumer GPUs. Its "k-quants" use super-blocks of 256 weights split into sub-blocks with their own (quantized) scales and minimums; "i-quants" add lattice codebooks and an **importance matrix** (imatrix) computed from calibration text, so that important weights get more precision.

| GGUF type | Bits per weight (approx.) | Notes |
|-----------|:---:|-------------|
| **Q8_0** | 8.5 | Blocks of 32 with an FP16 scale; essentially lossless |
| **Q6_K** | 6.6 | Near-lossless |
| **Q5_K_M** | 5.7 | Very close to FP16 |
| **Q4_K_M** | 4.8 | The popular default: good quality per byte |
| **IQ4_XS** | 4.3 | i-quant; similar quality to Q4_K_M, smaller |
| **Q4_0** | 4.5 | Legacy: blocks of 32 with one scale, no minimum; fast on ARM |
| **Q3_K_M / IQ3_XXS** | 3.9 / 3.1 | Noticeable loss; use an imatrix |
| **Q2_K / IQ2_XS** | ~2.3–3.0 | Large loss for small models; usable for very large MoE models |

```bash
python convert_hf_to_gguf.py ./Llama-3.1-8B-Instruct --outtype bf16 --outfile llama-8b-bf16.gguf
./llama-imatrix   -m llama-8b-bf16.gguf -f calibration.txt -o imatrix.dat
./llama-quantize  --imatrix imatrix.dat llama-8b-bf16.gguf llama-8b-Q4_K_M.gguf Q4_K_M
```

Mixed-precision GGUFs — keeping attention, embeddings and the first and last layers at higher precision while compressing MoE experts hard ("dynamic" quants popularised by Unsloth in 2025) — are how 600B–1T MoE models run on a single workstation.

### bitsandbytes

The library behind QLoRA, integrated into transformers:

| Mode | Method | Bits | Use |
|------|--------|:----:|----------|
| **LLM.int8()** | INT8 matmul with outlier feature dimensions kept in FP16 | 8 | Memory-saving inference |
| **NF4 / FP4** | 4-bit with block-wise absmax scales, optional double quantization | 4 | QLoRA fine-tuning; quick experiments |

bitsandbytes quantizes on load with no calibration, which makes it convenient — but its kernels are slower than GPTQ/AWQ/Marlin kernels for serving.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,          # quantize the scales too (~0.13 bit/param)
                         bnb_4bit_compute_dtype=torch.bfloat16)   # dequantize to BF16 for each matmul
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             quantization_config=bnb, device_map="auto")
model.get_memory_footprint()   # ≈ 5.7e9 bytes: NF4 linear layers + BF16 embedding and head (Chapter 13's arithmetic)
```

### Other Weight-Only Methods

<div class="diagram">
<div class="diagram-grid cols-2">
  <div class="diagram-card cyan">
    <div class="card-title">QuIP# (Tseng et al., 2024)</div>
    <div class="card-desc">Randomised Hadamard transforms make weights "incoherent" (no outliers), then E8 lattice codebooks quantize groups of 8 weights jointly. Strong at 2–3 bits.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">AQLM (Egiazarian et al., 2024)</div>
    <div class="card-desc">Additive quantization: each group of weights is a sum of entries from several learned codebooks. Among the best quality at 2 bits; slower to produce.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">HQQ (Badri &amp; Shaji, 2023)</div>
    <div class="card-desc">Half-quadratic optimisation of scales and zeros; no calibration data and very fast to apply.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">EXL2 / EXL3 (turboderp)</div>
    <div class="card-desc">ExLlama formats with per-layer variable bit-rates chosen by measured error; EXL3 (2025) adopts QTIP-style trellis quantization.</div>
  </div>
</div>
</div>

## Weight + Activation Quantization

### FP8 (W8A8)

On Hopper and newer GPUs, FP8 is the near-lossless default for datacenter serving: weights per-channel (or per-block) E4M3, activations per-token E4M3 computed on the fly ("dynamic"), FP32 accumulation. No calibration is needed for dynamic activation scales; static scales need a few hundred samples. Many labs now publish FP8 checkpoints themselves (DeepSeek-V3/R1 shipped natively in FP8).

```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", quantization="fp8")   # on-the-fly FP8 weights + dynamic activations
```

### SmoothQuant (Xiao et al., 2022)

**Weights are easy to quantize; activations are hard** (outlier channels). SmoothQuant moves difficulty from activations into weights with a per-channel scale that leaves the product unchanged:

$$Y = (X\,\mathrm{diag}(s)^{-1})\,(\mathrm{diag}(s)\,W) = \hat{X}\hat{W}, \qquad s_j = \frac{\max|X_j|^{\alpha}}{\max|W_j|^{1-\alpha}}$$

With α ≈ 0.5, both sides become quantizable to INT8. The paper reports up to 1.56× speed-up and 2× memory reduction for W8A8 versus FP16 with negligible accuracy loss.

<div class="diagram">
<div class="diagram-title">SmoothQuant — Migrating Outlier Difficulty</div>
<div class="flow">
  <div class="flow-node accent wide">Original: X has outlier channels, W is smooth <small>activations hard to quantize</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Divide X's channels by s, multiply W's matching rows by s <small>mathematically identical output</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Smoothed: X̂ tame, Ŵ slightly harder <small>both now quantizable</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">W8A8: INT8 × INT8 tensor-core matmul</div>
</div>
</div>

### Rotations: QuaRot and SpinQuant (2024)

A cleverer way to kill outliers: multiply activations by an orthogonal matrix R and weights by Rᵀ. The output is unchanged, $XW = (XR)(R^\top W)$, but a random Hadamard rotation spreads each outlier's energy across all channels, so the rotated activations are nearly Gaussian and quantize well even at **4 bits (W4A4)** — including the KV cache. QuaRot uses random Hadamard matrices; SpinQuant *learns* the rotations and was used by Meta for its quantized Llama 3.2 1B/3B releases. Hadamard transforms are now also part of FP4 training recipes.

### QServe / QoQ (W4A8KV4)

QServe (Lin et al., 2024) co-designs a W4A8KV4 scheme ("QoQ") and serving kernels: 4-bit weights with progressive group quantization so dequantization to INT8 is cheap, 8-bit per-token activations, and 4-bit KV cache.

| Component | Precision | Method |
|-----------|:---------:|--------|
| Weights | W4 | Progressive group-quantized INT4 (INT8 per-channel then INT4 groups) |
| Activations | A8 | Per-token dynamic INT8 |
| KV cache | KV4 | 4-bit with SmoothAttention for keys |

### MXFP4 and NVFP4 (W4A4 on Blackwell)

Blackwell-class GPUs execute 4-bit floating-point matmuls natively, with block scales ([Chapter 19](./19_data_types_and_precision.md)). PTQ to NVFP4 with ModelOpt or LLM Compressor keeps quality close to FP8 for many large models — NVIDIA publishes NVFP4 checkpoints of DeepSeek-R1, Llama and Qwen — and gives roughly 2× FP8's peak throughput. gpt-oss shipped its MoE weights in MXFP4 so that the 120B model fits in one 80 GB GPU.

## Trained with Quantization

### QAT Checkpoints

Quantization-aware training — fake-quantizing during a short fine-tuning phase so the model adapts — recovers most of the gap that PTQ leaves at 4 bits. In 2025 several labs released QAT checkpoints: Gemma 3 (Q4_0 for llama.cpp), gpt-oss (MXFP4 experts), Kimi K2 Thinking (native INT4 weights, roughly halving its memory and speeding RL rollouts).

### BitNet b1.58

BitNet b1.58 (Ma et al., 2024) trains transformers from scratch with **ternary** weights {−1, 0, +1} (log₂3 ≈ 1.58 bits) and 8-bit activations. Matmuls become additions and subtractions. In April 2025 Microsoft released **BitNet b1.58 2B4T**, an open 2B model trained on 4T tokens that is competitive with full-precision models of similar size, with the `bitnet.cpp` runtime for CPUs.

<div class="diagram">
<div class="diagram-title">BitNet — Ternary-Weight Transformer</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Standard transformer</div>
    <ul>
      <li>Weights: BF16 (16 bits per parameter)</li>
      <li>Matmul: multiply-accumulate</li>
      <li>2 bytes per parameter</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">BitNet b1.58</div>
    <ul>
      <li>Weights: {−1, 0, +1} with a per-tensor scale</li>
      <li>Matmul: add, subtract, skip</li>
      <li>~0.2 bytes per parameter</li>
    </ul>
  </div>
</div>
</div>

The catch: it must be trained that way from the start, and no frontier-scale BitNet exists yet.

## Master Comparison Table

| Method | Quantizes | Bits | Type | Calibration | Typical quality | Speed-up | Ecosystem |
|--------|--------|:----:|:-------:|:-----------:|:---:|:---:|----------|
| **FP8 (dynamic)** | W + A | 8 | PTQ | None | ≈ lossless | 1.3–1.6× end-to-end | vLLM, SGLang, TRT-LLM (Hopper+) |
| **NVFP4 / MXFP4** | W + A | ~4.5 | PTQ or QAT | Small set | Close to FP8 for large models | ~2× over FP8 peak | Blackwell; ModelOpt, LLM Compressor |
| **GPTQ** | W | 4 (3) | PTQ | 128–512 seqs | Small loss at 4-bit | Up to ~3× decode at batch 1 (Marlin) | vLLM, SGLang, transformers |
| **AWQ** | W | 4 | PTQ | ~128 seqs | Small loss; robust | Up to ~3× decode at batch 1 | vLLM, SGLang, transformers |
| **GGUF Q4_K_M / IQ4_XS** | W | 4.3–4.8 | PTQ | Optional imatrix | Small loss | Makes CPU / Mac inference practical | llama.cpp, Ollama, LM Studio |
| **bitsandbytes NF4** | W | 4 | PTQ (on load) | None | Small loss | Slower kernels | QLoRA, transformers |
| **SmoothQuant** | W + A | 8 | PTQ | ~512 seqs | ≈ lossless | Up to 1.56× (paper) | INT8 serving |
| **QuaRot / SpinQuant** | W + A + KV | 4 | PTQ (+ learned rotations) | Small set | Best-in-class W4A4 | Needs INT4 kernels | Research, Llama 3.2 quantized |
| **QServe** | W + A + KV | 4/8/4 | PTQ | Calibration | Small loss | High throughput | QServe |
| **QuIP# / AQLM** | W | 2–3 | PTQ (+ fine-tune) | Calibration | Best at 2 bits | Moderate | Research, some kernels |
| **BitNet b1.58** | W (+ A8) | 1.58 | From-scratch training | Full training | Competitive at 2B | Large on CPU (bitnet.cpp) | Microsoft |

## Dequantization at Runtime

Weight-only kernels dequantize on the fly:

<div class="diagram">
<div class="diagram-title">INT4 Weight-Only Matmul</div>
<div class="flow">
  <div class="flow-node accent wide">Packed INT4 weights: [d_out, d_in/8] int32 <small>8 values per word, plus group scales</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Load a tile into registers / shared memory <small>4× fewer bytes from HBM than BF16</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Unpack and dequantize: (q − z) × s → BF16 <small>in registers, never written back</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">BF16 tensor-core matmul with the activations</div>
</div>
</div>

Fused kernels such as **Marlin** (Ampere/Ada) and **Machete** (Hopper) do all of this in one pass and approach the ideal ~4× speed-up for bandwidth-bound decode at small batch; at large batch the extra dequantization work makes W4A16 *slower* than FP8 W8A8, which is why datacenter serving moved to FP8 and FP4 activation formats.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Rotations make activations Gaussian, codebooks make 2 bits work, and QAT makes 4 bits nearly free. If the 2026 frontier MoE models are trained in FP8 and served in FP4, is post-training quantization becoming a community-only practice — and what would make PTQ matter again?</p>
</div>

## Key Takeaways

- GPTQ (error compensation) and AWQ (activation-aware scaling) are the 4-bit weight-only workhorses; produce them today with GPTQModel or LLM Compressor.
- GGUF k-/i-quants with an importance matrix power local inference; mixed-precision GGUFs make giant MoE models runnable at home.
- For W+A quantization, FP8 is the near-lossless default; SmoothQuant moves outliers into weights; QuaRot/SpinQuant rotate them away to reach W4A4.
- Blackwell's NVFP4/MXFP4 make 4-bit activations practical in datacenters; labs increasingly ship QAT checkpoints.
- BitNet b1.58 shows ternary-from-scratch works at 2B; it is not yet a frontier recipe.

## Check Yourself

<details class="check">
<summary>What does GPTQ minimise, and why is that better than minimising ‖W − Ŵ‖?</summary>
<p>It minimises the layer output error ‖WX − ŴX‖² on calibration inputs. Weight error in directions the inputs rarely excite does not matter; GPTQ uses the Hessian XXᵀ to push error into those directions and compensate the rest.</p>
</details>

<details class="check">
<summary>AWQ scales a salient channel up by s = 2 before rounding. Why does that reduce its error, and what keeps the output unchanged?</summary>
<p>The rounding step (set by the group's scale) is roughly unchanged, so the channel's absolute rounding error stays similar while its magnitude doubles — halving its relative error. Dividing the matching activation channel by 2 (folded into the previous layer's weights or norm) keeps the product identical.</p>
</details>

<details class="check">
<summary>Why does a Hadamard rotation help 4-bit activation quantization?</summary>
<p>An outlier concentrated in one channel is spread evenly across all channels by the orthogonal rotation, so the maximum magnitude per token drops sharply while the vector's norm is unchanged. A smaller max means a smaller scale and finer resolution for everything else; the inverse rotation is folded into the weights.</p>
</details>

<details class="check">
<summary>Why can W4A16 be slower than W8A8 FP8 at batch size 64?</summary>
<p>At larger batches the matmul becomes compute-bound. W4A16 still multiplies in BF16 after dequantizing, so it gets no tensor-core speed-up and pays dequantization cost; FP8 W8A8 runs the matmul itself at twice BF16's peak.</p>
</details>

<details class="check">
<summary>How many bits per weight does a 671B MoE need to fit in 256 GB, and which formats get there?</summary>
<p>256 × 8 / 671 ≈ 3.05 bits per weight on average — below 4-bit. Mixed-precision GGUF (≈2–3-bit experts, higher-precision attention and shared layers) or 2–3-bit codebook methods; uniform 4-bit would need ~360 GB.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| GPTQ | Frantar et al., 2022 | [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323) | Hessian-based error compensation |
| AWQ | Lin et al., 2023 | [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978) | Activation-aware scaling |
| LLM.int8() | Dettmers et al., 2022 | [arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339) | Mixed-precision INT8 |
| SmoothQuant | Xiao et al., 2022 | [arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438) | Outlier migration |
| QuaRot | Ashkboos et al., 2024 | [arxiv.org/abs/2404.00456](https://arxiv.org/abs/2404.00456) | Hadamard rotations for W4A4KV4 |
| SpinQuant | Liu et al. (Meta), 2024 | [arxiv.org/abs/2405.16406](https://arxiv.org/abs/2405.16406) | Learned rotations |
| QuIP# | Tseng et al., 2024 | [arxiv.org/abs/2402.04396](https://arxiv.org/abs/2402.04396) | Incoherence + lattice codebooks |
| AQLM | Egiazarian et al., 2024 | [arxiv.org/abs/2401.06118](https://arxiv.org/abs/2401.06118) | Additive quantization |
| QServe | Lin et al., 2024 | [arxiv.org/abs/2405.04532](https://arxiv.org/abs/2405.04532) | W4A8KV4 co-design |
| The Era of 1-bit LLMs (BitNet b1.58) | Ma et al., 2024 | [arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764) | Ternary weights |
| BitNet b1.58 2B4T Technical Report | Microsoft, 2025 | [arxiv.org/abs/2504.12285](https://arxiv.org/abs/2504.12285) | Open ternary model |
| LLM Compressor | vLLM project, 2025 | [github.com/vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | Producing compressed-tensors checkpoints |
| llama.cpp quantization docs | ggml-org, 2025 | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize) | GGUF types, imatrix |

---

**Next:** [Chapter 22 — Quantization Benchmarks & Selection](./22_quantization_benchmarks.md) — how much each method actually costs in quality, and how to choose.

[← Back to Table of Contents](./README.md)
