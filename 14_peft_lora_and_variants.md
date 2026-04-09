---
title: "Chapter 14 — PEFT: LoRA, QLoRA & Variants"
---

[← Back to Table of Contents](./README.md)

# Chapter 14 — PEFT: LoRA, QLoRA & Variants

> *"You don't need to update 7 billion parameters to teach a model new tricks. A few million trainable parameters, properly placed, can reshape model behaviour entirely."*

## Why Parameter-Efficient Fine-Tuning?

Full fine-tuning updates all model weights — expensive in memory, time, and storage. **PEFT** methods train a small subset of parameters while keeping the base model frozen:

<div class="diagram">
<div class="diagram-title">Full Fine-Tuning vs PEFT</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Full Fine-Tuning (LLaMA-3-8B)</div>
    <ul>
      <li>All 8B parameters updated</li>
      <li>BF16 weights: 16 GB</li>
      <li>AdamW states (FP32): 64 GB</li>
      <li>Gradients: 32 GB</li>
      <li>Total: ~112+ GB → multi-A100</li>
      <li>Checkpoint: 16 GB per saved model</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">LoRA (r=16)</div>
    <ul>
      <li>~13M trainable parameters (0.16%)</li>
      <li>Frozen weights: 16 GB</li>
      <li>LoRA states + grads: ~200 MB</li>
      <li>Total: ~18 GB → fits RTX 3090</li>
      <li>Checkpoint: ~26 MB (adapter only)</li>
    </ul>
  </div>
</div>
</div>

---

## LoRA — Low-Rank Adaptation

**LoRA** (Hu et al., 2021) is the most widely used PEFT method. It freezes all pre-trained weights and injects trainable low-rank decomposition matrices into targeted layers.

### Mathematical Foundation

For a weight matrix `W ∈ ℝ^(d×k)`, instead of updating it directly, LoRA adds a **low-rank bypass**:

$$h = Wx + \underbrace{\alpha \cdot BAx}_{\text{LoRA path}}$$

where:
- `B ∈ ℝ^(d×r)` — up-projection matrix, initialised to **zeros** (ensures zero output at init)
- `A ∈ ℝ^(r×k)` — down-projection matrix, initialised with **random Gaussian**
- `r` — rank (hyperparameter, typically 4–128)
- `α` — scaling factor (typically set equal to `r`, or 16)
- Effective scaling: `α/r` multiplied by `BA`

The product `BA` has rank at most `r`, which is ≪ min(d, k) for large weight matrices.

### Tensor Shapes Through LoRA

<div class="diagram">
<div class="diagram-title">LoRA Forward Pass — Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Input x</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">e.g. [4, 2048, 4096]</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">W·x (frozen path)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">no gradient flows here</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">A·x (down-project)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">r</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">e.g. [4, 2048, 16]</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">B·(A·x) (up-project)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">LoRA delta</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 180px;">Output (W + α/r·BA)·x</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">frozen + trainable combined</span>
  </div>
</div>
</div>

### Which Layers to Apply LoRA To

LoRA can be applied to any linear layer. Common choices:

| Target Modules | Parameter Count | Notes |
|----------------|:--------------:|-------|
| `q_proj`, `v_proj` only | Minimal | Original LoRA paper recommendation |
| All attention: `q, k, v, o` | Moderate | Better for most tasks |
| All attention + FFN: `gate, up, down` | Higher | Best quality, recommended by recent work |
| All linear layers | Maximum | Rarely needed |

### Rank Selection

| Rank (r) | Trainable Params (8B) | Quality | Use Case |
|:--------:|:---------------------:|:-------:|----------|
| 4 | ~3M | Basic | Style transfer, simple classification |
| 8 | ~6M | Good | Standard instruction following |
| 16 | ~13M | Better | Code generation, complex reasoning |
| 32 | ~26M | Very good | Domain adaptation |
| 64 | ~52M | Near full FT | Challenging tasks |
| 128 | ~104M | Diminishing returns | Rarely necessary |

**Rule of thumb**: start with `r=16`, increase if quality is insufficient.

### LoRA Merging

At inference time, LoRA adapters can be **merged** back into the base weights with zero overhead:

```python
W_merged = W + (alpha / r) * B @ A  # shape [d, k]
```

After merging, there's no inference latency penalty. This makes LoRA convenient for serving.

---

## QLoRA — Quantized LoRA

**QLoRA** (Dettmers et al., 2023) enables fine-tuning of very large models on consumer GPUs by combining 4-bit quantisation with LoRA.

### QLoRA Components

<div class="diagram">
<div class="diagram-title">QLoRA Stack</div>
<div class="flow">
  <div class="flow-node accent wide"><strong>NF4 — Normal Float 4</strong><br>Base weights quantised to 4 bits using a data type optimised for normally distributed neural network weights</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide"><strong>Double Quantisation</strong><br>Quantise the quantisation constants themselves (saves ~0.37 bits/param extra)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide"><strong>Paged Optimisers</strong><br>Offload Adam optimizer states to CPU RAM during memory spikes (prefill/decode)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide"><strong>LoRA Adapters (BF16)</strong><br>Trainable adapters remain in full precision; gradients computed in BF16</div>
</div>
</div>

### NF4 — Normal Float 4

NF4 is theoretically optimal for weights assumed to be normally distributed (N(0, σ²)):

- The 16 quantisation bins are placed at the **quantiles** of the standard normal distribution
- Equal probability mass in each bin → minimises quantisation error for normally distributed data
- **vs INT4**: NF4 has ~5% better quantisation error on typical neural network weights

The forward pass dequantises weights on-the-fly during matrix multiplication:

```
W_nf4: [d, k] in NF4 (4 bits/param)
  ↓ dequantize
W_bf16: [d, k] in BF16  ← temporary, for this operation only
  ↓ matmul with activations
output: [B, T, d] in BF16
```

### QLoRA Memory Budget (LLaMA-3-8B)

| Component | Precision | Memory |
|-----------|-----------|:------:|
| Base weights | NF4 | 4.0 GB |
| LoRA A/B matrices | BF16 | 26 MB |
| Optimizer states (LoRA only) | FP32 | 52 MB |
| Activations | BF16 | ~1–2 GB |
| **Total** | — | **~5–6 GB** |

Compare to full fine-tuning: ~112 GB. QLoRA fits on a single RTX 4090 (24 GB VRAM).

### QLoRA Practical Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# 4-bit quantisation config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for training (handles frozen base + trainable LoRA)
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,044,765,184 || trainable%: 0.17
```

---

## DoRA — Weight-Decomposed LoRA

**DoRA** (Liu et al., 2024) decomposes weights into **magnitude** and **direction**, applying LoRA only to the directional component:

$$W' = \underbrace{m}_{\text{learnable magnitude}} \cdot \underbrace{\frac{W + BA}{\|W + BA\|_c}}_{\text{normalised direction (LoRA-updated)}}$$

where `‖·‖_c` denotes column-wise normalisation and `m ∈ ℝ^(1×k)` is a learnable vector.

**Why it works**: full fine-tuning tends to update both magnitude and direction together. DoRA explicitly factorises this, giving LoRA the ability to independently adjust both — closing the gap to full fine-tuning.

**Benchmarks**: DoRA consistently matches or exceeds full fine-tuning quality at the same LoRA rank, especially on instruction following and commonsense reasoning.

---

## LoRA Variants Comparison

| Method | Core Idea | Params | vs LoRA Quality | Notes |
|--------|-----------|:------:|:---------------:|-------|
| **LoRA** | Low-rank BA decomposition | r(d+k) | Baseline | Standard, widely supported |
| **QLoRA** | LoRA + NF4 quantised base | r(d+k) | Similar | Enables large models on small GPUs |
| **DoRA** | Magnitude-direction decomposition | r(d+k)+k | Better | Closes gap to full FT |
| **LoRA+** | Different LR for A and B matrices | r(d+k) | Slightly better | Simple improvement |
| **rsLoRA** | Scale by 1/√r instead of 1/r | r(d+k) | Better at high r | More stable rank scaling |
| **LoKr** | Kronecker product decomposition | Varies | Similar | Lower rank expressivity |
| **VeRA** | Shared A/B with per-layer scaling | Very small | Similar | Extreme parameter efficiency |
| **LoRA-FA** | Freeze A, only train B | r×d | Slightly worse | Half the memory of LoRA |
| **MoLoRA** | Mixture of LoRA experts | r(d+k)×E | Better | Multiple specialised adapters |

---

## Other PEFT Methods

### Adapters (Houlsby et al., 2019)

Small bottleneck MLPs inserted **after** attention and FFN layers:

```
Input → Attention → [Adapter: down-project → activation → up-project] → LayerNorm → ...
```

**Tensor shapes**:
- Down-project: `[d_model, r_adapter]` e.g., `[4096, 64]`
- Up-project: `[r_adapter, d_model]` e.g., `[64, 4096]`

**Drawback**: adds sequential computation → inference latency (can't be merged like LoRA).

### Prefix Tuning (Li & Liang, 2021)

Prepend learnable "virtual tokens" to the key and value sequences at each layer:

```
K' = [P_K ; K]   shape: [B, H, n_prefix + T, d_head]
V' = [P_V ; V]   shape: [B, H, n_prefix + T, d_head]
```

The prefix `P_K`, `P_V ∈ ℝ^(n_prefix × d_head)` are learned per task. No weight changes to the model.

**Drawback**: reduces effective context length by `n_prefix`.

### Prompt Tuning (Lester et al., 2021)

The simplest PEFT method: prepend learnable embeddings to the input:

```
input_embeds = [P ; embed(tokens)]
# P shape: [n_prefix, d_model]  e.g., [20, 4096] = 82K trainable params
```

At scale (≥10B params), prompt tuning approaches full fine-tuning quality. Very fast — only ~10K–100K params.

### IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

Learn **scaling vectors** for keys, values, and FFN activations:

```python
K' = l_K ⊙ K    # element-wise scaling, l_K shape: [d_head]
V' = l_V ⊙ V    # same
FFN_out' = l_FF ⊙ FFN_out
```

Even fewer parameters than LoRA — roughly 0.01% of model parameters. Good for few-shot adaptation.

---

## Choosing the Right PEFT Method

<div class="diagram">
<div class="diagram-title">PEFT Selection Guide</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Use LoRA when...</div>
    <div class="card-desc">Fine-tuning on standard hardware. You want mergeable adapters. You need broad task coverage. This is the default choice for 90% of use cases.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Use QLoRA when...</div>
    <div class="card-desc">Model is too large for your GPU in BF16. Fine-tuning 13B+ on consumer hardware. Willing to trade slight quality for accessibility.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Use DoRA when...</div>
    <div class="card-desc">Quality delta between LoRA and full FT is unacceptable. Supported by your training framework. Willing to pay marginal overhead.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Use Prompt Tuning when...</div>
    <div class="card-desc">Using a very large model (≥10B). Multiple tasks with shared model weights. Minimal storage budget per task.</div>
  </div>
</div>
</div>

| Factor | LoRA | QLoRA | DoRA | Adapters | Prompt Tuning |
|--------|:----:|:-----:|:----:|:--------:|:-------------:|
| Trainable % | 0.1–1% | 0.1–1% | 0.1–1%+ | 1–3% | <0.01% |
| Memory (8B) | ~18 GB | ~6 GB | ~18 GB | ~20 GB | ~18 GB |
| Inference latency | None (merge) | None (merge) | None (merge) | +5–10% | +minor |
| Quality vs full FT | 95–99% | 90–97% | ~99% | 95–99% | 90–97% (large) |
| Multi-task swap | Swap adapter | Swap adapter | Swap adapter | Swap adapter | Swap prefix |
| Framework support | ★★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★ |

---

## LoRA in Practice: Common Pitfalls

1. **Wrong target modules**: applying LoRA only to `q_proj` and `v_proj` is common but suboptimal — include all projection matrices including FFN for best results

2. **Rank too low for complex tasks**: `r=4` may be insufficient for code generation or domain-heavy tasks; start at `r=16`

3. **Alpha scaling**: `alpha = r` is a safe default (effective scale = 1). Some practitioners set `alpha = 2*r` for slightly faster convergence

4. **Forgetting to merge before deployment**: serving with active LoRA adapters adds a forward-pass overhead vs merging once and discarding the adapter

5. **QLoRA for small models**: quantisation overhead (dequant on-the-fly) slows training; only worthwhile for models ≥13B where memory savings matter

---

## What's Next

Fine-tuning makes models capable; **alignment** makes them behave. The next chapter covers RLHF, DPO, and how to shape model values and safety.

[← Previous: Chapter 13 — Fine-Tuning & Adaptation](./13_finetuning_and_adaptation.md) · **Next: [Chapter 15 — Alignment: RLHF & Beyond →](./15_alignment_rlhf_and_beyond.md)**

---

*Last updated: April 2026*
