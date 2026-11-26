---
title: "Chapter 14 — PEFT: LoRA, QLoRA & Variants"
description: "Low-rank adaptation from the maths up: shapes, initialisation, scaling, which layers to target and how many parameters that really is; QLoRA's NF4 and double quantization with an honest memory budget; DoRA, rsLoRA, LoRA+, PiSSA and other variants; adapters, prefix and prompt tuning; multi-LoRA serving."
related:
  - { id: huggingface, why: "PEFT and TRL in the Hugging Face ecosystem." }
  - { id: quantization, why: "The quantization behind QLoRA's 4-bit base." }
---

[← Back to Table of Contents](./README.md)

# Chapter 14 — PEFT: LoRA, QLoRA & Variants

> **The one-sentence version:** fine-tuning updates turn out to be low-rank, so instead of learning a d×k change to each weight matrix you can learn two thin matrices whose product is that change — and keep the base model frozen, shared and even quantized.

## Why Parameter-Efficient Fine-Tuning?

Full fine-tuning updates all model weights — expensive in memory, time, and storage. **PEFT** methods train a small subset of parameters while keeping the base model frozen:

<div class="diagram">
<div class="diagram-title">Full Fine-Tuning vs PEFT</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Full Fine-Tuning (Llama 3.1 8B)</div>
    <ul>
      <li>All 8B parameters updated</li>
      <li>BF16 weights + grads: 32 GB</li>
      <li>FP32 master weights + AdamW m, v: 96 GB</li>
      <li>Total ≈ 128 GB of state before activations → several GPUs with FSDP</li>
      <li>Checkpoint: 16 GB per saved model</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">LoRA (r = 16, all linear layers)</div>
    <ul>
      <li>41.9M trainable parameters (0.52%)</li>
      <li>Frozen BF16 weights: 16 GB</li>
      <li>Adapters + grads + Adam: ~0.5 GB</li>
      <li>Total ≈ 20 GB with activations → one 24 GB GPU</li>
      <li>Checkpoint: ~84 MB (adapter only)</li>
    </ul>
  </div>
</div>
</div>

---

## LoRA — Low-Rank Adaptation

<div class="img-caption">
  <img src="{{ '/assets/images/lora_decomposition.svg' | relative_url }}" alt="LoRA weight decomposition showing frozen W0 plus trainable low-rank BA matrices, with parameter count comparison">
  <figcaption>LoRA injects trainable rank-<em>r</em> matrices into frozen weights — same inference cost when merged</figcaption>
</div>

**LoRA** (Hu et al., 2021) is the most widely used PEFT method. It freezes all pre-trained weights and injects trainable low-rank decomposition matrices into targeted layers.

### Mathematical Foundation

For a weight matrix `W ∈ ℝ^(d×k)`, instead of updating it directly, LoRA adds a **low-rank bypass**:

$$h = Wx + \underbrace{\tfrac{\alpha}{r} \cdot BAx}_{\text{LoRA path}}$$

where:
- `B ∈ ℝ^(d×r)` — up-projection matrix, initialised to **zeros** (ensures zero output at init)
- `A ∈ ℝ^(r×k)` — down-projection matrix, initialised with **random Gaussian**
- `r` — rank (hyperparameter, typically 4–128)
- `α` — scaling numerator (commonly `α = r` or `α = 2r`; some recipes fix α = 16 or 32)
- Effective scaling: `α/r` multiplies `BA`, so changing r does not change the initial update size much

The product `BA` has rank at most `r`, which is ≪ min(d, k) for large weight matrices. Why should that be enough? Aghajanyan et al. (2020) showed fine-tuning has a low **intrinsic dimension** — a few thousand well-chosen directions recover most of the gain — and the LoRA paper found the learned ΔW concentrates its energy in a handful of singular directions. The lab below lets you see that trade-off on a synthetic update with a decaying spectrum:

<div class="lab" data-lab="lora"><noscript>Interactive lab: LoRA parameter counts per target module, and the fraction of a weight update's energy captured at each rank. Enable JavaScript to use it.</noscript></div>

### Tensor Shapes Through LoRA

<div class="diagram">
<div class="diagram-title">LoRA Forward Pass — Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Input x</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">k</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">e.g. [4, 2048, 4096]</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">W·x (frozen path)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">no gradient flows here</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">A·x (down-project)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">r</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">e.g. [4, 2048, 16]</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">B·(A·x) (up-project)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">LoRA delta</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Output (W + α/r·BA)·x</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">d</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">frozen + trainable combined</span>
  </div>
</div>
</div>

### Which Layers to Apply LoRA To

LoRA can be applied to any linear layer. Common choices:

| Target modules (Llama 3.1 8B) | Params per unit rank (all 32 layers) | Notes |
|----------------|:--------------:|-------|
| `q_proj`, `v_proj` | 425,984 | The original LoRA paper's choice (on GPT-3) |
| All attention: `q, k, v, o` | 851,968 | Common in 2023 recipes |
| All linear: attention + `gate, up, down` (`"all-linear"`) | 2,621,440 | **Recommended**: QLoRA (2023) and *LoRA Without Regret* (2025) found MLP LoRA matters most |

The MLP projections hold ~80% of a dense model's parameters ([Chapter 3](./03_the_transformer.md)), so leaving them out leaves most of the model unadaptable.

### Rank Selection

| Rank (r) | Trainable params (all-linear, 8B) | Typical use |
|:--------:|:---------------------:|----------|
| 1–4 | 2.6–10.5M | RL fine-tuning (policy gradients carry few bits per episode) |
| 8–16 | 21–42M | Instruction following, style, formatting |
| 32–64 | 84–168M | Larger SFT/distillation sets, code, domain data |
| 128–256 | 336–671M | Approaching full fine-tuning capacity; consider full FT instead |

**Rule of thumb**: capacity should scale with the *information* in your dataset — tokens of supervised data, not number of examples. Start at r = 16 on all linear layers; raise it if training loss plateaus above a full fine-tune's.

### LoRA Merging

At inference time, LoRA adapters can be **merged** back into the base weights with zero overhead:

```python
W_merged = W + (alpha / r) * B @ A  # shape [d, k]
```

After merging, there's no inference latency penalty. Alternatively, keep adapters **separate** and serve many of them on one base model: S-LoRA and Punica-style kernels (now built into vLLM and SGLang) batch requests for different adapters together, so one GPU can host hundreds of customer fine-tunes.

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
  <div class="flow-node green wide"><strong>Paged Optimisers</strong><br>Optimizer state lives in CUDA unified memory and pages out to CPU RAM during memory spikes (e.g., an unusually long training sequence) instead of crashing with out-of-memory</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide"><strong>LoRA Adapters (BF16)</strong><br>Trainable adapters remain in full precision; gradients computed in BF16</div>
</div>
</div>

### NF4 — Normal Float 4

NF4 is theoretically optimal for weights assumed to be normally distributed (N(0, σ²)):

- The 16 levels are placed at **quantiles** of a standard normal distribution, rescaled to [−1, 1], with an exact zero
- Each block of 64 weights is scaled by its absolute maximum, so the levels adapt to each block
- Equal probability mass per level means a normally distributed weight block uses all 16 codes evenly — the information-theoretically efficient choice for that distribution, and measurably better than uniform INT4 on real weights ([Chapter 20](./20_quantization_fundamentals.md))

The forward pass dequantises weights on-the-fly during matrix multiplication:

```
W_nf4: [d, k] in NF4 (4 bits/param)
  ↓ dequantize
W_bf16: [d, k] in BF16  ← temporary, for this operation only
  ↓ matmul with activations
output: [B, T, d] in BF16
```

### QLoRA Memory Budget (Llama 3.1 8B, r = 16 all-linear)

| Component | Precision | Memory |
|-----------|-----------|:------:|
| Linear-layer weights (7.0B params) | NF4 + block scales | ~3.7 GB |
| Embedding + LM head (1.05B params) | BF16 (not quantized) | 2.1 GB |
| LoRA A/B matrices (41.9M) | BF16 | 84 MB |
| LoRA gradients + AdamW m, v | BF16 + FP32 | ~420 MB |
| Activations (batch 4 × 2K tokens, gradient checkpointing) | BF16 | ~2–4 GB |
| **Total** | — | **~8–10 GB** |

Compare full fine-tuning at ~128 GB of state. QLoRA fits comfortably on a single 16–24 GB GPU. The price: dequantising weights in every forward and backward pass makes QLoRA noticeably slower per step than BF16 LoRA.

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
# trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196
```

---

## DoRA — Weight-Decomposed LoRA

**DoRA** (Liu et al., 2024) decomposes weights into **magnitude** and **direction**, applying LoRA only to the directional component:

$$W' = \underbrace{m}_{\text{learnable magnitude}} \cdot \underbrace{\frac{W + BA}{\|W + BA\|_c}}_{\text{normalised direction (LoRA-updated)}}$$

where `‖·‖_c` denotes column-wise normalisation and `m ∈ ℝ^(1×k)` is a learnable vector.

**Why it works**: the DoRA authors found that full fine-tuning often changes a weight column's direction a lot while barely changing its magnitude (or vice versa), whereas LoRA couples the two. Separating them gives LoRA a pattern of updates closer to full fine-tuning.

**Benchmarks**: in the paper DoRA beats LoRA at the same rank on commonsense reasoning and visual instruction tuning, by roughly 1–4 points, narrowing (not always closing) the gap to full fine-tuning. It costs extra training memory and time for the norm computation; after training it merges like LoRA.

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
| **PiSSA** | Initialise A, B from W's top singular vectors; freeze the residual | r(d+k) | Faster convergence | Better start than random A |
| **LoftQ / QA-LoRA** | Initialise or constrain LoRA to compensate quantization error | r(d+k) | Better than QLoRA at 2–3 bits | Quantization-aware PEFT |

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

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>RL fine-tuning works with LoRA of rank 1, while supervised distillation of long reasoning traces needs rank 64 or more. Count the bits: how much information does a scalar reward per episode carry, compared with a full target token sequence — and what does that suggest about where a model's new abilities come from in each case?</p>
</div>

## Key Takeaways

- LoRA adds (α/r)·BA to frozen weights; B starts at zero, A random; only A and B train, and they can be merged or hot-swapped.
- Target all linear layers: on Llama 3.1 8B that is 2.62M parameters per unit rank (r = 16 → 41.9M, 0.52%).
- QLoRA stores linear weights in NF4 (quantile levels, block-64 absmax, double-quantized scales) and trains BF16 adapters — an 8B fine-tune in ~8–10 GB, at some speed cost.
- DoRA, rsLoRA, LoRA+, PiSSA and LoftQ refine scaling, initialisation and quantization compensation; plain all-linear LoRA with a tuned LR is a strong default.
- Adapters, prefix and prompt tuning still exist but are rarely chosen for LLMs in 2026; multi-LoRA serving made adapters the unit of customisation.

## Check Yourself

<details class="check">
<summary>What is ΔW's rank for LoRA with r = 16 on a 4096 × 14336 projection, and how many parameters does it cost?</summary>
<p>Rank at most 16. Parameters: 16 × (4,096 + 14,336) = 294,912, versus 58.7M for the full matrix — about 0.5%.</p>
</details>

<details class="check">
<summary>Why does rsLoRA scale by α/√r instead of α/r?</summary>
<p>With α/r, the update's magnitude shrinks as r grows (the sum of r random rank-1 terms grows like √r, then is divided by r), so high-rank adapters learn slowly and gain little. Scaling by 1/√r keeps the update size stable across ranks, so raising r actually adds capacity.</p>
</details>

<details class="check">
<summary>Why are QLoRA training steps slower than BF16 LoRA steps despite using less memory?</summary>
<p>Every matmul with a frozen weight first dequantizes its NF4 block to BF16, in both the forward and backward passes, and the 4-bit kernels are less efficient than dense BF16 GEMMs. Memory falls ~3×; step time rises.</p>
</details>

<details class="check">
<summary>How do you serve 200 different customer fine-tunes of one 8B model on one GPU?</summary>
<p>Keep them as separate LoRA adapters (~84 MB each at r = 16) on a single shared base, and use a multi-LoRA serving engine (S-LoRA/Punica kernels in vLLM or SGLang) that batches requests for different adapters together, paging adapter weights in and out of GPU memory as needed.</p>
</details>

<details class="check">
<summary>A colleague applies LoRA only to q_proj and v_proj, sees poor results, and raises the rank to 256. What should they try first?</summary>
<p>Target all linear layers (especially the MLP projections, which hold most parameters) and raise the learning rate to around 10× the full fine-tuning LR; that usually beats a very high rank on two attention matrices.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning | Aghajanyan et al., 2020 | [arxiv.org/abs/2012.13255](https://arxiv.org/abs/2012.13255) | Why low rank suffices |
| LoRA | Hu et al., 2021 | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) | LoRA |
| QLoRA | Dettmers et al., 2023 | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) | NF4, double quantization, paged optimizers |
| DoRA | Liu et al., 2024 | [arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353) | Magnitude–direction decomposition |
| A Rank Stabilization Scaling Factor for LoRA (rsLoRA) | Kalajdzievski, 2023 | [arxiv.org/abs/2312.03732](https://arxiv.org/abs/2312.03732) | 1/√r scaling |
| LoRA+ | Hayou et al., 2024 | [arxiv.org/abs/2402.12354](https://arxiv.org/abs/2402.12354) | Separate LRs for A and B |
| PiSSA | Meng et al., 2024 | [arxiv.org/abs/2404.02948](https://arxiv.org/abs/2404.02948) | SVD initialisation |
| LoftQ | Li et al., 2023 | [arxiv.org/abs/2310.08659](https://arxiv.org/abs/2310.08659) | Quantization-aware LoRA init |
| S-LoRA: Serving Thousands of Concurrent LoRA Adapters | Sheng et al., 2023 | [arxiv.org/abs/2311.03285](https://arxiv.org/abs/2311.03285) | Multi-LoRA serving |
| Parameter-Efficient Transfer Learning for NLP (Adapters) | Houlsby et al., 2019 | [arxiv.org/abs/1902.00751](https://arxiv.org/abs/1902.00751) | Bottleneck adapters |
| Prefix-Tuning | Li & Liang, 2021 | [arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190) | Prefix tuning |
| The Power of Scale for Parameter-Efficient Prompt Tuning | Lester et al., 2021 | [arxiv.org/abs/2104.08691](https://arxiv.org/abs/2104.08691) | Prompt tuning |
| LoRA Without Regret | Thinking Machines, 2025 | [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) | All-linear targeting, rank for RL |

---

**Next:** [Chapter 15 — Alignment: RLHF & Beyond](./15_alignment_rlhf_and_beyond.md) — from imitation to optimisation: rewards, preferences and reinforcement learning.

[← Back to Table of Contents](./README.md)
