---
title: "Chapter 3 — The Transformer"
description: "The decoder block with tensor shapes at every step: residual stream, RMSNorm and where to put it, QK-norm, SwiGLU feed-forward layers, a parameter count that adds up, and how 2026 blocks differ from 2017's."
---

[← Back to Table of Contents](./README.md)

# Chapter 3 — The Transformer

> **The one-sentence version:** a transformer is a stack of identical blocks, each of which reads from a shared residual stream, mixes information *across tokens* with attention, transforms each token *independently* with a feed-forward network, and writes the result back.

## The Architecture That Changed Everything

The **transformer** is the architecture behind almost every modern LLM. Introduced in 2017 for machine translation, it replaced recurrence and convolution with attention, which looks at all positions at once — so training parallelises across the whole sequence. This chapter walks through every component with tensor shapes at each step, then shows how the 2026 version of the block differs from the 2017 one.

<div class="diagram">
<div class="diagram-title">Transformer Block (Decoder-Only, Pre-Norm)</div>
<div class="flow">
  <div class="flow-node wide">Input: x <small>[B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">RMSNorm <small>[B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Multi-Head Self-Attention <small>[B, T, d_model] → [B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">+ Residual Connection <small>x + attn_out → [B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">RMSNorm <small>[B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Feed-Forward Network <small>[B, T, d_model] → [B, T, d_ff] → [B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">+ Residual Connection <small>x + ffn_out → [B, T, d_model]</small></div>
</div>
</div>

A transformer model is **N** of these blocks stacked. Llama 3.1 8B uses N = 32, Llama 3.1 70B N = 80, DeepSeek-V3 N = 61.

## Tensor Flow Through a Full Model

<div class="diagram">
<div class="diagram-title">End-to-End Tensor Shapes (Llama 3.1 8B)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Token IDs</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">e.g. [1, 512]</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Token Embeddings</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">4096</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After each Block ×32</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">4096</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Final RMSNorm</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">4096</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">LM Head (logits)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim vocab">128256</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
</div>
</div>

The hidden dimension stays constant at d_model through all transformer blocks. Only the final linear projection (the "LM head") changes the dimension to the vocabulary size.

## Component Deep Dive

<div class="img-caption">
  <img src="{{ '/assets/images/transformer_block.svg' | relative_url }}" alt="Decoder-only transformer block showing RMSNorm, Self-Attention, and Feed-Forward layers with residual connections, alongside a model scale comparison table">
  <figcaption>Complete decoder-only transformer block with tensor shapes and model scale reference (Llama family)</figcaption>
</div>

### Residual Connections

Every sub-layer (attention and FFN) uses a **residual (skip) connection**: the input to the sub-layer is added to its output. This enables gradient flow through deep networks and lets each layer learn a *delta* rather than a full transformation:

$$\text{output} = x + \text{SubLayer}(\text{Norm}(x))$$

Without residual connections, training a 32-layer or 80-layer model would be extremely difficult due to vanishing gradients.

A useful mental model, popularised by interpretability research, is the **residual stream**: a `[T, d_model]` scratchpad that runs from the embedding to the output head. Each attention and FFN sub-layer *reads* a normalised copy of it and *adds* a small update back. Nothing is ever overwritten — later layers see the sum of everything written before them.

<div class="callout trend">
<span class="callout-title">2026 trend — wider residual streams</span>
<p>Hyper-connections replace the single residual stream with several parallel streams and learned mixing between them. DeepSeek-V4 (2026) uses a constrained variant, <em>manifold-constrained hyper-connections</em>, which keeps the mixing matrices well-conditioned so that signals neither explode nor vanish across hundreds of layers. It is the first change to the residual path itself to reach a frontier open model since pre-norm.</p>
</div>

### Layer Normalization vs RMS Normalization

Normalization stabilizes training by keeping activations in a consistent range.

**LayerNorm** (original transformer, BERT):
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

**RMSNorm** (LLaMA, Mistral, modern models):
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

RMSNorm drops the mean-centring and the bias. It needs one reduction over the hidden dimension instead of two, trains as well as LayerNorm in practice, and is the standard in modern LLMs. (Inside a fused kernel the speed difference is small; the simplicity is the main win.)

| Property | LayerNorm | RMSNorm |
|----------|-----------|---------|
| Mean centring | Yes (−μ) | No |
| Learnable bias | Yes (β) | No |
| Learnable scale | Yes (γ) | Yes (γ) |
| Reductions over d | Two (mean, variance) | One (mean square) |
| Used by | BERT, GPT-2, GPT-3 | Llama, Qwen, Mistral, Gemma, DeepSeek, gpt-oss |

### Pre-Norm vs Post-Norm

The **placement** of normalization matters:

<div class="diagram">
<div class="diagram-title">Pre-Norm vs Post-Norm</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Pre-Norm (Modern)</div>
    <ul>
      <li>x → Norm → SubLayer → + residual</li>
      <li>More stable training, less warm-up sensitivity</li>
      <li>Used by GPT-2/3, Llama, Qwen, DeepSeek</li>
      <li>Gradients flow directly through the residual path</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Post-Norm (Original)</div>
    <ul>
      <li>x → SubLayer → + residual → Norm</li>
      <li>Requires careful LR warmup</li>
      <li>Used by original Transformer, BERT</li>
      <li>Can produce slightly better results with care</li>
    </ul>
  </div>
</div>
</div>

Pre-norm has a known weakness: because nothing normalises the residual stream itself, its magnitude grows with depth and late layers' updates become relatively small. Two 2024–25 fixes are now common:

- **Sandwich norm** (Gemma 2/3): normalise both the input *and* the output of each sub-layer, `x + Norm(SubLayer(Norm(x)))`.
- **Reordered post-norm** (OLMo 2): `x + Norm(SubLayer(x))` — the norm moves inside the residual branch, which OLMo 2 found markedly more stable at scale.

### QK-Norm

A second source of instability is inside attention: if query and key vectors grow large, the attention logits $q \cdot k$ explode, softmax saturates to one-hot and the loss spikes. **QK-norm** applies an RMSNorm to every query and key head before the dot product. It was rare in 2023 and is standard in 2025–26 designs — OLMo 2, Qwen3, Gemma 3 and Kimi K2-style "QK-clip" variants all control attention-logit growth one way or another.

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### Feed-Forward Network (FFN)

Each transformer block has a **position-wise feed-forward network** that operates independently on each token's representation. It projects up to a larger dimension, applies a non-linearity, and projects back down:

<div class="diagram">
<div class="diagram-title">FFN Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Input</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Up projection</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_ff</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">d_ff = 14336 for Llama 3.1 8B</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Down projection</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
</div>
</div>

### Activation Functions

The FFN's non-linearity determines what the model can learn:

| Activation | Formula | Used By |
|-----------|---------|---------|
| **ReLU** | $\max(0, x)$ | Original Transformer, OPT |
| **GELU** | $x \cdot \Phi(x)$ | GPT-2, GPT-3, BERT |
| **SwiGLU** | $\text{SiLU}(xW_1) \odot xW_3$ | Llama, Qwen, Mistral, DeepSeek, gpt-oss (clamped variant) |
| **GeGLU** | $\text{GELU}(xW_1) \odot xW_3$ | Gemma, T5 v1.1 |

**SwiGLU** (Shazeer, 2020) is the dominant choice. It is a *gated* linear unit: one projection passed through SiLU acts as a gate, a second projection carries the values, and they are multiplied element-wise. SiLU itself, $x \cdot \sigma(x)$, is the smooth non-linearity inside the gate.

```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)  # up projection
        self.w_down = nn.Linear(d_ff, d_model, bias=False)   # down projection

    def forward(self, x):  # x: [B, T, d_model]
        gate = torch.nn.functional.silu(self.w_gate(x))  # [B, T, d_ff]
        up = self.w_up(x)                                 # [B, T, d_ff]
        return self.w_down(gate * up)                      # [B, T, d_model]
```

SwiGLU has **three** weight matrices instead of two. To match the parameters of a classic 4·d FFN, d_ff shrinks by 2/3 to $\tfrac{8}{3}d$ — 10,923 for d = 4,096. Llama 3 then scales that by an `ffn_dim_multiplier` of 1.3 and rounds up to a multiple of 1,024, giving 14,336 = 3.5·d. So Llama 3.1 8B's FFN is *larger* than the classic 4·d one, not equal to it.

## A Minimal Transformer Block

Combining all components (attention is a placeholder here — see [Chapter 4](./04_attention_sdpa_and_mha.md)):

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)  # Chapter 4
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, d_ff)

    def forward(self, x, mask=None):
        # Pre-norm + attention + residual
        h = x + self.attn(self.attn_norm(x), mask=mask)   # [B, T, d_model]
        # Pre-norm + FFN + residual
        out = h + self.ffn(self.ffn_norm(h))               # [B, T, d_model]
        return out

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)           # [B, T, d_model]
        for layer in self.layers:
            x = layer(x, mask=mask)          # [B, T, d_model]
        x = self.norm(x)                     # [B, T, d_model]
        logits = self.lm_head(x)             # [B, T, vocab_size]
        return logits
```

## Parameter Count

Where the parameters of Llama 3.1 8B live. It uses grouped-query attention (Chapter 5): 32 query heads but only 8 key/value heads, each 128 wide, so the K and V projections are a quarter the size of Q.

| Component | Shape | Count |
|-----------|-------|------:|
| Token embedding | [128256, 4096] | 525.3M |
| Per block: Q projection | [4096, 32 × 128] | 16.8M |
| Per block: K and V projections | [4096, 8 × 128] × 2 | 8.4M |
| Per block: output projection | [32 × 128, 4096] | 16.8M |
| Per block: FFN (gate, up, down) | [4096, 14336] × 3 | 176.2M |
| Per block: 2 × RMSNorm | [4096] × 2 | 8K |
| **Per block total** | | **218.1M** |
| 32 blocks | | 6,979.6M |
| Final RMSNorm | [4096] | 4K |
| LM head (untied) | [4096, 128256] | 525.3M |
| **Total** | | **8,030M** |

The FFN holds **81%** of each block's parameters; attention holds 19% (it would be 28% with full multi-head K and V). Llama 3.1 8B does *not* tie its embedding and output head, so the two vocabulary matrices together are 13% of the model. Change the knobs yourself — heads, KV heads, FFN width, vocabulary, tying, experts — and watch the budget move:

<div class="lab" data-lab="params"><noscript>Interactive lab: a transformer parameter and memory counter with presets for real models. Enable JavaScript to use it.</noscript></div>

## The 2026 Block, Side by Side

The 2017 block and a 2026 block share the same skeleton. What changed is every detail:

| | Original (2017) | Llama 3.1 8B (2024) | Qwen3-8B (2025) | Gemma 3 27B (2025) | DeepSeek-V3 (2024) | gpt-oss-120b (2025) |
|---|---|---|---|---|---|---|
| Norm | LayerNorm, post | RMSNorm, pre | RMSNorm, pre | RMSNorm, pre + post | RMSNorm, pre | RMSNorm, pre |
| QK-norm | – | – | Yes | Yes | Norms on MLA latents | – |
| Attention | MHA | GQA 32/8 | GQA 32/8 | GQA, 5 local : 1 global | MLA | GQA 64/8, alternating 128-token window / full, learned sinks |
| FFN | ReLU | SwiGLU | SwiGLU | GeGLU | MoE: 1 shared + top-8 of 256 | MoE: top-4 of 128 |
| Positions | Sinusoidal | RoPE 500K | RoPE 1M | RoPE 10K local / 1M global | Decoupled RoPE + YaRN | RoPE + YaRN |
| Tied head | Yes | No | No | Yes | No | No |

Two lessons hide in this table. First, *stability* features (QK-norm, extra norms, logit control) were added because training runs at 10²⁴–10²⁵ FLOPs cannot afford loss spikes. Second, the FFN is the part that scales: from 2024 the largest models replace it with a **mixture of experts** ([Chapter 33](./33_mixture_of_experts.md)), so most of a model's parameters sit in FFN experts that each token only partly uses.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>The FFN treats every token independently, yet it holds four-fifths of a dense model's parameters. If attention is what moves information between tokens, what is all that FFN capacity storing — and what does it suggest that MoE models can hold most of it in experts a token only partly uses?</p>
</div>

## Key Takeaways

- A decoder block is `x + Attn(Norm(x))` then `x + FFN(Norm(x))`; the hidden width d_model is constant through the stack, and only the output head maps to vocabulary size.
- RMSNorm in pre-norm position is the default; QK-norm and extra (sandwich or reordered) norms were added in 2024–25 to keep huge training runs stable.
- SwiGLU's third matrix shrinks d_ff to about $\tfrac{8}{3}d$ for the same parameters, but real models pick d_ff by multiplier and rounding (Llama 3.1 8B: 3.5·d).
- In a GQA dense model the FFN holds ~80% of block parameters; MoE models move that share into experts.
- Count parameters honestly: K/V projections shrink with GQA, and untied models pay for the vocabulary twice.

## Check Yourself

<details class="check">
<summary>Llama 3.1 70B has d_model = 8,192, 64 query heads and 8 KV heads of width 128. How many parameters are in one block's attention projections?</summary>
<p>Q and O: 2 × 8,192 × 8,192 = 134.2M. K and V: 2 × 8,192 × 1,024 = 16.8M. Total ≈ 151M per block.</p>
</details>

<details class="check">
<summary>Why does pre-norm train more stably than post-norm, and what problem does it introduce?</summary>
<p>In pre-norm the residual path from input to output is an identity with only additions, so gradients reach early layers undamped. But the stream is never normalised, so its scale grows with depth and each later layer's (normalised) update is a smaller relative change — motivating sandwich and reordered-norm variants.</p>
</details>

<details class="check">
<summary>What failure does QK-norm prevent?</summary>
<p>Attention-logit growth: if ‖q‖ and ‖k‖ drift upward during training, q·k becomes huge, softmax turns one-hot, gradients through attention vanish or spike, and the loss diverges. Normalising q and k per head bounds the logits.</p>
</details>

<details class="check">
<summary>A SwiGLU FFN with d_model = 2,048 should have the same parameter count as a classic ReLU FFN with d_ff = 8,192. What d_ff should it use?</summary>
<p>Classic: 2 × 2,048 × 8,192 = 33.6M. SwiGLU: 3 × 2,048 × d_ff = 33.6M → d_ff = 5,461 (8/3 × 2,048), usually rounded to a hardware-friendly 5,504 or 5,632.</p>
</details>

<details class="check">
<summary>Where do the "8B" parameters of Llama 3.1 8B go, to the nearest few percent?</summary>
<p>FFNs ≈ 70% (5.64B), attention ≈ 17% (1.34B), embedding + untied output head ≈ 13% (1.05B); norms are negligible.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Attention Is All You Need | Vaswani et al., 2017 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) | The original block |
| Root Mean Square Layer Normalization | Zhang & Sennrich, 2019 | [arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467) | RMSNorm |
| On Layer Normalization in the Transformer Architecture | Xiong et al., 2020 | [arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745) | Pre-norm vs post-norm |
| GLU Variants Improve Transformer | Shazeer, 2020 | [arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202) | SwiGLU, GeGLU |
| A Mathematical Framework for Transformer Circuits | Elhage et al. (Anthropic), 2021 | [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html) | The residual-stream view |
| Scaling Vision Transformers to 22 Billion Parameters | Dehghani et al., 2023 | [arxiv.org/abs/2302.05442](https://arxiv.org/abs/2302.05442) | QK-norm against attention-logit growth |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | Llama 3 dimensions |
| 2 OLMo 2 Furious | OLMo Team, 2024 | [arxiv.org/abs/2501.00656](https://arxiv.org/abs/2501.00656) | Reordered norm and QK-norm for stability |
| Gemma 3 Technical Report | Gemma Team, 2025 | [arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786) | Sandwich norm, QK-norm, 5:1 local/global |
| Hyper-Connections | Zhu et al., 2024 | [arxiv.org/abs/2409.19606](https://arxiv.org/abs/2409.19606) | Multiple residual streams |

---

**Next:** [Chapter 4 — Attention: SDPA & Multi-Head](./04_attention_sdpa_and_mha.md) — the sub-layer that lets tokens read from each other.

[← Back to Table of Contents](./README.md)
