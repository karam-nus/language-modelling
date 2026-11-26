---
title: "Deep Dive — Looped & Recurrent-Depth Transformers"
description: "Buying depth without buying parameters: Universal Transformers and ALBERT, why looped models are good at reasoning, the prelude–recurrent–coda design of Huginn, Mixture-of-Recursions, HRM and the 7M-parameter TRM, ByteDance's Ouro LoopLMs, adaptive exits, the real compute, memory and KV-cache costs, and how latent loops compare with chain of thought."
related:
  - { id: models, why: "Looped and recurrent designs next to the rest of the architecture history." }
  - { id: inference, why: "What variable-depth models mean for batching and serving." }
---

[← Back to Table of Contents](./README.md)

# Deep Dive — Looped & Recurrent-Depth Transformers

A standard transformer runs each token through L layers, each with its own weights: depth and parameter count grow together. A **looped** transformer runs the *same* block several times. Depth — the amount of serial computation per token — becomes a dial you can turn at training and even at inference time, while the parameter count stays fixed.

> **The one-sentence version:** looping a block buys depth without buying parameters, and when the number of loops can vary per token it becomes test-time compute that happens inside the hidden state instead of in generated text.

## Depth Without Parameters

<div class="diagram">
<div class="diagram-title">Three Ways to Get 12 Layers of Depth</div>
<div class="fig" data-fig="loopstack"><noscript>Figure: a standard 12-layer stack with 12 sets of weights; a 3-layer block looped 4 times with 3 sets of weights; and the prelude, recurrent block and coda design, where the input is re-injected at every pass.</noscript></div>
</div>

Write a block as a function of the hidden states, h ← B(h). A standard model computes B₁₂(…B₂(B₁(h))) with twelve different functions. A looped model computes B(B(B(…B(h)))) with one. Three things change:

| | Standard (L layers) | Looped (k-layer block × r) |
|---|---|---|
| Parameters in the stack | L × per-layer | k × per-layer |
| Serial depth per token | L | k · r — and r can change |
| FLOPs per token (≈ 2 × params touched) | 2 · L · p_layer | 2 · k · r · p_layer |
| Weight bytes read per token at batch 1 | L · p_layer · bytes | k · r · p_layer · bytes (re-read every pass) |
| Can spend more compute on a hard token? | No | Yes, if r varies per token |

The last two rows are the trade. Looping saves **memory capacity**, not compute and not necessarily memory **bandwidth**: during decode every pass re-reads the block's weights from HBM, unless the block is small enough to stay on chip. The win is a model that fits where a larger one would not, plus a knob for spending more computation exactly where it helps.

## A Short History

| Year | Work | Idea |
|---|---|---|
| 2016 | Adaptive Computation Time (Graves) | An RNN learns a halting probability per step, so the number of steps adapts to the input |
| 2018 | **Universal Transformer** (Dehghani et al.) | One transformer block applied recurrently over depth, with ACT deciding per position when to stop |
| 2019 | **ALBERT** (Lan et al.) | Share all layers' weights to shrink BERT by an order of magnitude, at a modest quality cost |
| 2021 | PonderNet (Banino et al.) | A probabilistic, better-behaved halting mechanism |
| 2023 | Looped Transformers as Programmable Computers (Giannou et al.) | A small looped transformer can emulate a general-purpose computer |
| 2023–24 | Looped transformers for in-context algorithms and length generalisation | Loops learn iterative algorithms and extrapolate to longer inputs by running more iterations |
| 2024 | Relaxed Recursive Transformers (Bae et al.) | Convert a pretrained LLM into a looped one; per-loop LoRA "relaxes" the sharing |
| 2025 | **Reasoning with Latent Thoughts** (Saunshi et al.) | Reasoning needs depth more than parameters; a k-layer model looped L times nearly matches a kL-layer model |
| 2025 | **Huginn** (Geiping et al.) | 3.5B recurrent-depth LM trained on 800B tokens; more loops at test time keep improving reasoning |
| 2025 | **Mixture-of-Recursions** (Bae et al.) | A router gives each token its own recursion depth |
| 2025 | **HRM** and **TRM** | Tiny recursive models (27M, then 7M parameters) that iterate on puzzles like ARC-AGI and Sudoku |
| 2025 | **Ouro** (ByteDance Seed) | LoopLMs at 1.4B and 2.6B with a learned exit, trained as looped models from the start |
| 2026 | Stability and scaling-law studies | When is one more loop worth one more layer? How to keep deep loops stable? |

## Why Loops Help Reasoning

Saunshi et al. (2025) made the argument precise. Many reasoning problems — multi-step arithmetic, following a chain of pointers, composing functions — need a certain number of **serial steps**. A transformer performs a fixed number of serial steps per token: its depth. Two ways to get more:

- **Chain of thought** adds serial steps by generating tokens: each new token gets a full pass through the model and can read everything written before.
- **Looping** adds serial steps inside one forward pass: the same block refines the hidden state again.

They proved that a looped model can simulate T steps of chain of thought with T loops, and found empirically that, at equal perplexity, looped models do **better on reasoning** benchmarks than non-looped ones — while needing more parameters to match them on pure memorisation. Ouro's authors reach a compatible conclusion at 1.4–2.6B parameters: their looped models' advantage comes from better *manipulation* of knowledge rather than from storing more of it.

## Anatomy of a Recurrent-Depth LM

Huginn's design is now the common template:

<div class="diagram">
<div class="diagram-title">Prelude, Recurrent Block, Coda</div>
<div class="tensor-flow">
  <div class="tensor-flow-row"><span class="ts-label">tokens</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-bracket">]</span></div><span class="ts-note">input ids</span></div>
  <div class="tensor-flow-row"><span class="ts-label">prelude P → e</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim hidden">d</span><span class="ts-bracket">]</span></div><span class="ts-note">embedding + a few layers, run once</span></div>
  <div class="tensor-flow-row"><span class="ts-label">state s₀</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim hidden">d</span><span class="ts-bracket">]</span></div><span class="ts-note">random or zero initial state</span></div>
  <div class="tensor-flow-row"><span class="ts-label">sᵢ = R(sᵢ₋₁, e)</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim hidden">d</span><span class="ts-bracket">]</span></div><span class="ts-note">same weights, i = 1 … r; e re-injected every pass</span></div>
  <div class="tensor-flow-row"><span class="ts-label">coda C → logits</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim vocab">V</span><span class="ts-bracket">]</span></div><span class="ts-note">a few layers + LM head, run once</span></div>
</div>
</div>

Three design choices matter:

- **Input injection.** The prelude's output e is fed into every pass (concatenated with the state and projected, or added). Without it the state can drift away from the input over many loops; with it, each pass is a function of "what I was given" and "what I have worked out so far" — the structure of an iterative solver.
- **Where the loop sits.** Looping only the middle (prelude and coda unshared) keeps token-specific input and output processing out of the loop; Ouro instead loops its whole layer stack.
- **Normalisation and scale.** A block applied 32 times can blow up or collapse. Sandwich normalisation, careful residual scaling and small initial updates are what make deep loops trainable; much of the 2026 work on looped models is about this stability.

A minimal looped block in NumPy, with a per-token exit when the state stops changing:

```python
import numpy as np

rng = np.random.default_rng(0)
d, T = 64, 6                                         # hidden size, tokens

Q, _ = np.linalg.qr(rng.standard_normal((d, d)))     # orthogonal matrix
A = 0.9 * Q                                          # recurrent weights, spectral norm 0.9 → a contraction
B = rng.standard_normal((d, d)) / np.sqrt(d)         # input-injection weights

def block(h, e):
    """The same weights at every pass; e (the prelude's output) is re-injected each time."""
    return np.tanh(h @ A + e @ B)

e = rng.standard_normal((T, d)) * np.array([0.3, 0.6, 1.0, 1.5, 2.5, 4.0])[:, None]   # [T, d]
h = np.zeros((T, d))                                 # initial state
active, loops = np.ones(T, bool), np.zeros(T, int)

for r in range(1, 65):                               # at most 64 passes
    h_new = block(h, e)
    delta = np.linalg.norm(h_new - h, axis=-1) / np.linalg.norm(h_new, axis=-1)
    h = np.where(active[:, None], h_new, h)          # exited tokens keep their state
    loops[active] = r
    active &= delta > 1e-3                           # exit once an update changes < 0.1% of the state
    if not active.any():
        break

print("recurrent parameters:", A.size + B.size)
print("passes per token:    ", loops.tolist())
print("FLOPs vs. always 64: ", f"{loops.mean() / 64:.0%}")
# recurrent parameters: 8192
# passes per token:     [27, 20, 14, 10, 11, 10]
# FLOPs vs. always 64:  24%
```

Tokens whose inputs push the block into saturation settle in about 10 passes; the weakest-driven token needs 27. Real models make the same decision with a learned signal instead of a hand-set contraction, but the economics are the same: compute follows difficulty.

## Training a Looped Model

| Choice | Options | Who does it |
|---|---|---|
| Number of loops during training | Fixed r; or sampled per step so the model works at any depth | Ouro: fixed (4); Huginn: random, heavy-tailed (mean 32) |
| Backpropagation | Through every pass (memory grows like depth) or only the last few (truncated) | Huginn: only the last 8 passes |
| Supervision | Loss only after the last pass, or after every pass ("deep supervision") | TRM, Ouro (loss at every exit) |
| Learning when to stop | Halting probability (ACT, PonderNet), exit gate trained with an entropy penalty, or a router choosing depth up front | Ouro: learned exit distribution; MoR: router |
| Starting point | From scratch, or convert a pretrained model by sharing its layers | Relaxed Recursive Transformers convert and add per-loop LoRA |

Without a pressure to stop early, a learned exit always picks the maximum depth — more compute never hurts the training loss. Ouro adds an **entropy regulariser** on the exit distribution so that the model spreads probability across depths and learns to stop when further loops stop paying off.

## Adaptive Depth: Who Gets More Loops?

<div class="lab" data-lab="loop"><noscript>Interactive lab: a toy recurrent block run per token until its update falls below a threshold; shows loops per token, convergence curves and the compute saved. Enable JavaScript to use it.</noscript></div>

Three families of exit rule:

1. **Convergence.** Stop when the state (or the output distribution) stops changing — Huginn measures the KL divergence between successive passes' next-token distributions. Needs no extra parameters; works because a well-trained recurrent block behaves like an iterative solver.
2. **Learned halting.** A small head predicts the probability of stopping after each pass (ACT, PonderNet, Ouro). Trained jointly, so it can learn when *more* loops would actually change the answer.
3. **Routing up front.** Mixture-of-Recursions' router assigns each token a depth before the loops start, so the batch's shape is known in advance — easier to run efficiently, at the cost of deciding before seeing the intermediate states.

## What It Costs to Run

| Resource | Looped vs a dense model of the same depth |
|---|---|
| Weight memory | **k/L** of it — the main win |
| FLOPs per token | The same (each pass costs a layer's worth) |
| Weight reads at batch 1 | The same, unless the block fits in on-chip memory |
| KV cache | Up to r × the block's layers if every pass keeps its own K and V; can be cut by sharing or overwriting across passes |
| Batching | Harder with per-token exits: tokens in a batch finish at different depths |

The **KV cache** deserves care. If each of the r passes writes keys and values for every block layer, a looped model's cache is as large as the dense model it imitates. The fixes are to let later passes reuse one cache slot per layer (keeping only the latest pass, or the first, as Mixture-of-Recursions' KV-sharing variant does) at a small accuracy cost, or to cache only the prelude and coda.

**Batching** is the engineering problem behind the slow adoption. A serving engine wants every sequence in a batch to run the same kernels; per-token exits break that. Mixture-of-Recursions keeps a fixed depth per token chosen up front, and 2025–26 work on "continuous depth batching" refills slots freed by exited tokens with new work. At the time of writing, popular engines run looped models at their maximum depth — Ouro's model card notes that vLLM ignores the adaptive exit.

**Worked example.** Loop a 1.4B-parameter stack 4 times. Per token that is about 2 × 1.4 × 10⁹ × 4 ≈ 11 GFLOPs — the compute of a dense ~5.6B model — but the weights occupy 2.8 GB in BF16 instead of 11.2 GB. At batch 1 on a GPU the four passes re-read those 2.8 GB four times, so decode speed is close to the 5.6B model's; the saving is in memory, which is what decides whether a model fits on a phone or a single consumer GPU.

## The 2025–26 Models

| Model | Size | Loop structure | Notable result |
|---|---|---|---|
| **Huginn-0125** (Geiping et al.) | 3.5B, 800B tokens | Prelude → recurrent block (random depth in training) → coda | Reasoning keeps improving with more test-time loops, up to compute equivalent to a ~50B model |
| **Mixture-of-Recursions** (KAIST, Mila, Google) | 135M – 1.7B | Shared stack; router picks depth per token; attention only among active tokens | Better perplexity and throughput than vanilla and recursive baselines at equal training FLOPs |
| **HRM** (Sapient) | 27M | Two coupled recurrent modules at different speeds | Strong ARC-AGI, Sudoku and maze results from ~1,000 training examples |
| **TRM** (Samsung SAIL Montréal) | 7M | One 2-layer network recursing on a latent and an answer, with deep supervision | 45% on ARC-AGI-1 and 8% on ARC-AGI-2 — beating far larger LLMs on these puzzles |
| **Ouro** (ByteDance Seed) | 1.4B, 2.6B | Whole stack looped up to 4 times, learned exit with entropy regularisation | Rivals dense models several times larger on reasoning-heavy benchmarks |

HRM and TRM are task-specific puzzle solvers trained on small datasets, not language models; a follow-up analysis attributes part of TRM's ARC score to task conditioning and heavy test-time computation. They belong here because they show how much iterative depth can do with very few parameters.

## Latent Loops vs Chain of Thought

| | Chain of thought (Chapter 35) | Looped depth |
|---|---|---|
| Where the extra computation happens | In generated tokens | In the hidden state |
| Serial steps available | Unbounded — as many tokens as the budget allows | Bounded by the maximum number of loops |
| Cost per step | One full forward pass per token, plus KV cache | One pass of the block |
| Can humans read it? | Yes (if faithful) — useful for monitoring | No — the reasoning is a vector |
| Parallel over the prompt? | No: tokens are generated one by one | Yes: prefill loops over all positions at once |

The two compose: Ouro ships "Thinking" variants that loop *and* write chain of thought. The open question is the trade between them — loops are cheaper and parallel, tokens are unbounded and inspectable — and the monitorability cost of moving reasoning into hidden states ([Chapter 38](./38_frontier.md)).

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A looped model that thinks for longer by looping more is doing test-time compute that no one can read. If future reasoning models move most of their thinking from text into loops, what replaces chain-of-thought monitoring as a safety tool — and would you accept a less capable model to keep the reasoning visible?</p>
</div>

## Key Takeaways

- Looping a k-layer block r times gives depth k·r with the parameters of k layers; FLOPs per token follow depth, not parameters.
- Reasoning tasks need serial depth more than parameters; looped models can simulate chain-of-thought steps internally (Saunshi et al., 2025).
- The common design is prelude → recurrent block with input injection → coda (Huginn); training samples or fixes the loop count, often with truncated backpropagation and losses at every exit.
- Adaptive depth — convergence-based exits, learned halting (Ouro) or up-front routing (Mixture-of-Recursions) — spends compute where tokens are hard.
- The win is weight memory. FLOPs, batch-1 weight reads and (without sharing) the KV cache scale with the number of loops, and per-token exits complicate batching.

## Check Yourself

<details class="check">
<summary>A 2-layer block looped 12 times and a 24-layer model have the same per-layer size. Compare their parameters, FLOPs per token and KV cache (without sharing).</summary>
<p>Parameters: the looped model has 2/24 = 1/12 of the stack's parameters. FLOPs per token: the same — 24 block-layer passes either way. KV cache without sharing: also the same, because each of the 24 passes writes keys and values. Only weight memory shrinks unless the cache is shared across passes.</p>
</details>

<details class="check">
<summary>Why does a learned exit need an entropy (or compute) penalty during training?</summary>
<p>More loops never make the training loss worse in expectation, so without a penalty the exit head learns to always use the maximum depth. An entropy regulariser on the exit distribution (Ouro) or an explicit compute cost (ACT, PonderNet) makes stopping early worth something, so the model learns to stop when extra loops stop changing its prediction.</p>
</details>

<details class="check">
<summary>Huginn samples a random number of loops at every training step. What does that buy at inference time?</summary>
<p>The model learns to produce a usable state after any number of passes and to keep improving with more, so at inference you can choose the depth — few loops for easy prompts, many for hard ones — instead of being locked to the depth used in training.</p>
</details>

<details class="check">
<summary>A looped model runs on a phone with weights in DRAM. Does looping make each generated token cheaper than a dense model of the same depth?</summary>
<p>Not much. Each pass re-reads the block's weights, so the bytes read per token — which set decode speed when memory-bound — are about the same as for the dense model. The saving is capacity: a model that would not fit in the phone's memory now does. Only blocks small enough to stay in on-chip SRAM also save bandwidth.</p>
</details>

<details class="check">
<summary>Why is per-token adaptive depth hard for serving engines, and how does Mixture-of-Recursions sidestep part of the problem?</summary>
<p>Engines batch many sequences through the same kernels; if tokens exit at different loops, the batch shrinks unevenly and the GPU runs partly empty or needs complex re-packing. Mixture-of-Recursions assigns each token's depth with a router before looping, so the number of tokens active at each depth is known up front and attention runs only among the active tokens.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this page takes from it |
|---|---|---|---|
| Adaptive Computation Time for Recurrent Neural Networks | Graves, 2016 | [arxiv.org/abs/1603.08983](https://arxiv.org/abs/1603.08983) | Learned halting |
| Universal Transformers | Dehghani et al., 2018 | [arxiv.org/abs/1807.03819](https://arxiv.org/abs/1807.03819) | Recurrence over depth with ACT |
| ALBERT | Lan et al., 2019 | [arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942) | Cross-layer parameter sharing |
| PonderNet: Learning to Ponder | Banino et al., 2021 | [arxiv.org/abs/2107.05407](https://arxiv.org/abs/2107.05407) | Probabilistic halting |
| Looped Transformers as Programmable Computers | Giannou et al., 2023 | [arxiv.org/abs/2301.13196](https://arxiv.org/abs/2301.13196) | Expressivity of looping |
| Relaxed Recursive Transformers | Bae et al., 2024 | [arxiv.org/abs/2410.20672](https://arxiv.org/abs/2410.20672) | Converting LLMs to looped models |
| Reasoning with Latent Thoughts: On the Power of Looped Transformers | Saunshi et al., 2025 | [arxiv.org/abs/2502.17416](https://arxiv.org/abs/2502.17416) | Depth vs parameters; loops simulate CoT |
| Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach | Geiping et al., 2025 | [arxiv.org/abs/2502.05171](https://arxiv.org/abs/2502.05171) | Huginn: prelude, recurrent block, coda |
| Mixture-of-Recursions | Bae et al., 2025 | [arxiv.org/abs/2507.10524](https://arxiv.org/abs/2507.10524) | Token-level recursion depth, KV sharing |
| Hierarchical Reasoning Model | Wang et al., 2025 | [arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734) | Two-timescale recurrent model |
| Less is More: Recursive Reasoning with Tiny Networks | Jolicoeur-Martineau, 2025 | [arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871) | TRM, 7M parameters |
| Scaling Latent Reasoning via Looped Language Models (Ouro) | Zhu et al. (ByteDance Seed), 2025 | [arxiv.org/abs/2510.25741](https://arxiv.org/abs/2510.25741) | LoopLMs with learned exits |
| Training LMs to Reason in a Continuous Latent Space (Coconut) | Hao et al., 2024 | [arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769) | Latent reasoning without loops |

---

**Next:** [Chapter 35 — Reasoning Models](./35_reasoning_models.md) — the other way to buy test-time compute: thinking in tokens, trained with reinforcement learning.

[← Back to Table of Contents](./README.md)
