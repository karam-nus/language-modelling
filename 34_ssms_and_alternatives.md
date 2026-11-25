---
title: "Chapter 34 — SSMs, Linear Attention & Hybrids"
description: "Alternatives to quadratic attention and why hybrids won: state space models from S4 to Mamba-2, linear attention as a fixed-size memory, the delta rule and Gated DeltaNet, RWKV-7, a verified recall experiment showing what fixed state forgets, and the 2025–26 hybrid wave (Qwen3-Next/3.5, Kimi Linear and K3, Nemotron-H, Granite 4, MiniMax) — plus diffusion language models."
---

[← Back to Table of Contents](./README.md)

# Chapter 34 — SSMs, Linear Attention & Hybrids

Attention compares every token with every earlier token: O(T²) work in training and a KV cache that grows with every token in inference. Recurrent alternatives keep a **fixed-size state** instead. For years they lost to transformers on quality; in 2025–26 they won a place inside frontier models — not by replacing attention, but by doing most of its work.

> **The one-sentence version:** a linear-time layer compresses the past into a fixed-size state that is cheap but forgetful, an attention layer keeps everything but costs memory and time — so the best 2026 designs use mostly the first and a little of the second.

## The Quadratic Attention Problem

- **Compute**: O(T²·d) per layer for a full sequence; FlashAttention removes the memory blow-up but not the FLOPs.
- **Inference memory**: the KV cache grows linearly with context — at 1M tokens it dwarfs the weights ([Chapter 17](./17_kv_cache_mechanics.md)).
- **Decode cost per token**: O(T) — reading the whole cache — so long generations slow down as they grow.

## State Space Models

An SSM maps a sequence through a latent state:

$$h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \qquad y_t = C\,h_t + D\,x_t$$

with $h_t \in \mathbb{R}^N$ per channel and $\bar A, \bar B$ discretised from continuous-time parameters by a step size Δ.

<div class="diagram">
<div class="diagram-title">SSM vs Attention</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Attention</div>
    <ul>
      <li>Every token reads every earlier token</li>
      <li>KV cache grows with sequence length</li>
      <li>O(T²·d) training compute per layer</li>
      <li>Exact retrieval of anything in context</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">SSM / linear recurrence</div>
    <ul>
      <li>Past compressed into a fixed-size state</li>
      <li>Constant memory and time per generated token</li>
      <li>O(T·d·N) training compute per layer</li>
      <li>Lossy: can forget or blur specific facts</li>
    </ul>
  </div>
</div>
</div>

### S4 (2022)

S4 made SSMs competitive on long-range benchmarks: **HiPPO** initialisation of A for long memory; training as a long **convolution** (computed with FFTs, O(T log T)) because a time-invariant recurrence unrolls into one; inference as a recurrence, O(1) per step.

### Mamba: Selective State Spaces (2023)

Mamba (Gu & Dao) made B, C and Δ **functions of the input** — the model decides per token what to write into the state and what to forget. That breaks the convolution trick, so Mamba trains with a hardware-aware parallel **scan** kernel.

<div class="diagram">
<div class="diagram-title">Mamba Block</div>
<div class="flow">
  <div class="flow-step accent">Input: [B, T, d]</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">Linear projection → [B, T, 2·d_inner] (main path + gate)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">Short causal 1D convolution</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step orange">Selective SSM: Δ, B, C computed from the input</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">Gate ⊙ SSM output → linear projection → [B, T, d]</div>
</div>
</div>

```python
# Selective SSM, conceptually (one layer; real kernels fuse and parallelise this)
def selective_ssm(x, A, D, dt_proj, B_proj, C_proj):
    """x: [B, T, d_inner]; A: [d_inner, N] (negative real)."""
    delta = F.softplus(dt_proj(x))                        # [B, T, d_inner] — how much to update / forget
    Bt, Ct = B_proj(x), C_proj(x)                         # [B, T, N] each — what to write, what to read
    A_bar = torch.exp(delta.unsqueeze(-1) * A)            # [B, T, d_inner, N] — per-token decay
    B_bar = delta.unsqueeze(-1) * Bt.unsqueeze(2)         # [B, T, d_inner, N]
    return selective_scan(x, A_bar, B_bar, Ct, D)         # parallel scan → [B, T, d_inner]
```

Mamba-3B matched transformers of twice its size on language modelling, with about 5× higher inference throughput. **Mamba-2** (2024) showed that selective SSMs are a form of structured linear attention ("state space duality"), letting them run as chunked matrix multiplications on tensor cores — 2–8× faster than Mamba-1's scan.

## Linear Attention as Memory

Replace softmax with a feature map φ and reorder the product:

$$\text{LinearAttn}(Q,K,V)_t = \phi(q_t)^\top \sum_{i \le t} \phi(k_i)\, v_i^\top = \phi(q_t)^\top S_t, \qquad S_t = S_{t-1} + \phi(k_t)\,v_t^\top$$

The running sum $S_t$ is a fixed d×d matrix — an **associative memory** that stores key–value pairs by adding their outer products. Reading it with a query returns a blend of stored values weighted by key similarity. The catch: a d×d matrix can hold only about d independent associations; beyond that, writes interfere.

### The Delta Rule and Gated DeltaNet

**DeltaNet** writes with the *delta rule*: before adding a pair, subtract what the memory currently returns for that key, so a write *corrects* rather than piles up:

$$S_t = S_{t-1} + \beta_t\,(v_t - S_{t-1}k_t)\,k_t^\top$$

**Gated DeltaNet** (2024–25) adds a learned decay $\alpha_t$ (forget gate, like Mamba-2's) to the update — combining targeted overwriting with fast forgetting. It is the linear layer in Qwen3-Next and Qwen3.5; Kimi Delta Attention (KDA) refines it with per-channel gates. RWKV-7 "Goose" (2025) uses a generalised delta rule too.

What does a fixed state actually lose? Store n random key–value pairs, then query every key:

```python
import numpy as np

def recall_error(n_pairs, d=64, rule="linear", seed=0):
    """Store n (key, value) pairs, then query every key; return relative error of the values read back."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((n_pairs, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.standard_normal((n_pairs, d))
    if rule == "softmax":                                # attention over a KV cache that keeps every pair
        scores = 20.0 * K @ K.T                          # sharp softmax: each query matches its own key
        P = np.exp(scores - scores.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        return np.linalg.norm(P @ V - V) / np.linalg.norm(V)
    S = np.zeros((d, d))                                 # fixed d×d state, whatever n is
    for k, v in zip(K, V):
        S += np.outer(v, k) if rule == "linear" else np.outer(v - S @ k, k)   # delta rule corrects S·k → v
    return np.linalg.norm(K @ S.T - V) / np.linalg.norm(V)

for n in (16, 64, 256):
    print(f"pairs={n:3d}  linear={recall_error(n):.2f}  delta={recall_error(n, rule='delta'):.2f}"
          f"  softmax+KV cache={recall_error(n, rule='softmax'):.2f}")
# pairs= 16  linear=0.43  delta=0.31  softmax+KV cache=0.00
# pairs= 64  linear=1.00  delta=0.76  softmax+KV cache=0.00
# pairs=256  linear=1.97  delta=1.22  softmax+KV cache=0.00
```

The fixed state degrades as soon as the number of stored facts approaches its dimension (the delta rule more gracefully than plain linear attention); softmax attention over an ever-growing cache recalls everything — at a price that grows with every token. That single result explains the hybrid designs below.

<div class="lab" data-lab="ssm"><noscript>Interactive lab: compare a growing KV cache with a fixed recurrent state — memory per token and recall of stored facts as context grows. Enable JavaScript to use it.</noscript></div>

## RWKV

RWKV trains in parallel like a transformer and runs as an RNN at inference, with no KV cache:

| Version | Key idea |
|------|---------------|
| RWKV-4 (2023) | WKV operator: exponentially decaying linear attention |
| RWKV-5 "Eagle" / RWKV-6 "Finch" (2024) | Matrix-valued states; data-dependent decay |
| RWKV-7 "Goose" (2025) | Generalised delta rule with vector-valued gates |

## Hybrid Architectures: The 2025–26 Consensus

Mix many cheap linear-time layers with a few attention layers that provide exact retrieval:

| Model | Year | Linear layer | Linear : attention ratio | Notes |
|-------|---|------------|---------|---|
| Jamba / Jamba 1.5 (AI21) | 2024 | Mamba | 7 : 1 | + MoE; 256K context |
| Samba (Microsoft) | 2024 | Mamba + sliding-window attention | 1 : 1 | Long-context extrapolation |
| Nemotron-H, Nemotron Nano 2 (NVIDIA) | 2025 | Mamba-2 | ~10 : 1 (few attention layers) | Up to ~3× faster inference at long outputs |
| Granite 4.0 (IBM) | 2025 | Mamba-2 | ~9 : 1 | Enterprise open models; large memory savings |
| Falcon-H1 (TII) | 2025 | Mamba-2 (parallel with attention heads) | Parallel hybrid | 0.5B–34B |
| MiniMax-M1 | 2025 | Lightning (linear) attention | 7 : 1 | 1M context reasoning model |
| Qwen3-Next / Qwen3.5 | 2025–26 | Gated DeltaNet | 3 : 1 (gated attention) | Qwen's main line from 3.5 onward |
| Kimi Linear | 2025 | Kimi Delta Attention | 3 : 1 (MLA) | Up to 75% less KV cache |
| Kimi K3 | 2026 | Kimi Delta Attention | 69 : 24 (gated MLA) | 2.8T MoE, 1M context |

<div class="callout warn">
<span class="callout-title">Not a one-way street</span>
<p>MiniMax's M1 was a 7:1 lightning-attention hybrid, but its successor M2 (October 2025) returned to <em>full</em> attention; the team reported that the efficient variants lost accuracy on complex, multi-hop and agentic tasks at their scale. Hybrids save memory and time; whether they keep enough retrieval for long agent transcripts is still decided case by case — usually by keeping a quarter or more of the layers as attention.</p>
</div>

## Complexity Comparison

| Architecture | Training compute per layer | Decode cost per token | Generation memory |
|-------------|-------|----------|---------|
| Full attention (FlashAttention) | O(T²·d) | O(T·d) | KV cache: O(T) |
| Sliding window (w) | O(T·w·d) | O(w·d) | O(w) |
| Linear attention / DeltaNet | O(T·d²) (chunked) | O(d²) | Fixed d×d state per head |
| SSM (Mamba-2) | O(T·d·N) | O(d·N) | Fixed state |
| Hybrid (3:1 linear : attention) | ~¼ of the quadratic cost | ~¼ of attention's | ~¼ of the KV cache + fixed states |

## Diffusion Language Models

A different departure: instead of generating left to right, **diffusion LMs** start from a fully masked (or noised) sequence and denoise many positions in parallel over a few steps. LLaDA (2025) trained an 8B masked-diffusion model competitive with similar-size autoregressive models; Inception's Mercury and Google's Gemini Diffusion (2025) showed very high tokens-per-second for code and text. Diffusion LMs trade the strict left-to-right factorisation for parallel decoding and the ability to revise; whether they can match autoregressive models at the frontier remains open ([Chapter 38](./38_frontier.md)).

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A d×d state can hold about d facts; a 1M-token codebase holds far more. If three-quarters of a model's layers can only remember a summary, what exactly must the remaining attention layers do — and is that why hybrid ratios settled near 3:1 rather than 10:1 for models meant to work as long-running agents?</p>
</div>

## Key Takeaways

- Attention costs O(T²) compute and an O(T) cache; SSMs and linear attention keep a fixed-size state with O(1) cost per generated token.
- Mamba made SSMs input-selective; Mamba-2 recast them as structured linear attention on tensor cores.
- Linear attention is an associative memory; the delta rule (DeltaNet, Gated DeltaNet, KDA, RWKV-7) overwrites instead of accumulating, degrading more gracefully.
- Fixed state cannot recall arbitrary facts beyond its capacity; attention can — hence hybrids with roughly 3:1 to 10:1 linear-to-attention layers, now in Qwen3.5, Kimi K3, Nemotron and Granite.
- Diffusion LMs are a separate bet on parallel, non-autoregressive generation.

## Check Yourself

<details class="check">
<summary>A model has 48 layers, 12 of them full attention with 8 KV heads of 128 (BF16), the rest Gated DeltaNet. How large is its KV cache at 256K tokens compared with an all-attention version?</summary>
<p>Attention layers: 2 × 12 × 8 × 128 × 2 B = 48 KiB per token → 262,144 × 48 KiB = 12 GiB. All-attention would be 4× that, 48 GiB. The DeltaNet layers add fixed states that do not grow with context.</p>
</details>

<details class="check">
<summary>Why could S4 train as a convolution but Mamba could not?</summary>
<p>S4's A, B, C are the same at every time step (time-invariant), so the recurrence unrolls into a fixed convolution kernel. Mamba's B, C and Δ depend on each input token, so the "kernel" changes along the sequence and must be computed with a scan instead.</p>
</details>

<details class="check">
<summary>In the delta rule, what does the term (v − S k) represent?</summary>
<p>The error between the value you want stored for key k and what the memory currently returns for k. Writing that difference corrects the association instead of adding a second, conflicting copy — the online least-squares (Widrow–Hoff) update.</p>
</details>

<details class="check">
<summary>Why do almost all hybrids keep some full-attention layers rather than going 100% linear?</summary>
<p>Exact retrieval of arbitrary earlier tokens (copying, multi-hop lookup, long-context QA) needs a memory that grows with context; a fixed-size state can only hold about d independent associations per head. A few attention layers restore precise recall at a fraction of the total cost.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Efficiently Modeling Long Sequences with Structured State Spaces (S4) | Gu et al., 2021 | [arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396) | S4 |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao, 2023 | [arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752) | Selective SSMs |
| Transformers are SSMs (Mamba-2) | Dao & Gu, 2024 | [arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) | State space duality |
| Transformers are RNNs (Linear Attention) | Katharopoulos et al., 2020 | [arxiv.org/abs/2006.16236](https://arxiv.org/abs/2006.16236) | Linear attention |
| Parallelizing Linear Transformers with the Delta Rule | Yang et al., 2024 | [arxiv.org/abs/2406.06484](https://arxiv.org/abs/2406.06484) | DeltaNet |
| Gated Delta Networks | Yang, Kautz & Hatamizadeh, 2024 | [arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464) | Gated DeltaNet |
| RWKV-7 "Goose" | Peng et al., 2025 | [arxiv.org/abs/2503.14456](https://arxiv.org/abs/2503.14456) | Generalised delta rule RNN |
| Jamba | Lieber et al. (AI21), 2024 | [arxiv.org/abs/2403.19887](https://arxiv.org/abs/2403.19887) | Mamba–attention–MoE hybrid |
| Nemotron-H | NVIDIA, 2025 | [arxiv.org/abs/2504.03624](https://arxiv.org/abs/2504.03624) | Mamba-2 hybrids at scale |
| MiniMax-M1 | MiniMax, 2025 | [arxiv.org/abs/2506.13585](https://arxiv.org/abs/2506.13585) | Lightning-attention hybrid reasoning model |
| Kimi Linear | Moonshot AI, 2025 | [arxiv.org/abs/2510.26692](https://arxiv.org/abs/2510.26692) | Kimi Delta Attention |
| Large Language Diffusion Models (LLaDA) | Nie et al., 2025 | [arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992) | Masked diffusion LM |

---

**Next:** [Chapter 35 — Reasoning Models](./35_reasoning_models.md) — spending compute at inference time instead of only in training.

[← Back to Table of Contents](./README.md)
