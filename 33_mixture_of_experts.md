---
title: "Chapter 33 — Mixture of Experts (MoE)"
description: "Sparse models from Switch Transformer to Kimi K3: routers (softmax and sigmoid), top-k, shared and fine-grained experts, load balancing with auxiliary losses and DeepSeek's bias trick, capacity and dropless execution, MoE memory and bandwidth in serving, expert parallelism, the 2024–26 model table, and fine-tuning MoEs."
related:
  - { id: models, why: "MoE across model generations." }
---

[← Back to Table of Contents](./README.md)

# Chapter 33 — Mixture of Experts (MoE)

Every open frontier model of 2025–26 is a mixture of experts: DeepSeek-V3/V4, Qwen3/3.5, Kimi K2/K3, GLM-4.5/5, gpt-oss, Llama 4, Mistral Large 3. MoE decouples how much a model *knows* (total parameters) from how much it *computes* per token (active parameters).

> **The one-sentence version:** replace each feed-forward layer with many smaller "expert" FFNs and a router that sends each token to a few of them — so parameters grow with the number of experts while FLOPs per token grow only with the number chosen.

## Dense vs Sparse

<div class="diagram">
<div class="diagram-title">Dense vs Sparse</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Dense (Llama 3.1 70B)</div>
    <ul>
      <li>Every parameter used for every token</li>
      <li>70.6B parameters → 70.6B active</li>
      <li>Training ≈ 6 × 70.6B ≈ 424 GFLOP per token</li>
      <li>141 GB of BF16 weights to hold</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Sparse MoE (DeepSeek-V3)</div>
    <ul>
      <li>1 shared + 8 of 256 routed experts per token</li>
      <li>671B parameters → 37B active</li>
      <li>Training ≈ 6 × 37B ≈ 222 GFLOP per token</li>
      <li>671 GB of FP8 weights to hold</li>
    </ul>
  </div>
</div>
</div>

MoE gives the quality of a much larger model at the per-token compute of a smaller one — **if you can afford the memory** to hold all experts.

## MoE Architecture

<div class="img-caption">
  <img src="{{ '/assets/images/moe_routing.svg' | relative_url }}" alt="MoE sparse routing diagram showing token embedding passing through router to activate top-2 of 8 experts, with weighted output combination">
  <figcaption>MoE routing: each token activates only its top-k experts; outputs are combined by the router's weights</figcaption>
</div>

<div class="diagram">
<div class="diagram-title">MoE Layer</div>
<div class="flow">
  <div class="flow-step accent">Input hidden states: [B, T, d]</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step orange">Router: linear d → E scores per token (softmax or sigmoid)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">Top-k: choose k experts per token (k = 1–16 in current models)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">Dispatch: group tokens by expert; run each expert's FFN on its tokens</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">Combine: weighted sum of the k outputs (+ shared expert) → [B, T, d]</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row"><span class="ts-label">Input</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B·T</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">Router scores</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B·T</span><span class="ts-sep">,</span><span class="ts-dim generic">E</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">Top-k weights / ids</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B·T</span><span class="ts-sep">,</span><span class="ts-dim generic">k</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">Expert e input</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim seq">n_e</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div><span class="ts-note">n_e = tokens routed to expert e; Σ n_e = B·T·k</span></div>
  <div class="tensor-flow-row"><span class="ts-label">Combined output</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B·T</span><span class="ts-sep">,</span><span class="ts-dim feature">d</span><span class="ts-bracket">]</span></div></div>
</div>
</div>

### The Router

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """Mixtral-style: pick top-k logits, softmax over just those k."""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):                              # x: [N, d]
        logits = self.gate(x)                          # [N, E]
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)
        return F.softmax(top_logits, dim=-1), top_idx  # [N, k], [N, k]
```

Variants: Switch and Qwen apply softmax over all experts *then* take the top-k; DeepSeek-V3 uses **sigmoid** affinities per expert and normalises over the selected ones, so experts do not compete in a single softmax; Llama 4 Maverick routes each token to just one routed expert plus a shared one.

## Load Balancing

Left alone, routers collapse: a few experts get most tokens, get trained more, become even more attractive, and the rest die. Imbalance also wrecks efficiency, because with expert parallelism the busiest GPU sets the pace.

**Auxiliary loss** (Switch Transformer):

$$\mathcal{L}_{\text{balance}} = \alpha \cdot E \sum_{i=1}^{E} f_i \, P_i$$

where $f_i$ is the fraction of tokens dispatched to expert i and $P_i$ the mean router probability for i; α ≈ 0.001–0.01. It is minimised when both are uniform — but a strong α fights the language-modelling objective.

**Auxiliary-loss-free balancing** (DeepSeek-V3): add a per-expert bias $b_i$ to the routing scores *only for choosing* the top-k (not for weighting outputs). After each step, lower $b_i$ for overloaded experts and raise it for underloaded ones. Balance is maintained without a gradient that distorts the model; a tiny sequence-level auxiliary loss remains as a safeguard. Qwen instead computes the balance loss over the **global batch** rather than per micro-batch, which lets experts specialise by domain.

Watch routing collapse and the two fixes on a stream of synthetic tokens:

<div class="lab" data-lab="moe"><noscript>Interactive lab: route tokens to experts with top-k, watch load imbalance, and compare an auxiliary loss with bias-based balancing and capacity limits. Enable JavaScript to use it.</noscript></div>

### Capacity and Dropping

Early systems (GShard, Switch) gave each expert a fixed **capacity** — capacity factor × (tokens × k / E) — and *dropped* overflow tokens (they skipped the layer via the residual). Modern implementations are **dropless**: MegaBlocks-style block-sparse kernels and **grouped GEMMs** process variable token counts per expert efficiently, so no token is discarded.

## Shared and Fine-Grained Experts

DeepSeekMoE (2024) introduced two ideas that most 2025–26 models adopted:

<div class="diagram">
<div class="diagram-title">DeepSeek-V3 MoE Layer</div>
<div class="layer-stack">
  <div class="layer orange">1 shared expert — always active; learns common knowledge so routed experts need not duplicate it</div>
  <div class="layer accent">Router selects 8 of 256 routed experts per token (sigmoid scores, bias-balanced)</div>
  <div class="layer green">Fine-grained: each expert is small (FFN width 2,048 vs a dense-equivalent 18,432), so combinations are far more varied</div>
  <div class="layer purple">Output = shared(x) + Σ g_k · expert_k(x) over the top 8</div>
</div>
</div>

With 256 experts choosing 8, there are about 4 × 10¹⁴ possible expert combinations per layer, versus 28 for 8-choose-2. Fine-grained routing lets the model compose specialists flexibly. Interpretability studies (from Mixtral onward) find experts specialise more by syntax and token type than by human-recognisable topic.

## A Simple MoE Layer

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        super().__init__()
        self.router = TopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList(SwiGLU_FFN(d_model, d_ff) for _ in range(n_experts))  # Chapter 3

    def forward(self, x):                                   # [B, T, d]
        B, T, d = x.shape
        flat = x.reshape(-1, d)                             # [N, d]
        weights, idx = self.router(flat)                    # [N, k], [N, k]
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_ids, slot = (idx == e).nonzero(as_tuple=True)   # which tokens chose e, and in which slot
            if token_ids.numel():
                out.index_add_(0, token_ids, weights[token_ids, slot, None] * expert(flat[token_ids]))
        return out.reshape(B, T, d)
```

Production kernels replace the Python loop with a sort-by-expert + grouped GEMM (MegaBlocks, PyTorch's grouped matmul, vLLM/SGLang fused MoE kernels), and with expert parallelism the "sort" becomes an all-to-all across GPUs ([Chapter 26](./26_distributed_training.md)).

## Notable MoE Models

| Model | Year | Total / active | Experts (routed + shared) | Top-k | Notes |
|-------|---|-------------|---------|-------|-------------------|
| Switch Transformer | 2021 | 1.6T (sparse T5) | 2,048 | 1 | Top-1 routing, capacity factor |
| Mixtral 8×7B / 8×22B | 2023–24 | 46.7B / 12.9B; 141B / 39B | 8 | 2 | Popularised open MoE |
| Grok-1 | 2024 | 314B / ~86B | 8 | 2 | xAI open weights |
| DBRX | 2024 | 132B / 36B | 16 | 4 | Fine-grained |
| Qwen1.5-MoE-A2.7B | 2024 | 14.3B / 2.7B | 60 + 4 shared | 4 | Upcycled from a dense model |
| DeepSeek-V2 | 2024 | 236B / 21B | 160 + 2 | 6 | DeepSeekMoE + MLA |
| DeepSeek-V3 | 2024 | 671B / 37B | 256 + 1 | 8 | Bias balancing, node-limited routing, MTP |
| Llama 4 Maverick | 2025 | 400B / 17B | 128 + 1 | 1 | Alternating dense and MoE layers |
| Qwen3-235B-A22B | 2025 | 235B / 22B | 128 | 8 | No shared expert; global-batch balancing |
| Kimi K2 | 2025 | 1T / 32B | 384 + 1 | 8 | MuonClip; MLA |
| gpt-oss-120b | 2025 | 117B / 5.1B | 128 | 4 | MXFP4 experts |
| GLM-5 | 2026 | 744B / 40B | 256 + 1 | 8 | DeepSeek Sparse Attention |
| Qwen3.5-397B-A17B | 2026 | 397B / 17B | 512 | — | Hybrid Gated DeltaNet attention |
| DeepSeek-V4-Pro | 2026 | 1.6T / 49B | DeepSeekMoE | — | Compressed sparse attention, 1M context |
| Kimi K3 | 2026 | 2.8T / ~104B | 896 | 16 | KDA + gated MLA hybrid |

The trend is unmistakable: more experts, smaller experts, lower activation ratios (DeepSeek-V3 5.5%; Qwen3.5 4.3%; Kimi K3 3.7%).

## Memory, Bandwidth and the KV Cache

- **KV cache** depends only on the attention layers, which MoE leaves dense. A MoE model's cache is that of its attention design (MLA in DeepSeek-V3 → 69 KiB per token), unrelated to expert count ([Chapter 17](./17_kv_cache_mechanics.md)).
- **Weight memory** is the *total* parameter count — every expert must be resident somewhere.
- **Decode bandwidth** is subtler. At batch 1, a decode step reads only the active experts — bytes per token ≈ active parameters, so an MoE decodes like a small dense model *if it fits*. At large batch, the tokens in a step collectively touch almost every expert, so each step reads nearly all weights — but amortised over many tokens. MoE serving therefore wants large batches and wide expert parallelism ([Chapter 31](./31_serving_and_deployment.md)).

## Challenges

| Challenge | Impact | Mitigation |
|-----------|--------|-----------|
| Memory for all experts | 671 GB for DeepSeek-V3 in FP8 | FP8/FP4 experts, multi-GPU EP, CPU/SSD offload for local use |
| Load imbalance | Stragglers in EP; dead experts | Bias balancing, global-batch balancing, replicate hot experts at serve time |
| All-to-all communication | Inter-node bandwidth bound | Node-limited routing, DeepEP-style kernels, overlap with compute |
| Router instability | Loss spikes, routing churn | Router z-loss, FP32 router, sigmoid scores |
| RL on MoE | Token-level importance ratios noisy because routing changes | Sequence-level ratios (GSPO), routing replay |
| Fine-tuning | Router drift; which experts to adapt | Freeze the router; LoRA on all experts; expert-specialised fine-tuning (train only experts a task activates) |

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Activation ratios keep falling — Kimi K3 activates under 4% of its weights per token. If the limit is set by memory and communication rather than compute, what is the ideal ratio, and does the answer change between a datacentre serving millions of users and a workstation serving one?</p>
</div>

## Key Takeaways

- MoE replaces FFNs with E experts and a router; parameters scale with E, compute with k.
- Routers use softmax or sigmoid scores; balancing moved from auxiliary losses to bias adjustment (DeepSeek) and global-batch losses (Qwen).
- Shared + fine-grained experts are the 2024–26 standard; execution is dropless with grouped GEMMs.
- KV cache is set by attention, not experts; weight memory by total parameters; decode bandwidth by active parameters at small batch.
- Every open frontier model is MoE, with activation ratios of 3–5%.

## Check Yourself

<details class="check">
<summary>A layer has 64 experts, top-4 routing and 1 shared expert, each expert of FFN width 1,024 with d = 2,048 (SwiGLU). How many parameters does the layer hold, and how many are active per token?</summary>
<p>Each expert: 3 × 2,048 × 1,024 = 6.29M. Total: 65 × 6.29M ≈ 409M (plus a tiny 2,048 × 64 router). Active: 5 × 6.29M ≈ 31.5M — about 7.7%.</p>
</details>

<details class="check">
<summary>Why does DeepSeek's balancing bias affect expert selection but not the output weights?</summary>
<p>If the bias entered the combination weights, it would change what the model computes (a learned function would be distorted to fix a load problem). Using it only for top-k selection steers traffic while the output still uses the router's true affinities.</p>
</details>

<details class="check">
<summary>Why can a 30B-A3B MoE decode a single stream faster than a dense 14B model on the same GPU, if both fit?</summary>
<p>At batch 1 decode reads roughly the active parameters per token: ~3B for the MoE versus 14B for the dense model, so ~4–5× fewer bytes per token (attention and router overhead aside).</p>
</details>

<details class="check">
<summary>What does "dropless" mean, and why did capacity factors exist?</summary>
<p>Dropless MoE processes every token routed to every expert, however uneven the counts. Capacity factors existed because early kernels needed fixed-size per-expert batches; overflowing tokens were dropped (passed through the residual), losing information. Block-sparse kernels and grouped GEMMs removed the need.</p>
</details>

<details class="check">
<summary>An MoE's KV cache per token is smaller than a dense model with the same active parameter count. Is that because it is an MoE?</summary>
<p>No. KV size depends only on the attention design (layers, KV heads, head dim, MLA, windows). If an MoE has a smaller cache, it is because of choices like MLA or fewer KV heads, not because of its experts.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Outrageously Large Neural Networks (Sparsely-Gated MoE) | Shazeer et al., 2017 | [arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538) | Gated MoE |
| GShard | Lepikhin et al., 2020 | [arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668) | Capacity, expert parallelism |
| Switch Transformers | Fedus et al., 2021 | [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961) | Top-1 routing, balance loss |
| ST-MoE: Designing Stable and Transferable Sparse Expert Models | Zoph et al., 2022 | [arxiv.org/abs/2202.08906](https://arxiv.org/abs/2202.08906) | Router z-loss |
| MegaBlocks | Gale et al., 2022 | [arxiv.org/abs/2211.15841](https://arxiv.org/abs/2211.15841) | Dropless MoE kernels |
| Mixtral of Experts | Jiang et al., 2024 | [arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088) | Open sparse MoE, expert analysis |
| DeepSeekMoE | Dai et al., 2024 | [arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066) | Fine-grained + shared experts |
| Auxiliary-Loss-Free Load Balancing for MoE | Wang et al. (DeepSeek), 2024 | [arxiv.org/abs/2408.15664](https://arxiv.org/abs/2408.15664) | Bias-based balancing |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | 256-expert design |
| Qwen3 Technical Report | Qwen Team, 2025 | [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) | Global-batch balancing, no shared expert |
| Kimi K2 | Moonshot AI, 2025 | [arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534) | 384-expert, 1T MoE |

---

**Next:** [Chapter 34 — SSMs & Beyond Transformers](./34_ssms_and_alternatives.md) — replacing attention itself with fixed-size recurrent state.

[← Back to Table of Contents](./README.md)
