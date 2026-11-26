---
title: "Chapter 16 — Inference & Sampling Strategies"
description: "Prefill and decode, latency metrics, greedy, temperature, top-k, top-p and min-p sampling, repetition penalties done correctly, beam search, parallel sampling, speculative decoding from rejection sampling to EAGLE-3 and MTP heads, thinking budgets, and grammar-constrained structured output."
related:
  - { id: inference, why: "Inference techniques, examples and recommendations." }
---

[← Back to Table of Contents](./README.md)

# Chapter 16 — Inference & Sampling Strategies

Training produces a distribution over next tokens. Inference has to turn it into text — fast, and in a way that suits the task. The same model can be precise or creative, terse or exhaustive, depending on how you decode it.

> **The one-sentence version:** generation is a loop of "forward pass → reshape the distribution → pick a token", and nearly every inference trick either reshapes that distribution (sampling), picks several tokens per forward pass (speculative decoding), or restricts what can be picked (constrained decoding).

## Autoregressive Generation

<div class="diagram">
<div class="diagram-title">Autoregressive Generation — Step by Step</div>
<div class="flow">
  <div class="flow-node accent wide">Prompt: "The capital of France is" <small>[B, T_prompt]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Prefill: process all prompt tokens in parallel <small>→ fills the KV cache; logits for the last position</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Decode step 1: sample from the last logits → " Paris" <small>append to the sequence</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Decode step 2: feed " Paris" only (past is cached) → "." <small>append</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Decode step N: → end-of-turn token <small>stop</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Prefill vs Decode</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Prefill (prompt processing)</div>
    <ul>
      <li>All prompt tokens at once — big matrix multiplies</li>
      <li>Compute-bound</li>
      <li>Writes the KV cache</li>
      <li>Sets <strong>time to first token (TTFT)</strong></li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Decode (generation)</div>
    <ul>
      <li>One token per sequence per step</li>
      <li>Memory-bandwidth-bound: every step reads all weights and the KV cache</li>
      <li>Appends to the KV cache</li>
      <li>Sets <strong>time per output token (TPOT)</strong> / inter-token latency</li>
    </ul>
  </div>
</div>
</div>

For a reasoning model that writes 10,000 thinking tokens, decode dominates everything — which is why Chapters 17, 18 and 31 are mostly about making decode cheaper.

## Greedy Decoding

<div class="img-caption">
  <img src="{{ '/assets/images/sampling_strategies.svg' | relative_url }}" alt="Four sampling strategy comparison: greedy always picks argmax; temperature scales logits; top-k keeps k most likely; top-p nucleus truncates by cumulative probability">
  <figcaption>Sampling strategies compared — from deterministic greedy to adaptive nucleus sampling</figcaption>
</div>

Always pick the most likely token:

```python
def greedy(logits):          # logits: [B, V]
    return logits.argmax(dim=-1)          # [B]
```

Deterministic and reproducible — the default for many evaluations and for short factual or code completions. In long generations greedy decoding tends to fall into **repetition loops**, and reasoning-model providers explicitly warn against it for thinking mode.

## Temperature

Divide the logits by τ before the softmax:

$$P(x_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

τ < 1 sharpens the distribution, τ > 1 flattens it, τ → 0 approaches greedy.

| Temperature (τ) | Effect | Typical use |
|:---:|---------|---------|
| 0 (greedy) | Deterministic | Evals, extraction, short code |
| 0.2–0.5 | Focused | Factual QA, tool calls |
| 0.6–0.8 | Balanced | Chat; recommended by many reasoning models (e.g. 0.6 for DeepSeek-R1 and Qwen3 thinking) |
| 0.9–1.2 | Diverse | Brainstorming, fiction, sampling many candidates for RL or best-of-N |

<div class="callout note">
<span class="callout-title">Start from the model card</span>
<p>Post-trained models are tuned with specific decoding settings, and most ship them in <code>generation_config.json</code>. Qwen3, for instance, recommends temperature 0.6, top-p 0.95, top-k 20 in thinking mode and 0.7 / 0.8 / 20 otherwise. Deviating is fine — but deviate on purpose.</p>
</div>

## Truncation: Top-k, Top-p, Min-p

Temperature alone still leaves a long tail of junk tokens with tiny but non-zero probability; over hundreds of steps one of them eventually gets sampled. Truncation removes the tail.

| Method | Keeps | Adapts to confidence? |
|---|---|---|
| **Top-k** | The k most likely tokens | No — k = 50 is too many when the model is sure, too few when it is not |
| **Top-p (nucleus)** | Smallest set whose total probability ≥ p | Yes |
| **Min-p** | Tokens with probability ≥ p_min × p(top token) | Yes, and robust at high temperature |

```python
import torch

def sample(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """logits: [B, V] → token ids [B]. One common order: temperature, top-k, min-p, top-p."""
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k > 0:                                         # keep k largest
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if min_p > 0:                                         # keep p >= min_p * p_max
        probs = probs.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, 0)
    if top_p < 1.0:                                       # nucleus
        sp, si = probs.sort(-1, descending=True)
        drop = sp.cumsum(-1) - sp > top_p                 # mass *before* this token already ≥ p
        sp = sp.masked_fill(drop, 0)
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1).squeeze(-1)
```

Shape the same distribution yourself — temperature, top-k, top-p, min-p — and watch which tokens survive:

<div class="lab" data-lab="sampling"><noscript>Interactive lab: reshape a next-token distribution with temperature, top-k, top-p and min-p, and sample from it. Enable JavaScript to use it.</noscript></div>

## Beam Search

Beam search keeps the B highest-scoring partial sequences and extends each by its best continuations:

```
beam width 3
step 1: "The" → ("The cat", −0.9), ("The dog", −1.2), ("The man", −1.6)
step 2: expand all, keep best 3 by total log-prob → ("The cat sat", …), ("The cat is", …), ("The dog ran", …)
```

It approximates the single most likely *sequence*, which suits tasks with one right answer (translation, speech recognition — Whisper uses it). For open-ended text the most likely sequence is bland and repetitive, and beam search is rarely used with chat LLMs.

## Repetition Penalties

| Penalty | What it does | Where |
|---------|---------|--------|
| **Repetition penalty** (CTRL-style) | For every token already in the context: divide its logit by θ if positive, multiply by θ if negative | Hugging Face `repetition_penalty` (θ ≈ 1.05–1.2) |
| **Frequency penalty** | Subtract α × (count of the token so far) | OpenAI-style APIs, vLLM |
| **Presence penalty** | Subtract α once if the token has appeared at all | OpenAI-style APIs, vLLM; Qwen suggests it against endless repetition |

Penalties are blunt: they also punish tokens that *should* repeat (variable names in code, a person's name, closing brackets). Prefer good truncation and the model's recommended settings; use penalties to fix a specific looping problem.

## Many Samples: Best-of-N and Self-Consistency

Sampling is also a way to buy accuracy with compute:

- **Self-consistency**: sample N reasoning paths at τ ≈ 0.7 and majority-vote the final answers.
- **Best-of-N**: sample N candidates and pick the one a reward model or verifier scores highest.
- **Thinking budgets**: reasoning models expose an effort or token-budget setting that controls how long they think before answering.

These are the inference-time half of test-time compute scaling, covered in [Chapter 35](./35_reasoning_models.md).

## Speculative Decoding

Decode is bandwidth-bound: at small batch sizes the GPU spends most of each step reading weights, not computing. Checking 5 tokens costs almost the same as generating 1. **Speculative decoding** exploits this: a cheap *draft* proposes K tokens, the target model scores all K (plus one) in a single forward pass, and a rejection-sampling rule decides how many to keep.

<div class="diagram">
<div class="diagram-title">Speculative Decoding</div>
<div class="flow">
  <div class="flow-node accent wide">Draft proposes K tokens <small>small model, MTP/Medusa/EAGLE heads, or n-gram lookup</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Target scores all K positions in one forward pass <small>same cost as ~one decode step when bandwidth-bound</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Accept token i with probability min(1, p(x)/q(x)) <small>on the first rejection, resample from max(0, p − q), normalised</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Emit accepted tokens + one target token <small>output distribution identical to the target model alone</small></div>
</div>
</div>

The acceptance rule makes it **lossless**: samples are distributed exactly as if the target had generated them alone. If each draft token is accepted with probability α, the expected number of tokens per target forward pass is

$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

— 2.9 tokens for α = 0.7, K = 5; 4.7 for α = 0.9. Speed-up is lower than that, because drafting is not free and verification gets more expensive as batch size grows and decode stops being bandwidth-bound.

<div class="lab" data-lab="specdec"><noscript>Interactive lab: expected tokens per pass and speed-up for speculative decoding as a function of acceptance rate, draft length and draft cost. Enable JavaScript to use it.</noscript></div>

| Drafter | How | Notes |
|---|---|---|
| **Separate small model** | e.g. Llama 3.2 1B drafting for Llama 3.1 70B | Needs the same tokenizer |
| **Medusa heads** (2024) | Extra heads predict tokens t+2, t+3 … from the last hidden state | Tree of candidates verified at once |
| **EAGLE-1/2/3** (2024–25) | A one-layer drafter that reads the target's hidden features; EAGLE-3 fuses features from several layers | Among the highest acceptance rates; built into vLLM, SGLang, TensorRT-LLM |
| **MTP heads** | DeepSeek-V3/V4's multi-token-prediction modules double as drafters | 85–90% acceptance for the second token |
| **Prompt / n-gram lookup** | Copy continuations from the prompt itself | Free; great for editing and RAG answers that quote context |

## Structured Output and Constrained Decoding

Tool calls and APIs need output that parses. **Constrained decoding** compiles a JSON schema, regular expression or context-free grammar into a state machine and, at every step, masks the logits of tokens that would make the output invalid. Since 2024–25 this runs inside the serving engines with fast grammar backends (XGrammar, llguidance, Outlines-core), adding little latency:

```python
from openai import OpenAI       # any OpenAI-compatible server: vLLM, SGLang, TGI, llama.cpp …

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
schema = {"type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}

resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Extract name and age: 'Ada, 36, writes compilers.'"}],
    response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": schema}},
)
print(resp.choices[0].message.content)    # always parses and matches the schema
```

Two caveats. Constraints guarantee *syntax*, not *truth* — the age can still be wrong. And forcing a reasoning model to emit JSON from its first token can suppress its thinking and hurt accuracy; let it reason first, then constrain only the final answer (most engines support this for reasoning models).

## Practical Configuration

```python
from transformers import GenerationConfig

precise  = GenerationConfig(do_sample=False, max_new_tokens=512)                       # evals, extraction
chat     = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=1024)
thinking = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                            max_new_tokens=32768)                                       # reasoning models: leave room to think
creative = GenerationConfig(do_sample=True, temperature=1.0, min_p=0.05, max_new_tokens=2048)
```

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Speculative decoding is lossless, yet production systems sometimes loosen the acceptance rule ("typical acceptance") to accept more draft tokens. What exactly do you give up, and how would you measure whether users notice?</p>
</div>

## Key Takeaways

- Prefill is compute-bound and sets TTFT; decode is bandwidth-bound and sets per-token latency — and dominates for long reasoning outputs.
- Temperature reshapes, top-k/top-p/min-p truncate; use the model card's recommended settings, especially for reasoning models.
- Repetition penalties are blunt instruments; beam search suits single-answer tasks, not chat.
- Speculative decoding verifies K drafted tokens in one target pass and is exactly distribution-preserving; expected tokens per pass = (1 − α^{K+1})/(1 − α).
- Grammar-constrained decoding inside the server guarantees parseable JSON or tool calls — syntax, not correctness.

## Check Yourself

<details class="check">
<summary>Logits are [4, 2, 1, 0]. What is the top token's probability at τ = 1 and at τ = 0.5?</summary>
<p>τ = 1: e⁴ / (e⁴ + e² + e¹ + e⁰) = 54.6 / 65.7 ≈ 0.83. τ = 0.5 doubles the logits to [8, 4, 2, 0]: e⁸ / (e⁸ + e⁴ + e² + 1) = 2981 / 3044 ≈ 0.98.</p>
</details>

<details class="check">
<summary>Top-p = 0.9 on probabilities [0.5, 0.3, 0.15, 0.05]. Which tokens survive?</summary>
<p>The first three: cumulative 0.5, 0.8, 0.95. The third is needed to reach 0.9; the fourth starts after 0.95 ≥ 0.9 and is dropped.</p>
</details>

<details class="check">
<summary>Why is speculative decoding's output distribution identical to the target model's?</summary>
<p>Each draft token x is accepted with probability min(1, p(x)/q(x)); on rejection, the replacement is drawn from the normalised residual max(0, p − q). Summing the two paths gives exactly p(x) for every token — standard rejection-sampling algebra.</p>
</details>

<details class="check">
<summary>Speculative decoding gave 2.5× at batch size 1 but almost nothing at batch size 128. Why?</summary>
<p>At batch 1, decode is bandwidth-bound, so verifying K extra tokens is nearly free. At batch 128, each step already does enough arithmetic per weight byte to be closer to compute-bound; verifying K+1 tokens per sequence multiplies the compute, eating the gain.</p>
</details>

<details class="check">
<summary>A schema-constrained extraction returns valid JSON with a wrong field. Does constrained decoding guarantee anything about correctness?</summary>
<p>No. It guarantees that every emitted token keeps the output inside the grammar, so the result parses and matches the schema. The values are still the model's predictions, and forcing structure too early can even reduce accuracy for reasoning models.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| The Curious Case of Neural Text Degeneration | Holtzman et al., 2019 | [arxiv.org/abs/1904.09751](https://arxiv.org/abs/1904.09751) | Nucleus (top-p) sampling |
| CTRL | Keskar et al., 2019 | [arxiv.org/abs/1909.05858](https://arxiv.org/abs/1909.05858) | Repetition penalty |
| Turning Up the Heat: Min-p Sampling | Nguyen et al., 2024 | [arxiv.org/abs/2407.01082](https://arxiv.org/abs/2407.01082) | Min-p |
| Self-Consistency Improves Chain of Thought Reasoning | Wang et al., 2022 | [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171) | Majority voting over samples |
| Fast Inference from Transformers via Speculative Decoding | Leviathan et al., 2022 | [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192) | Speculative decoding, acceptance rule |
| Accelerating LLM Decoding with Speculative Sampling | Chen et al. (DeepMind), 2023 | [arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318) | Speculative sampling |
| Medusa | Cai et al., 2024 | [arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774) | Multiple decoding heads |
| EAGLE-3 | Li et al., 2025 | [arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840) | Feature-fusion drafting |
| XGrammar | Dong et al., 2024 | [arxiv.org/abs/2411.15100](https://arxiv.org/abs/2411.15100) | Fast grammar-constrained decoding |
| Qwen3 model card | Qwen Team, 2025 | [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Recommended sampling settings |

---

**Next:** [Chapter 17 — KV-Cache: Mechanics & Memory](./17_kv_cache_mechanics.md) — the data structure that makes decode possible, and the memory bill it runs up.

[← Back to Table of Contents](./README.md)
