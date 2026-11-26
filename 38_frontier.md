---
title: "Chapter 38 — The Frontier: Open Problems & What's Next"
description: "Where language modelling stands in late 2026 and what is unsolved: long context that works, continual learning, efficiency, multimodal and world models, diffusion and other non-autoregressive generators, interpretability, alignment and safety, evaluation, the open-weight ecosystem, and the economics of scale."
related:
  - { id: models, why: "The model generations so far." }
  - { id: agents, why: "Agents, the fastest-moving frontier." }
---

[← Back to Table of Contents](./README.md)

# Chapter 38 — The Frontier: Open Problems & What's Next

Two years ago this chapter listed reasoning and agents as frontier problems. Both became products. This version tries to separate what changed between 2024 and late 2026 from what is still genuinely open — and to give you the tools to read the next two years' news critically.

> **The one-sentence version:** capability now comes from three compounding scaling axes — pre-training, RL and test-time compute — while the open problems have shifted from "can models reason?" to "can they keep learning, use everything they are given, stay trustworthy, and be afforded?"

## What Changed, 2024 → 2026

| Then (2024) | Now (late 2026) |
|---|---|
| Chat models answer single prompts | Agents work for hours in tools, terminals and browsers |
| Reasoning via prompting | RL-trained thinking with adjustable budgets in every frontier model |
| Dense models; MoE at a few labs | MoE everywhere at the top; 3–5% of parameters active |
| Attention + GQA | MLA, learned sparse attention, linear-attention hybrids |
| BF16 training, INT4 community quants | FP8 training standard, FP4 serving on Blackwell, QAT checkpoints |
| Open models a year behind | Open-weight models within months of the frontier on many benchmarks |
| MMLU-style benchmarks | Agentic, expert and continuously refreshed evaluations |

## Long Context That Works

Windows of 1M tokens are common (Gemini, Llama 4, Qwen2.5-1M, DeepSeek-V4, Kimi K3), but **using** them remains hard:

- **Effective vs advertised**: multi-hop reasoning and retrieval of several facts degrade far before the window is full ([Chapter 12](./12_mid_training.md)).
- **Cost**: every token of context is paid for on every call; sparse and hybrid attention plus prefix caching are making it cheaper, not free ([Chapters 5](./05_attention_gqa_mqa_mla.md), [18](./18_kv_cache_optimization.md), [34](./34_ssms_and_alternatives.md)).
- **Agents generate their own context**: the hardest long-context workload is an agent's transcript, which grows with every tool call — hence context engineering ([Chapter 37](./37_agents_and_tool_use.md)).

Open question: is the answer bigger windows, better memory architectures, or models that learn what to keep?

## Continual Learning

A deployed model's weights are frozen at its training cutoff. It cannot learn from yesterday's conversation, today's codebase or a correction you gave it an hour ago — except through the context window. Fine-tuning causes **catastrophic forgetting**; retrieval and memory files are workarounds, not learning. Research directions in 2025–26: test-time training (updating fast weights on the current context), memory layers, sparse "memory" parameters updated per user, and frequent small updates via RL on fresh data. No frontier model learns continually in deployment yet.

## Efficiency

The cost of a token has fallen by orders of magnitude since 2023 for a given capability level, through:

<div class="diagram">
<div class="diagram-title">Where the Efficiency Gains Come From</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card green">
    <div class="card-title">Architecture</div>
    <div class="card-desc">MoE (compute per token), MLA and hybrids (KV cache), multi-token prediction (speculative drafts).</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Numerics</div>
    <div class="card-desc">FP8 training; FP4 inference; QAT; KV quantization.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Data and distillation</div>
    <div class="card-desc">Small models trained on 10–40T filtered and synthetic tokens and distilled from larger ones.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Systems</div>
    <div class="card-desc">Paged and prefix-cached KV, disaggregated serving, wide expert parallelism, speculative decoding.</div>
  </div>
</div>
</div>

But reasoning and agents multiply tokens per task by 10–1,000×, so total compute demand keeps rising — a Jevons paradox for intelligence. Power, not chips, became the binding constraint for the largest 2026 build-outs.

## Multimodality and World Models

Frontier models read images, video and audio natively and increasingly generate them ([Chapter 8](./08_multimodal_models.md)). Video generators (Veo, Sora, and open models such as Wan) produce physically plausible clips; interactive "world models" (Genie 3, 2025) generate explorable environments in real time; vision-language-action models drive robots. The open question is whether next-token (or next-frame) prediction on enough video and interaction data yields models that understand physics and causality well enough to plan in the real world — or whether embodiment and new objectives are needed.

## Beyond Left-to-Right Generation

Autoregressive decoding is sequential by construction. Alternatives being explored:

- **Diffusion language models** — denoise all positions in parallel over a few steps (LLaDA, Mercury, Gemini Diffusion; [Chapter 34](./34_ssms_and_alternatives.md)): very fast, can revise earlier tokens, not yet at the frontier on reasoning.
- **Multi-token prediction** heads — already used for training signal and speculative decoding.
- **Latent reasoning** — thinking in continuous hidden states rather than text tokens (e.g., Coconut, 2024): potentially more efficient, but at the cost of monitorability.

## Interpretability

Mechanistic interpretability moved from toy models to production models: sparse autoencoders and transcoders extract millions of human-interpretable features from frontier LLMs, and **attribution graphs** ("circuit tracing", Anthropic 2025) trace how a specific answer was computed — revealing, for instance, planning of rhyme words ahead of time and multi-step reasoning inside a single forward pass. Open problems: completeness (how much of the computation is explained?), scalability to every behaviour, and using interpretability to *verify* properties like honesty rather than only to illustrate them.

## Alignment and Safety

<div class="diagram">
<div class="diagram-title">Safety Research Areas (2026)</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">Oversight</div>
    <div class="card-desc">Chain-of-thought monitoring, AI-assisted review of agent transcripts, scalable oversight for tasks humans cannot easily check.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Robustness</div>
    <div class="card-desc">Jailbreaks, prompt injection against agents, reward hacking, evaluation awareness, sycophancy.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Governance</div>
    <div class="card-desc">Frontier safety frameworks with capability thresholds; EU AI Act obligations for general-purpose models (from 2025); third-party pre-deployment testing.</div>
  </div>
</div>
</div>

Live debates: open versus closed weights as capability grows; how to evaluate dangerous capabilities (biology, cyber, autonomy) credibly; whether alignment techniques that work today — RLHF, constitutions, deliberative alignment — keep working on systems that act for days with little supervision.

## Evaluation

Benchmarks now saturate within a year or two of release ([Chapter 30](./30_evaluation_and_benchmarks.md)). The frontier of evaluation is expensive: expert-written questions, live agent environments, human professionals grading real work products, long-horizon tasks measured in hours. A durable, cheap, uncontaminated measure of general capability does not exist.

## The Open-Weight Ecosystem

Open weights narrowed the gap to the closed frontier dramatically in 2025, led by DeepSeek, Qwen, Moonshot (Kimi), Zhipu (GLM) and MiniMax, with Google (Gemma), OpenAI (gpt-oss), Mistral, NVIDIA and Ai2 (fully open OLMo) contributing important models — while Meta moved its flagship to a closed model. The flywheel still turns:

<div class="diagram">
<div class="diagram-title">The Open-Model Flywheel</div>
<div class="cycle">
  <div class="cycle-step green">Labs release open weights and technical reports</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step accent">Community fine-tunes, quantizes, serves, evaluates</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step purple">Research finds techniques (LoRA, DPO, GRPO variants, kernels, quantization)</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step orange">Labs adopt them → better open models → repeat</div>
</div>
</div>

Open questions: whether frontier-scale open releases continue as capabilities grow; how licences, export controls and safety obligations shape them; and whether fully open (data + code) models can keep pace.

## What Remains Unsolved

| Problem | Status (late 2026) |
|---------|--------|
| Reliable long-horizon autonomy | Hours of work with oversight; error compounding and recovery remain the bottleneck |
| Continual learning | Not solved in deployed systems; context and retrieval are stand-ins |
| Using long context | Advertised windows ≫ effective windows for multi-step use |
| Hallucination and calibration | Reduced, not eliminated; models still state falsehoods fluently |
| Novel scientific discovery | Strong assistance in maths, code and biology; genuinely new results still rare and contested |
| Faithful, verifiable reasoning | CoT is useful but incomplete; interpretability cannot yet certify behaviour |
| Efficiency at the frontier | Cost per token falls fast; cost per *task* rises with thinking and agents |
| Evaluation | No durable general measure; benchmarks saturate fast |
| Alignment for autonomous systems | Active research; oversight scales poorly with task length |

## Predictions, Revisited

The previous edition of this chapter (early 2025) made five predictions. A scorecard:

1. **"Models will get smaller and better"** — ✅ 8B-class models of 2026 clearly beat 70B models of 2024 on most benchmarks.
2. **"Hybrid architectures will win"** — ✅ mostly: MoE is universal and linear-attention hybrids reached Qwen3.5 and Kimi K3 — though some labs returned to full attention for agentic quality.
3. **"Reasoning will be the key differentiator"** — ✅ every frontier model is a reasoning model.
4. **"Inference will matter more than training"** — ✅ test-time compute and serving dominate cost.
5. **"Agents will become the interface"** — ✅ underway: coding agents are the clearest case.

Easy calls in hindsight. Harder ones for 2027–28: whether continual learning arrives; whether diffusion or latent reasoning displaces token-by-token thinking; and whether RL scaling keeps paying off as smoothly as pre-training did.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Every chapter of this guide describes a technique that was a research paper three years before it was standard. Pick one open problem from the table above. What would the paper that solves it have to show — and which chapter of this guide would it rewrite?</p>
</div>

## Key Takeaways

- 2024 → 2026: reasoning and agents became products; MoE, efficient attention and low precision became standard; open weights closed much of the gap.
- Open problems now centre on long-context use, continual learning, reliability over long horizons, faithful reasoning, evaluation and alignment of autonomous systems.
- Efficiency per token keeps improving while tokens per task explode; power is the new constraint.
- Read new results through the lenses of this guide: what does it cost (Ch 10, 31), what does it measure (Ch 30), and what bottleneck does it move (Ch 17, 24)?

## Check Yourself

<details class="check">
<summary>Why does falling cost per token not imply falling total compute demand?</summary>
<p>Cheaper tokens make new uses viable — reasoning models think for thousands of tokens, agents run for hours, more people and products use AI — so tokens consumed grow faster than cost per token falls (a Jevons effect).</p>
</details>

<details class="check">
<summary>Why isn't retrieval or a memory file "continual learning"?</summary>
<p>They change what is in the context, not the model's weights; the model does not get better at the underlying skill, and the information must be re-read (and paid for) every time. Continual learning would update the model itself without forgetting earlier abilities.</p>
</details>

<details class="check">
<summary>What is the safety trade-off of reasoning in latent space instead of text?</summary>
<p>It could be more efficient and expressive, but it removes the human-readable chain of thought that currently lets monitors detect reward hacking, deception or unsafe plans.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| On the Biology of a Large Language Model (attribution graphs) | Lindsey et al. (Anthropic), 2025 | [transformer-circuits.pub/2025/attribution-graphs/biology.html](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Circuit tracing in a production model |
| Scaling Monosemanticity | Templeton et al. (Anthropic), 2024 | [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | Sparse autoencoders at scale |
| Training Large Language Models to Reason in a Continuous Latent Space (Coconut) | Hao et al., 2024 | [arxiv.org/abs/2412.06769](https://arxiv.org/abs/2412.06769) | Latent reasoning |
| Genie 3 | Google DeepMind, 2025 | [deepmind.google](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) | Interactive world models |
| Chain of Thought Monitorability | Korbak et al., 2025 | [arxiv.org/abs/2507.11473](https://arxiv.org/abs/2507.11473) | Oversight via CoT |
| Measuring AI Ability to Complete Long Tasks | METR, 2025 | [arxiv.org/abs/2503.14499](https://arxiv.org/abs/2503.14499) | Autonomy trend |
| International AI Safety Report | Bengio et al., 2025 | [gov.uk](https://www.gov.uk/government/publications/international-ai-safety-report-2025) | State of safety evidence |
| EU AI Act — general-purpose AI obligations | European Commission, 2025 | [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) | Governance |

---

**You've reached the end of the guide.** From the arithmetic of attention to the economics of serving, from tokenizers to agents — you now have a map of the whole field, and the tools to update it yourself. The field moves fast; read papers, run experiments, build things, and check every number.

**Appendices:** [A — History](./appendix_a_history.md) · [B — Tokenization](./appendix_b_tokenization.md) · [C — Model Config Reference](./appendix_c_model_config.md) · Interactive: [⚗ Labs](./labs.md)

[← Back to Table of Contents](./README.md)
