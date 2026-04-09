---
title: "Chapter 34 — The Frontier — Open Problems & What's Next"
---

[← Back to Table of Contents](./README.md)

# Chapter 34 — The Frontier — Open Problems & What's Next

> *"The most exciting phrase in science is not 'Eureka!' but 'That's funny...'"* — Isaac Asimov

This final chapter surveys the frontiers of language modelling — where rapid progress is being made, and what remains stubbornly unsolved.

## Long Context

The push toward million-token context windows:

| Model | Max Context | Method |
|-------|-----------|--------|
| GPT-4 Turbo | 128K | Proprietary |
| Claude 3.5 | 200K | Proprietary |
| Gemini 1.5 Pro | 2M | Ring Attention + RoPE scaling |
| LLaMA-3.1 | 128K | Progressive RoPE ABF training |
| Yarn / LongRoPE | Extensible | NTK-aware interpolation |
| Mamba/SSMs | Theoretically ∞ | Fixed-size recurrent state |

Open challenges:
- **Retrieval across long context**: models struggle to use information buried in the middle ("lost in the middle" problem)
- **Cost**: attention is O(T²) — 1M tokens means 10¹² operations per layer
- **Evaluation**: few benchmarks test genuine long-context reasoning (RULER, BABILong are early attempts)

## Efficiency Frontiers

Making models smaller, faster, and cheaper without sacrificing quality:

<div class="diagram">
<div class="diagram-title">Efficiency Research Directions</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card green">
    <div class="card-title">Architecture Efficiency</div>
    <div class="card-desc">MoE (sparse activation), SSMs (linear complexity), hybrid models, speculative decoding, early exit.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Compression</div>
    <div class="card-desc">Quantization to 2-4 bits, pruning (structured + unstructured), distillation, neural architecture search.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Training Efficiency</div>
    <div class="card-desc">Better data curation (quality > quantity), curriculum learning, parameter-efficient fine-tuning, continual pre-training.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Inference Optimization</div>
    <div class="card-desc">KV-cache compression, prefix caching, request batching, hardware-aware kernel design.</div>
  </div>
</div>
</div>

The trend: **smaller models trained on more, better data**. LLaMA-3-8B matches LLaMA-2-70B. Phi-3-mini (3.8B) matches LLaMA-2-13B.

## Multimodal

The convergence toward unified models that see, hear, read, and generate across modalities:

<div class="diagram">
<div class="diagram-title">Multimodal Evolution</div>
<div class="timeline">
  <div class="timeline-item green">
    <div class="timeline-date">Text only</div>
    <div class="timeline-desc">GPT-3, LLaMA — language in, language out</div>
  </div>
  <div class="timeline-item accent">
    <div class="timeline-date">Vision + Language</div>
    <div class="timeline-desc">GPT-4V, LLaVA, Gemini — understand images + text</div>
  </div>
  <div class="timeline-item purple">
    <div class="timeline-date">Audio + Vision + Language</div>
    <div class="timeline-desc">GPT-4o, Gemini 2 — native speech, image understanding</div>
  </div>
  <div class="timeline-item orange">
    <div class="timeline-date">Generation</div>
    <div class="timeline-desc">DALL-E 3, Sora, Veo — generate images, video from text</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">Unified</div>
    <div class="timeline-desc">Any modality in → any modality out (emerging)</div>
  </div>
</div>
</div>

Open frontiers: video understanding at scale, spatial reasoning, audio-visual grounding, embodied AI.

## World Models

Can language models learn a model of the **physical world** — not just statistical patterns in text?

- **Video prediction**: Sora, Veo generate physically plausible video
- **Simulation**: models that can predict what happens next in a 3D environment
- **Embodied AI**: robots using LLMs/VLMs for planning and control (RT-2, Figure)
- **Open question**: is next-token prediction on enough data sufficient to learn world models, or is a fundamentally different approach needed?

## Safety & Alignment

<div class="diagram">
<div class="diagram-title">Safety Research Areas</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">Interpretability</div>
    <div class="card-desc">Understanding what's happening inside models. Sparse autoencoders, probing, mechanistic interpretability, circuit discovery.</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Robustness</div>
    <div class="card-desc">Jailbreak resistance, adversarial robustness, consistent behavior under distribution shift.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Governance</div>
    <div class="card-desc">EU AI Act, export controls, model evaluation for dangerous capabilities, responsible release practices.</div>
  </div>
</div>
</div>

Key debates:
- **Open vs closed weights**: open models enable research but also misuse
- **Scaling risk**: do larger models have qualitatively new risks?
- **Alignment tax**: how much capability do we sacrifice for safety?

## What Remains Unsolved

| Problem | Status |
|---------|--------|
| **Reliable reasoning** | Improving rapidly (o1, R1), but still fails on novel problems |
| **Catastrophic forgetting** | Models lose capabilities when fine-tuned for new tasks |
| **Continual learning** | Can't efficiently learn from a stream of new data without retraining |
| **Causal reasoning** | Models learn correlations, struggle with true causal understanding |
| **Planning** | Multi-step planning with long horizons remains fragile |
| **Grounding** | Language models don't truly "understand" — they pattern match (or do they?) |
| **Efficiency at the frontier** | Training a frontier model still costs $100M+ and takes months |
| **Evaluation** | No benchmark fully captures what we mean by "intelligent" |

## The Open-Source Flywheel

<div class="diagram">
<div class="diagram-title">The Virtuous Cycle</div>
<div class="cycle">
  <div class="cycle-step green">Meta/Mistral/DeepSeek release open weights</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step accent">Community fine-tunes, evaluates, improves</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step purple">Research papers discover new techniques (LoRA, DPO, etc.)</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step orange">Techniques adopted by labs → better open models → repeat</div>
</div>
</div>

## Looking Forward

The pace of progress in language modelling is extraordinary. A few predictions that are likely to age poorly:

1. **Models will get smaller and better** — the 8B model of 2026 will match the 70B model of 2024
2. **Hybrid architectures will win** — attention + SSMs + MoE, not a single architecture
3. **Reasoning will be the key differentiator** — raw knowledge can be retrieved, reasoning can't
4. **Inference will matter more than training** — test-time compute scaling is just beginning
5. **Agents will become the interface** — models that can act, not just answer

---

**You've reached the end of the guide.** 🎉

If you've read this far, you now have a comprehensive understanding of language modelling — from the mathematics of attention to the engineering of distributed training, from tokenization to the frontier of AI research.

The field moves fast. The best way to keep up is to **read papers, run experiments, and build things**.

---

**Appendices:**
- [Appendix A — A Brief History of Language Modelling](./appendix_a_history.md)
- [Appendix B — Tokenization Deep Dive](./appendix_b_tokenization.md)

[← Previous: Chapter 33 — Agents & Tool Use](./33_agents_and_tool_use.md) · [Back to Table of Contents](./README.md)

---

*Last updated: April 2026*
