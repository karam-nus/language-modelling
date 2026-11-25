---
title: "Chapter 12 — Mid-Training & Continued Pre-Training"
description: "The stage between pre-training and post-training: annealing on high-quality data, long-context extension (what Llama 3.1 actually did), domain adaptation with replay, fill-in-the-middle, and the 2025–26 shift to reasoning and agentic mid-training that prepares a base model for RL."
---

[← Back to Table of Contents](./README.md)

# Chapter 12 — Mid-Training & Continued Pre-Training

**Mid-training** is the stage between general pre-training and post-training. It keeps the next-token objective but changes almost everything else: the data gets smaller and better, the sequences get longer, the learning rate decays, and — since 2025 — the data starts to look like the reasoning and agent work the model will later be trained to do with RL.

> **The one-sentence version:** mid-training is where a base model is finished — annealed on its best data, stretched to long context, and seeded with the skills post-training will amplify.

## Where Mid-Training Sits

<div class="diagram">
<div class="diagram-title">Full Training Pipeline (2026)</div>
<div class="flow">
  <div class="flow-node accent wide">Pre-training <small>10–40T tokens, broad filtered mix, 4–8K context — the bulk of base-model compute</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Mid-training <small>~0.1–5T tokens: annealing on high-quality and synthetic data, long-context stages, reasoning and agent traces</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Supervised fine-tuning <small>~1–30M curated examples: instructions, reasoning, tool use</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Reinforcement learning <small>preferences + verifiable rewards + agent environments</small></div>
</div>
</div>

The boundary is fuzzy, and labs use the term differently. Qwen3 describes a three-stage pre-training: ~30T general tokens at 4K context, then ~5T tokens rich in STEM, code and reasoning, then a long-context stage. OLMo 2 calls its second stage "mid-training" (the Dolmino mix: ~50–300B high-quality tokens while the learning rate decays to zero). Both fit the definition used here.

## Annealing

The highest-leverage mid-training step is **annealing**: decay the learning rate (often to zero) while upweighting the best data — curated web, textbooks, maths, code, synthetic Q&A, instruction-like text.

<div class="diagram">
<div class="diagram-title">Annealing Phase</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Without annealing</div>
    <ul>
      <li>Learning rate decays on the general mix</li>
      <li>High-quality data diluted across the whole run</li>
      <li>Good, but leaves benchmark points on the table</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">With annealing</div>
    <ul>
      <li>LR → 0 over the final few percent of tokens</li>
      <li>Mix switched to the highest-quality sources</li>
      <li>Large gains on maths, code and knowledge benchmarks for small models</li>
    </ul>
  </div>
</div>
</div>

Two practical uses beyond the final model:

- **Data valuation.** Llama 3 annealed small models on candidate datasets and measured the benchmark lift to decide which data deserved a place in the mix — far cheaper than a full ablation run.
- **Checkpoint branching.** With a warmup-stable-decay schedule ([Chapter 10](./10_pretraining_at_scale.md)), any stable-phase checkpoint can be annealed into a finished model, so a lab can ship intermediate models without restarting.

Llama 3 405B linearly annealed its learning rate to zero over the final 40M tokens of pre-training, at 128K context, with high-quality sources upsampled, and averaged checkpoints from the annealing phase.

## Long-Context Extension

Models are pre-trained mostly at 4–8K tokens because attention cost grows quadratically. Long context is added afterwards in stages.

### What Llama 3.1 Actually Did

Llama 3.1 kept RoPE base 500,000 from pre-training and applied a frequency-dependent scaling (`rope_type: "llama3"`, factor 8: high-frequency pairs untouched, low-frequency pairs interpolated). It then **trained** on progressively longer sequences — six stages from 8K to 128K over about 800B tokens — moving to the next stage only once short-context benchmarks had recovered and needle-in-a-haystack retrieval was perfect at the current length.

<div class="diagram">
<div class="diagram-title">Long-Context Recipe (Llama 3.1 pattern)</div>
<div class="flow">
  <div class="flow-node accent wide">Pre-train at 8K <small>15T tokens, RoPE base 500K</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Rescale RoPE for the target length <small>NTK-by-parts / YaRN-style, see Chapter 2</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Train in stages: 8K → 16K → … → 128K <small>~800B tokens total; long documents upsampled, short data kept in the mix</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Gate each stage <small>short-context scores recovered + retrieval works at the new length</small></div>
</div>
</div>

Other 2024–26 recipes are cheaper: DeepSeek-V3 used YaRN and two 1,000-step stages (4K → 32K → 128K); Qwen3 and Kimi K2 also used YaRN-based extension. Models with 1M-token windows (Qwen2.5-1M, Llama 4, DeepSeek-V4, Kimi K3) add further stages and rely on sparse or hybrid attention ([Chapter 5](./05_attention_gqa_mqa_mla.md)) to make training at that length affordable at all.

| Method | What it changes | Where it shows up |
|--------|-------------|--------|
| **Adjusted base frequency (ABF)** | Raise RoPE θ | Code Llama (10K → 1M), Llama 3 (500K from the start), Qwen3 (1M) |
| **Position Interpolation** | Squeeze positions by the extension factor | LLaMA 1 → 32K (Chen et al., 2023) |
| **NTK-by-parts** | Interpolate only slow frequency pairs | Llama 3.1 (`rope_type: "llama3"`) |
| **YaRN** | NTK-by-parts + attention temperature | Qwen2.5/3, DeepSeek-V3, Kimi K2, gpt-oss |

<div class="diagram">
<div class="diagram-title">Context Extension Tensor Impact</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Short context</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">8K</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">pre-training context</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Extended context</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">128K</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">16× the KV cache; 256× the attention FLOPs per sequence</span>
  </div>
</div>
</div>

Long-context training needs **context parallelism** (splitting one sequence across GPUs, e.g. ring attention, [Chapter 26](./26_distributed_training.md)) as well as FlashAttention — the activations of a single 128K sequence do not fit on one GPU.

<div class="callout warn">
<span class="callout-title">Advertised ≠ effective context</span>
<p>A model that accepts 1M tokens is not necessarily able to <em>use</em> them. Benchmarks such as RULER, HELMET and NoLiMa repeatedly found effective lengths well below advertised windows, especially for multi-hop reasoning over the context. Evaluate on your own long-context task before trusting the number on the model card.</p>
</div>

## Domain Adaptation

Continued pre-training (CPT) adapts a general model to a domain by training further on domain data while replaying some general data.

<div class="diagram">
<div class="diagram-title">Domain-Adapted Models</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Code Llama (2023)</div>
    <div class="card-desc">Llama 2 + 500B code tokens; RoPE base raised to 1M; 16K training sequences extrapolating to 100K; infilling objective.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">DeepSeek-Coder-V2 (2024)</div>
    <div class="card-desc">Continued from a DeepSeek-V2 checkpoint with 6T more tokens (code, maths, text); 338 programming languages; 128K context.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Llemma (2023)</div>
    <div class="card-desc">Code Llama + 55B tokens of Proof-Pile-2 (maths papers, maths web, maths code). Strong on MATH for its size.</div>
  </div>
</div>
</div>

### Avoiding Catastrophic Forgetting

Training on domain data alone erodes general abilities. The standard recipe for CPT (Ibrahim et al., 2024) has three parts:

1. **Re-warm** the learning rate (a short warmup back to a moderate peak), then **re-decay** it.
2. **Replay** a slice of the original pre-training distribution — a few percent to a few tens of percent is usually enough.
3. Keep the **tokenizer** unless the domain is badly served (new scripts or languages); extending a vocabulary needs careful embedding initialisation.

```python
# Illustrative continued-pre-training mix
data_mix = {
    "domain_code":      0.60,   # the new domain
    "general_web":      0.25,   # replay to limit forgetting
    "curated_hq":       0.10,   # encyclopaedic, maths, books
    "instruction_like": 0.05,   # QA-style text keeps the model easy to post-train
}
```

| Mix strategy | Domain gain | General retention | Note |
|-------------|:-:|:-:|------|
| 100% domain | High | Low | Forgetting; fine only if general ability is irrelevant |
| Domain + 5–30% replay | High | Good | The common default |
| Progressive (domain share rises over time) | High | Good | More tuning, smoother transition |

### A Minimal CPT Setup

```python
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

base = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(base)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa")

raw = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
ds = raw.map(lambda b: tok(b["content"], truncation=True, max_length=4096),
             batched=True, remove_columns=raw.column_names)

args = TrainingArguments(
    output_dir="./llama-code-cpt",
    per_device_train_batch_size=2, gradient_accumulation_steps=16,
    learning_rate=3e-5,                 # re-warm to well below the pre-training peak (3e-4)
    lr_scheduler_type="cosine", warmup_ratio=0.02,
    num_train_epochs=1, bf16=True,
    gradient_checkpointing=True,        # trade ~20–30% speed for activation memory
    logging_steps=10, save_steps=1000,
)
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()   # labels = input_ids
```

Real CPT runs also mix in replay data, pack sequences instead of padding, and shard the model with FSDP across GPUs; an 8B model's full AdamW state alone is 128 GB ([Chapter 11](./11_optimizers_and_loss_functions.md)).

## Fill-in-the-Middle (FIM) Training

Code models add a **fill-in-the-middle** objective so they can complete code at the cursor rather than only at the end of a file:

```python
# Original document
# "def add(a, b):\n    return a + b\n"

# FIM transformation (applied to ~50% of training documents), prefix-suffix-middle order:
# "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n<|fim_middle|>return a + b"
#
# The model learns to generate the middle after seeing the prefix and the suffix.
```

Because the rearranged document is still trained left-to-right, FIM costs nothing in ordinary completion quality at a 50% rate (Bavarian et al., 2022). IDE completion sends prefix and suffix around the cursor in exactly this format.

## Reasoning and Agentic Mid-Training (2025–26)

The newest use of mid-training is to prepare a base model for reinforcement learning:

- **Reasoning data in pre-training.** Qwen3's second stage, Llama-Nemotron and many 2025–26 models put long chain-of-thought traces, maths and code solutions into mid-training, so the base model already "knows how to think" before RL.
- **Agent trajectories.** GLM-4.5 described a mid-training stage with repository-level code, synthetic reasoning and agent trajectories at long context; Kimi K2 and DeepSeek models similarly synthesise large volumes of tool-use data.
- **RL-readiness.** OctoThinker (2025) found that whether RL on a base model takes off depends heavily on the mid-training mix — Llama bases improved far more after mid-training on high-quality maths and reasoning data — which is one explanation for why some model families "respond to RL" and others do not.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>If RL mostly <em>amplifies</em> behaviours that already appear in the base model's samples, then mid-training decides what RL can discover. What would you put into a mid-training mix for a model that will later be RL-trained to use a terminal for hours at a time?</p>
</div>

## Key Takeaways

- Mid-training keeps next-token prediction but switches to smaller, better, longer data while the learning rate decays.
- Annealing on high-quality data gives outsized gains and doubles as a cheap data-valuation tool.
- Llama 3.1 reached 128K with frequency-dependent RoPE scaling plus ~800B tokens of staged long-context training — not by raising the base to 8M; YaRN-based recipes are cheaper.
- Continued pre-training = re-warm + re-decay + replay; FIM teaches code infilling at no cost to left-to-right ability.
- In 2025–26, mid-training carries reasoning traces and agent trajectories, shaping what RL can later amplify.

## Check Yourself

<details class="check">
<summary>A model was pre-trained at 8K tokens. How much larger are its KV cache and its per-sequence attention FLOPs at 128K?</summary>
<p>The KV cache grows linearly: 16×. Attention FLOPs per sequence grow quadratically: 16² = 256×.</p>
</details>

<details class="check">
<summary>Why does continued pre-training "re-warm" the learning rate rather than starting at the final, near-zero pre-training LR?</summary>
<p>At a near-zero LR the model barely moves and cannot absorb the new domain; jumping straight to a high LR causes a loss spike and forgetting. A short warmup to a moderate peak, followed by a fresh decay, adapts efficiently while replay data limits forgetting.</p>
</details>

<details class="check">
<summary>Why might an advertised 1M-token model fail a 200K-token multi-hop question?</summary>
<p>Accepting long inputs (positions in range, memory available) is not the same as retrieving and combining facts spread across them. Long-context training data rarely requires multi-hop use of distant information, so effective context — measured by RULER-style tasks — is often far below the advertised window.</p>
</details>

<details class="check">
<summary>In PSM-format FIM, what does the model see and what does it produce?</summary>
<p>It sees <code>&lt;fim_prefix&gt;</code> + prefix + <code>&lt;fim_suffix&gt;</code> + suffix + <code>&lt;fim_middle&gt;</code> and generates the middle, ending with an end-of-middle or EOS token.</p>
</details>

<details class="check">
<summary>How can annealing small models on a candidate dataset tell you whether to include it in a large run?</summary>
<p>Annealing magnifies the effect of the final data on benchmarks. If annealing a partially trained small model on a 50/50 mix of the candidate and the default high-quality data lifts target benchmarks more than the default alone, the data is likely valuable at scale — at a tiny fraction of the cost of a full training run.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Extending Context Window of LLMs via Positional Interpolation | Chen et al., 2023 | [arxiv.org/abs/2306.15595](https://arxiv.org/abs/2306.15595) | Position Interpolation |
| Code Llama | Rozière et al., 2023 | [arxiv.org/abs/2308.12950](https://arxiv.org/abs/2308.12950) | Code CPT, ABF, infilling |
| Llemma | Azerbayev et al., 2023 | [arxiv.org/abs/2310.10631](https://arxiv.org/abs/2310.10631) | Maths CPT |
| Efficient Training of Language Models to Fill in the Middle | Bavarian et al., 2022 | [arxiv.org/abs/2207.14255](https://arxiv.org/abs/2207.14255) | FIM |
| Simple and Scalable Strategies to Continually Pre-train LLMs | Ibrahim et al., 2024 | [arxiv.org/abs/2403.08763](https://arxiv.org/abs/2403.08763) | Re-warm, re-decay, replay |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | Long-context stages, annealing |
| DeepSeek-Coder-V2 | DeepSeek-AI, 2024 | [arxiv.org/abs/2406.11931](https://arxiv.org/abs/2406.11931) | CPT from a general checkpoint |
| 2 OLMo 2 Furious | OLMo Team, 2024 | [arxiv.org/abs/2501.00656](https://arxiv.org/abs/2501.00656) | Dolmino mid-training |
| Qwen3 Technical Report | Qwen Team, 2025 | [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) | Three-stage pre-training |
| RULER: What's the Real Context Size of Your Long-Context LMs? | Hsieh et al., 2024 | [arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654) | Effective vs advertised context |
| OctoThinker: Mid-training Incentivizes RL Scaling | Wang et al., 2025 | [arxiv.org/abs/2506.20512](https://arxiv.org/abs/2506.20512) | Mid-training and RL-readiness |

---

**Next:** [Chapter 13 — Fine-Tuning & Adaptation](./13_finetuning_and_adaptation.md) — turning a base model into an assistant with supervised examples.

[← Back to Table of Contents](./README.md)
