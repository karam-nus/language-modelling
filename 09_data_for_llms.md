---
title: "Chapter 9 — Data for LLMs & VLMs"
---

[← Back to Table of Contents](./README.md)

# Chapter 9 — Data for LLMs & VLMs

> *"A model is only as good as its data. Every capability you see in a frontier model traces back to some text, image, or interaction in the training set."*

## The Data Hierarchy

Modern LLM/VLM training unfolds in distinct phases — each with its own data format, objective, and tensor shapes:

<div class="diagram">
<div class="diagram-title">Data Across Training Phases</div>
<div class="flow">
  <div class="flow-node accent wide">📚 Pre-Training — raw internet text, code, books (trillions of tokens)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🔧 Mid-Training / Continued Pre-Training — curated domain/long-context data</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🎯 Post-Training: SFT — instruction-response pairs (millions of examples)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">⚖️ Post-Training: RLHF/DPO — preference pairs, reward signals</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">🤔 Post-Training: RL / Process Rewards — reasoning traces, verifiable rewards</div>
</div>
</div>

---

## Pre-Training Data

Pre-training builds world knowledge and language understanding. Models learn by predicting the next token over massive, diverse corpora.

### What Pre-Training Data Looks Like

**Objective**: Next-token prediction (causal language modelling)

**Data source examples**: Common Crawl, GitHub, Wikipedia, ArXiv, Books, StackExchange

**Raw example** (one document, packed into a context window):

```
The transformer architecture was introduced in 2017 by Vaswani et al.
It replaced recurrent networks with self-attention, enabling parallelism...
```

**After tokenization and packing**:

```
tokens = [464, 47058, 10478, 373, 5495, 287, 2177, 416, 569, 853, 648, ...]
# shape: [seq_len]  e.g. [4096]
```

Multiple documents are concatenated with a special separator token `<|endoftext|>` and packed to fill `max_seq_len`, minimising padding waste:

```
doc1_tokens + [EOS] + doc2_tokens + [EOS] + doc3_tokens... → packed to [4096]
```

### Tensor Shapes — Pre-Training Batch

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | `[B, T]` | Token IDs fed into the model |
| `labels` | `[B, T]` | Same as `input_ids`, shifted by 1 position |
| `attention_mask` | `[B, T]` | 1 for real tokens, 0 for padding |
| `position_ids` | `[B, T]` | 0…T-1, reset at document boundaries |

Where `B` = batch size (e.g., 1024–4096 with gradient accumulation), `T` = sequence length (e.g., 4096–8192).

**Concrete example** (LLaMA-3 8B training):
- `B = 1024`, `T = 4096` → 4.2M tokens per step
- Global batch: often 4M–16M tokens
- `input_ids`: `[1024, 4096]` of dtype `int32`
- Logits output: `[1024, 4096, 128256]` (vocab size 128k)
- Loss: scalar (mean cross-entropy over all `B×T` positions)

### Pre-Training Data Mix

| Source | Typical Weight | Why |
|--------|:--------------:|-----|
| Common Crawl (filtered) | 60–70% | Broad world knowledge |
| Code (GitHub, etc.) | 10–15% | Reasoning, structure |
| Books / long documents | 5–10% | Coherence, narrative |
| Wikipedia / encyclopaedias | 3–5% | Factual, clean |
| Scientific papers (ArXiv) | 2–5% | STEM reasoning |
| Multilingual web | 5–10% | Cross-lingual transfer |

**Key pre-training datasets**:

| Dataset | Tokens | Notes |
|---------|-------:|-------|
| The Pile | 825B | 22 diverse sources, early open standard |
| RedPajama v2 | 30T+ | Filtered Common Crawl + others |
| Dolma | 3T | OLMo training set, fully open |
| FineWeb | 15T | Hugging Face, high-quality web filter |
| DCLM | 4T | DataComp competition winner |

---

## Mid-Training / Continued Pre-Training Data

Mid-training extends or adapts a pre-trained model. The data format is identical to pre-training (next-token prediction), but the **source mix changes** to achieve specific goals.

### Goals and Data Strategies

<div class="diagram">
<div class="diagram-title">Mid-Training Use Cases</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Long-Context Extension</div>
    <div class="card-desc">Add long documents (books, codebases). Change T from 4096 → 32K/128K. Adjust RoPE base frequency.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Domain Adaptation</div>
    <div class="card-desc">Upsample domain data (medical, legal, code). Maintain diversity to prevent forgetting.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Language Extension</div>
    <div class="card-desc">Add new languages post-hoc. Mix with original language data to preserve performance.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Annealing / Data Cleaning</div>
    <div class="card-desc">Final training phase: heavily upweight high-quality data (textbooks, curated Q&A).</div>
  </div>
</div>
</div>

**Tensor shapes**: identical to pre-training but `T` is larger:
- Long-context: `input_ids` shape `[B, 32768]` or `[B, 131072]`
- Batch size `B` decreases as `T` increases to maintain memory budget

---

## Post-Training Data: SFT (Supervised Fine-Tuning)

SFT teaches the model to follow instructions. Data is formatted as conversations.

### Chat Format

Modern models use a structured chat template. Example (Llama-3 chat format):

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Explain gradient descent in simple terms.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Gradient descent is an optimisation algorithm...
<|eot_id|>
```

**Loss masking**: loss is computed **only on assistant tokens** (instruction/system tokens are masked):

```python
# labels shape: [B, T]
# -100 is the ignore index for cross-entropy loss
labels = [-100, -100, ..., -100,   # system + user tokens (masked)
           tok1, tok2, ..., tokN]   # assistant tokens (loss computed here)
```

### SFT Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| `input_ids` | `[B, T]` | Full conversation including system/user/assistant |
| `labels` | `[B, T]` | -100 for masked positions, token IDs for assistant |
| `attention_mask` | `[B, T]` | All 1s (no padding if packing) |

**Typical dimensions**: `B=4–32`, `T=2048–8192`

### SFT Data Sample

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is backpropagation?"},
    {"role": "assistant", "content": "Backpropagation is the algorithm used to train neural networks. It computes gradients of the loss with respect to each parameter by applying the chain rule of calculus backwards through the network..."}
  ]
}
```

**Key SFT datasets**:

| Dataset | Size | Quality | Source |
|---------|:----:|:-------:|--------|
| OpenHermes 2.5 | 1M | High | GPT-4 generated |
| UltraChat 200K | 200K | Very High | Filtered multi-turn |
| Tulu 3 SFT Mix | 939K | Very High | Curated open-source |
| FLAN Collection | 1.8M | Medium | Multi-task NLP |
| ShareGPT | 90K | Medium | Real ChatGPT conversations |

---

## Post-Training Data: RLHF & Preference Learning

After SFT, alignment training uses **comparison data** — pairs of responses where humans (or models) indicate which is better.

### Reward Model Data

Used to train a reward model that scores responses:

```json
{
  "prompt": "Explain quantum entanglement.",
  "chosen": "Quantum entanglement is a phenomenon where two particles...",
  "rejected": "Quantum entanglement is when particles are quantum."
}
```

**Tensor shapes for reward model training**:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `chosen_input_ids` | `[B, T]` | Preferred response tokens |
| `rejected_input_ids` | `[B, T]` | Dispreferred response tokens |
| `chosen_reward` | `[B]` | Scalar reward for chosen |
| `rejected_reward` | `[B]` | Scalar reward for rejected |

The reward model outputs a scalar from the last token's hidden state:

```
hidden_states: [B, T, d_model]  →  last token: [B, d_model]  →  reward: [B, 1]
```

### DPO / Direct Preference Optimisation Data

DPO skips the reward model and trains directly on preference pairs:

```
(prompt, chosen_response, rejected_response)
```

The DPO loss uses log-probabilities of both responses under the current and reference model.

**Key preference datasets**:

| Dataset | Pairs | Notes |
|---------|------:|-------|
| UltraFeedback | 64K prompts | GPT-4 rated, 4 responses each |
| HelpSteer2 | 21K | NVIDIA human annotations |
| Anthropic HH-RLHF | 170K | Human preference pairs |
| OpenAI WebGPT | 20K | Human + web comparison |

---

## Post-Training Data: Reinforcement Learning (RL / GRPO / RLVR)

The latest frontier in post-training uses **verifiable rewards** — the model generates answers and receives a binary or graded reward signal based on correctness.

### RL with Verifiable Rewards (RLVR)

Used for math, code, and reasoning tasks where answers can be checked programmatically.

**Data format**: problems with ground-truth answers

```json
{
  "problem": "Solve: 3x + 7 = 22. Find x.",
  "answer": "5",
  "domain": "algebra"
}
```

**Training loop**:
1. Model generates multiple completions (rollouts): `[B × G, T]` where `G` = group size (e.g., 8)
2. Each completion is scored: correct answer → reward `+1`, wrong → `0` or `-1`
3. GRPO computes advantage by comparing against group mean reward
4. Policy gradient update improves high-reward completions

**Tensor shapes for GRPO**:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | `[B×G, T]` | B problems × G rollouts each |
| `rewards` | `[B×G]` | Scalar reward per rollout |
| `advantages` | `[B×G]` | Normalised within group |
| `old_log_probs` | `[B×G, T]` | Log-probs from reference policy |

**Key RL datasets**:

| Dataset | Size | Domain |
|---------|:----:|--------|
| MATH | 12.5K | Competition mathematics |
| GSM8K | 8.5K | Grade school math |
| APPS | 10K | Programming problems |
| LiveCodeBench | 400+ | Recent coding contests |
| NuminaMath-TIR | 860K | Tool-integrated reasoning |

### Process Reward Models (PRMs)

Instead of just rewarding the final answer, **PRMs** give step-level rewards for each reasoning step in a chain-of-thought.

**Data format** (step-annotated):

```json
{
  "problem": "If a train travels at 60 mph for 2.5 hours, how far does it go?",
  "steps": [
    {"text": "Distance = speed × time", "correct": true},
    {"text": "Distance = 60 × 2.5", "correct": true},
    {"text": "Distance = 150 miles", "correct": true}
  ]
}
```

**Reward tensor shapes**:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `step_rewards` | `[B, num_steps]` | +1/-1 per step |
| `hidden_states` | `[B, T, d_model]` | At step-boundary positions |

---

## VLM Data — Vision-Language Models

VLMs require paired image-text data across all training phases.

### Pre-Training VLM Data

Large-scale image-text pairs (alt-text, web captions):

```
[image_tokens (256–2048 tokens)] + [caption_tokens]
```

**Tensor shapes** (vision encoder output fused with language model):

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pixel_values` | `[B, C, H, W]` | Raw images e.g. `[B, 3, 224, 224]` |
| `image_features` | `[B, N_patches, d_vision]` | ViT output e.g. `[B, 256, 1024]` |
| `projected_features` | `[B, N_patches, d_llm]` | After MLP projector e.g. `[B, 256, 4096]` |
| `input_ids` | `[B, T_text + N_patches]` | Image tokens + text tokens concatenated |

**Key VLM pre-training datasets**:

| Dataset | Scale | Notes |
|---------|------:|-------|
| LAION-400M | 400M pairs | Web-scraped, noisy |
| LAION-5B | 5B pairs | Larger, filtered version |
| COYO-700M | 700M pairs | Higher quality filtering |
| DataComp | 1.28B–12.8B | Competition benchmark |

### SFT for VLMs

Structured conversations with image inputs:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<base64_or_path>"},
        {"type": "text", "text": "Describe what you see in this image."}
      ]
    },
    {
      "role": "assistant",
      "content": "The image shows a sunset over the ocean..."
    }
  ]
}
```

**Key VLM instruction datasets**:

| Dataset | Size | Type |
|---------|:----:|------|
| LLaVA-1.5 Mix | 665K | GPT-4V generated |
| ShareGPT4V | 100K | GPT-4V detailed captions |
| MMDU | 20K | Multi-image dialogue |
| DocVQA | 50K | Document understanding |

---

## Data Quality vs. Quantity

A common insight from scaling: **quality beats raw token count** at a fixed compute budget.

<div class="diagram">
<div class="diagram-title">The Data Quality Spectrum</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Quantity-First (Pre-Training)</div>
    <ul>
      <li>Trillions of tokens</li>
      <li>Noisy, diverse web data</li>
      <li>Minimal filtering (language ID, dedup)</li>
      <li>Goal: broad knowledge coverage</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Quality-First (Post-Training)</div>
    <ul>
      <li>Thousands to millions of examples</li>
      <li>Carefully curated/verified</li>
      <li>Every example counts → no duplicates</li>
      <li>Goal: targeted capability shaping</li>
    </ul>
  </div>
</div>
</div>

Key findings from research:
- **Phi-1** (2023): 1.3B model trained on "textbook quality" synthetic data matched models 10× larger
- **LIMA** (2023): 1000 carefully curated SFT examples matched 52K example models on alignment
- **MiniCPM** (2024): Annealing on high-quality data in final training stage boosts benchmarks significantly

---

## Data Contamination & Deduplication

A critical but often overlooked issue: **benchmark contamination** — when test data leaks into training data.

**Deduplication methods**:

| Method | Scale | Use Case |
|--------|:-----:|----------|
| Exact match (SHA hash) | Exact duplicates | Fast, handles verbatim copies |
| MinHash LSH | Near-duplicates | Handles minor edits |
| SimHash | Near-duplicates | Streaming-friendly |
| SemDeDup | Semantic duplicates | Embedding-based clustering |

**Contamination detection**: n-gram overlap between training data and benchmark test sets. Most responsible training runs report contamination analysis.

---

## Summary: Data Shapes Across Phases

| Phase | Input Shape | Label Shape | Key Characteristic |
|-------|------------|-------------|-------------------|
| Pre-training | `[B, T]` | `[B, T]` (shifted) | All tokens labelled |
| Mid-training | `[B, T_long]` | `[B, T_long]` | T up to 128K |
| SFT | `[B, T]` | `[B, T]` (-100 for non-assistant) | Masked instruction tokens |
| Reward model | `[2B, T]` | scalar `[B]` per pair | Paired chosen/rejected |
| DPO | `[2B, T]` | implicit from log-probs | No explicit reward model |
| GRPO/RLVR | `[B×G, T]` | `[B×G]` reward | Multiple rollouts per prompt |
| VLM pre-train | `[B, C, H, W]` + `[B, T]` | `[B, T]` | Image + text |

## What's Next

With a solid understanding of data across all training phases, we're ready to look at **pre-training at scale** — how you orchestrate a trillion-token training run.

[← Previous: Chapter 8 — Multimodal Models](./08_multimodal_models.md) · **Next: [Chapter 10 — Pre-Training at Scale →](./10_pretraining_at_scale.md)**

---

*Last updated: April 2026*
