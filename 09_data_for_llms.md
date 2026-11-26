---
title: "Chapter 9 — Data for LLMs & VLMs"
description: "What the data looks like at every training phase — pre-training mixes and filtered web corpora, synthetic data, mid-training, SFT chat formats and loss masks, preference pairs, RL environments with verifiable rewards — with tensor shapes, datasets and the dedup and contamination checks around them."
related:
  - { id: training, why: "The training side of the data pipeline." }
---

[← Back to Table of Contents](./README.md)

# Chapter 9 — Data for LLMs & VLMs

> **The one-sentence version:** every training phase is next-token prediction or policy optimisation over a different kind of data — trillions of filtered web tokens, then curated and synthetic mixes, then conversations, then preferences, then environments that grade the model's own attempts.

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
  <div class="flow-node cyan wide">🤔 Post-Training: RL — verifiable rewards, rubrics, agent environments</div>
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
tokens = [464, 47058, 10478, 373, 5495, 287, 2177, 416, 569, 853, 648, ...]   # illustrative IDs
# shape: [seq_len]  e.g. [8192]
```

Multiple documents are concatenated with a special separator token `<|endoftext|>` and packed to fill `max_seq_len`, minimising padding waste:

```
doc1_tokens + [EOS] + doc2_tokens + [EOS] + doc3_tokens... → packed to [4096]
```

### Tensor Shapes — Pre-Training Batch

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | `[B, T]` | Token IDs fed into the model |
| `labels` | `[B, T]` | A copy of `input_ids`; the model shifts by one internally (position t predicts token t+1) |
| `attention_mask` | `[B, T]` | 1 for real tokens, 0 for padding (all 1s when packed) |
| `position_ids` | `[B, T]` | 0…T−1; reset at document boundaries when document masking is used |

Where `B` = sequences per step (summed over all GPUs and accumulation micro-steps) and `T` = sequence length (8K is typical for the main pre-training phase in 2024–26).

**Concrete example** (Llama 3 405B, from its paper): sequences grew from 4,096 to 8,192 tokens early in training while the global batch ramped from 4M to 8M to 16M tokens (≈ 2,000 sequences of 8K per step), over 15.6T tokens in total.

The logits for a whole step would be `[2048, 8192, 128256]` — 2.1 × 10¹² values, 4 TB in BF16. They never exist at once: each GPU holds only its micro-batch, and modern trainers compute the loss in **chunks** (or fused with the LM head, as in Liger Kernel and Apple's Cut Cross-Entropy) so that even one micro-batch's `[b, T, V]` logits are never fully materialised. With 128K–262K vocabularies, the output layer is one of the largest activation-memory consumers in the model.

### Pre-Training Data Mix

Labs rarely publish exact mixes. Llama 3's paper is one of the few that does, and it is a good reference point:

| Category (Llama 3 final mix) | Share | Why |
|--------|:--------------:|-----|
| General knowledge (filtered web, books, encyclopaedic) | ~50% | Breadth of world knowledge |
| Mathematical and reasoning data | ~25% | Maths, science, step-by-step text |
| Code | ~17% | Structure, precision, and a boost to reasoning |
| Multilingual | ~8% | Cross-lingual transfer |

Compared with 2020 (GPT-3: ~60% filtered Common Crawl, ~3% Wikipedia), the trend is clear: far more code and maths, and web text that has been filtered by model-based quality classifiers rather than simple heuristics. Mixes are chosen with small-scale **proxy experiments** — train many 1B-scale models on candidate mixes and extrapolate (RegMix, DoReMi-style methods, Llama 3's scaling-law approach).

**Key pre-training datasets**:

| Dataset | Size | Notes |
|---------|-------:|-------|
| The Pile (2020) | 825 GiB (~300B tokens) | 22 diverse sources; the early open standard |
| RedPajama v2 (2023) | 30T tokens (raw) | Common Crawl with quality signals to filter by |
| Dolma (2024) | 3T tokens | OLMo's corpus, fully documented |
| FineWeb / FineWeb-Edu (2024) | 15T / 1.3T tokens | Hugging Face; Edu keeps pages an LLM-trained classifier rates as educational |
| DCLM-Baseline (2024) | ~4T tokens | DataComp-LM; a simple fastText quality classifier beat elaborate heuristics |
| Nemotron-CC (2024) | 6.3T tokens | NVIDIA; classifier ensembles plus LLM rephrasing of lower-quality pages |
| FineWeb2 (2024–25) | 1,000+ languages | Multilingual FineWeb pipeline |
| The Stack v2 (2024) | ~900B tokens | Permissively licensed code (BigCode) |
| Common Pile v0.1 (2025) | 8 TB of text | Openly licensed and public-domain only |

## Synthetic Data

By 2025–26 a large and growing share of every frontier mix is **generated by other models**:

| Kind | What it is | Example |
|---|---|---|
| **Rephrased web** | An LLM rewrites noisy pages as clean Q&A, Wikipedia-style or textbook prose — same facts, better form | Nemotron-CC, WRAP, Kimi K2's rephrasing pipeline |
| **Synthetic textbooks and exercises** | Generated explanations and problems targeted at skills | Phi-1 → Phi-4 (about 40% synthetic tokens in Phi-4) |
| **Distilled reasoning traces** | Long chains of thought from a strong reasoning model, used in mid-training and SFT | OpenThoughts, DeepSeek-R1 distillation sets |
| **Self-generated RL data** | The model's own rollouts, filtered by a verifier | Every RLVR run (Chapter 35) |

Synthetic data is cheap, controllable and decontaminated by construction — but it copies the generator's blind spots and, used carelessly in loops, narrows the output distribution ("model collapse"). The practical rule: synthetic data should *re-express* or *verify* knowledge from real sources, not replace them.

<div class="callout warn">
<span class="callout-title">Provenance matters</span>
<p>Data sourcing is now a legal as well as a technical question. In 2025 a US court found training on lawfully purchased books to be fair use while letting claims over pirated copies proceed, and the case (<em>Bartz v. Anthropic</em>) settled for $1.5B. Openly licensed corpora such as Common Pile, and documented pipelines such as Dolma and FineWeb, exist partly so that provenance can be audited.</p>
</div>

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

| Dataset | Size | Source |
|---------|:----:|--------|
| FLAN Collection (2023) | 1.8K tasks, millions of examples | Templated academic NLP tasks |
| OpenHermes 2.5 (2023) | 1M | GPT-4-generated |
| UltraChat 200K (2023) | 200K | Filtered multi-turn synthetic dialogue |
| Tulu 3 SFT Mix (2024) | 939K | Curated open mix with safety and maths |
| SmolTalk / SmolTalk2 (2024–25) | ~1M+ | Hugging Face; includes reasoning traces |
| OpenThoughts3 (2025) | 1.2M | Reasoning traces for maths, code, science |
| Llama-Nemotron Post-Training (2025) | ~30M | NVIDIA; reasoning on/off SFT data |

Two shifts: SFT sets now carry **reasoning traces** (with a "thinking" segment the chat template wraps in special tokens) and **tool-call transcripts** (function calls and results as structured turns), and almost all of them are synthetic or model-filtered.

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
| Anthropic HH-RLHF (2022) | ~170K | Human helpfulness and harmlessness pairs |
| OpenAI WebGPT comparisons (2021) | 20K | Human comparisons of web-grounded answers |
| UltraFeedback (2023) | 64K prompts × 4 responses | GPT-4 ratings |
| HelpSteer2 / HelpSteer3 (2024–25) | ~21K / ~40K | NVIDIA, human multi-attribute ratings |
| Skywork-Reward-Preference-80K (2024) | 80K | Curated mix behind top open reward models |
| Tulu 3 preference mix (2024) | ~270K | On-policy pairs scored by an LLM judge |

---

## Post-Training Data: Reinforcement Learning (RL / GRPO / RLVR)

The latest frontier in post-training uses **verifiable rewards** — the model generates answers and receives a binary or graded reward signal based on correctness.

### RL with Verifiable Rewards (RLVR)

Used for math, code, and reasoning tasks where answers can be checked programmatically.

**Data format**: problems with ground-truth answers (or tests, or a grader)

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
| `advantages` | `[B×G]` | (r − group mean) / group std, broadcast over the rollout's tokens |
| `old_log_probs` | `[B×G, T]` | Log-probs under the policy that *generated* the rollouts (for the importance ratio) |
| `ref_log_probs` | `[B×G, T]` | Log-probs under the frozen reference model (for the KL penalty, if used) |
| `completion_mask` | `[B×G, T]` | 1 on generated tokens, 0 on prompt and padding |

**Key RL datasets**:

| Dataset | Size | Domain |
|---------|:----:|--------|
| GSM8K (2021) | 8.5K | Grade-school maths |
| MATH (2021) | 12.5K | Competition mathematics |
| APPS (2021) | 10K | Programming problems with tests |
| NuminaMath-1.5 (2025) | ~900K | Olympiad-style maths with verified answers |
| DeepScaleR (2025) | 40K | Curated hard maths for RL |
| Big-Math (2025) | 250K | Filtered for RL-friendly, checkable answers |
| SWE-Gym / SWE-smith (2025) | thousands of repo tasks | Executable software-engineering environments |

### From Datasets to Environments

The 2025–26 shift in RL data is from static problem sets to **environments**: a container with a repository and a test suite, a terminal, a browser, a spreadsheet, a simulated customer. The model acts for many turns, and a grader scores the end state. Labs now build and buy environments the way they once bought labelled data, and the bottleneck is writing graders that cannot be gamed ("reward hacking", Chapter 15). For tasks without a checkable answer — writing, advice, research — **rubric-based rewards** (an LLM judge scoring against a task-specific checklist) extend RL beyond maths and code.

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

Datasets: OpenAI's PRM800K (800K human step labels on MATH solutions); Math-Shepherd (labels estimated automatically by rolling out from each step). After DeepSeek-R1 showed that outcome-only rewards suffice for strong reasoning, PRMs moved from training signal to mostly *verification* and search (Chapter 35).

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
| LAION-400M / LAION-5B (2021–22) | 0.4B / 5.85B pairs | Web alt-text; 5B withdrawn in 2023 after harmful content was found, re-released as Re-LAION-5B (2024) |
| COYO-700M (2022) | 700M pairs | Stricter filtering |
| DataComp CommonPool (2023) | 12.8B pairs | Benchmark for filtering strategies |
| OBELICS / MINT-1T (2023–24) | 141M docs / 1T tokens | *Interleaved* image–text documents |

Since 2024, most VLM pre-training data is **re-captioned**: alt-text is replaced or augmented with dense captions written by an existing VLM, and OCR, chart, table and UI-screenshot data are synthesised at scale.

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
| LLaVA-1.5 Mix (2023) | 665K | Academic VQA + GPT-4-generated conversations |
| ShareGPT4V (2023) | 100K (+1.2M) | Detailed captions |
| The Cauldron (2024) | 50 datasets | Hugging Face collection used for Idefics2 |
| Cambrian-10M (2024) | 10M | Large open visual-instruction mix |
| FineVision (2025) | ~24M samples | Hugging Face; deduplicated and decontaminated open VLM mix |
| DocVQA (2020) | 50K questions | Document understanding (mostly an eval) |

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
- **Phi-1** (2023): a 1.3B model trained on "textbook-quality" filtered and synthetic code reached 50.6% on HumanEval, beating models more than 10× larger.
- **LIMA** (2023): 1,000 carefully curated SFT examples were enough to elicit good chat behaviour from a strong base model — the "superficial alignment hypothesis".
- **FineWeb-Edu, DCLM** (2024): a cheap classifier trained on LLM judgements of "educational value" improved downstream benchmarks more than any heuristic filter, at the cost of discarding ~90% of the web.
- **MiniCPM, Llama 3** (2024): **annealing** — upweighting the best data while the learning rate decays to zero — gives outsized gains, and is now the standard end of pre-training (Chapter 12).

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

**Contamination detection**: n-gram overlap between training data and benchmark test sets (GPT-3 used 13-grams; Llama 3 used 8-grams), plus behavioural checks — does the model complete benchmark questions verbatim, or score far better on old versions of a benchmark than on fresh ones? Contamination is why 2025–26 evaluation favours private or continuously refreshed test sets (LiveCodeBench, LiveBench, SWE-bench variants) — see [Chapter 30](./30_evaluation_and_benchmarks.md).

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
| Agentic RL | many turns per episode | reward on the final state | Tool outputs are masked from the loss |
| VLM pre-train | `[B, C, H, W]` + `[B, T]` | `[B, T]` | Image + text |

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Quality filters like FineWeb-Edu throw away ~90% of the web, and frontier runs already train on 15–40T tokens. If high-quality human text runs out before compute does, which of this chapter's tools — rephrasing, synthetic textbooks, multimodal data, RL environments, repeating data for several epochs — buys the most, and which risks narrowing what the model knows?</p>
</div>

## Key Takeaways

- Each phase has its own data: filtered web + code + maths (pre-training), curated and long-context mixes (mid-training), chat transcripts with assistant-only loss (SFT), preference pairs (RLHF/DPO), and graded rollouts in environments (RL).
- Pre-training data is now selected by model-based quality classifiers and increasingly rephrased or generated by LLMs; synthetic data should re-express real knowledge, not replace it.
- Shapes are simple — `[B, T]` everywhere, `[B×G, T]` for group RL — but the `[B, T, V]` logits are large enough that loss is computed in chunks.
- RL data has become environments plus graders; reward hacking and rubric design are the new data-quality problems.
- Deduplicate (exact, MinHash, semantic) and check contamination; provenance is now a legal requirement, not only good hygiene.

## Check Yourself

<details class="check">
<summary>In Hugging Face, why do you pass <code>labels = input_ids.clone()</code> rather than labels shifted by one?</summary>
<p>The model shifts internally: logits at position t are compared with labels at position t+1, and the last position has no target. Shifting yourself would make the model predict two tokens ahead.</p>
</details>

<details class="check">
<summary>An SFT batch has 1,200 prompt tokens and 300 assistant tokens per example. How many positions contribute to the loss, and what goes in <code>labels</code> for the rest?</summary>
<p>Only the 300 assistant tokens (plus the end-of-turn token). The 1,200 prompt positions get −100, the index PyTorch's cross-entropy ignores.</p>
</details>

<details class="check">
<summary>In GRPO, what is the difference between <code>old_log_probs</code> and <code>ref_log_probs</code>?</summary>
<p><code>old_log_probs</code> come from the policy that generated the rollouts and form the importance ratio π<sub>θ</sub>/π<sub>old</sub> that PPO-style clipping acts on. <code>ref_log_probs</code> come from a frozen reference (usually the SFT model) and feed an optional KL penalty that keeps the policy from drifting too far.</p>
</details>

<details class="check">
<summary>Why did DCLM's simple fastText filter beat sophisticated heuristic pipelines?</summary>
<p>It was trained to recognise text resembling high-quality reference data (instruction data and highly rated forum answers), so it optimised directly for "looks like what helps downstream", whereas heuristics (length, punctuation, stop-words) only remove obvious junk.</p>
</details>

<details class="check">
<summary>Name two ways benchmark contamination can be detected, and one way evaluation designers now avoid it.</summary>
<p>N-gram overlap between the training corpus and test items; behavioural probes such as verbatim completion of test questions or a performance gap between old and freshly written items. Avoidance: private held-out sets or continuously refreshed benchmarks dated after the model's training cutoff.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| The Pile | Gao et al., 2020 | [arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027) | Early open pre-training corpus |
| The FineWeb Datasets | Penedo et al., 2024 | [arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557) | Web filtering, FineWeb-Edu |
| DataComp-LM | Li et al., 2024 | [arxiv.org/abs/2406.11794](https://arxiv.org/abs/2406.11794) | Classifier-based filtering |
| Nemotron-CC | Su et al. (NVIDIA), 2024 | [arxiv.org/abs/2412.02595](https://arxiv.org/abs/2412.02595) | Rephrasing and classifier ensembles |
| Textbooks Are All You Need (Phi-1) | Gunasekar et al., 2023 | [arxiv.org/abs/2306.11644](https://arxiv.org/abs/2306.11644) | Synthetic, textbook-quality data |
| Phi-4 Technical Report | Abdin et al., 2024 | [arxiv.org/abs/2412.08905](https://arxiv.org/abs/2412.08905) | Synthetic data at scale |
| LIMA: Less Is More for Alignment | Zhou et al., 2023 | [arxiv.org/abs/2305.11206](https://arxiv.org/abs/2305.11206) | Small high-quality SFT |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | Data mix, batch schedule, contamination |
| Tulu 3 | Lambert et al. (Ai2), 2024 | [arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124) | Open SFT, preference and RLVR data |
| Let's Verify Step by Step (PRM800K) | Lightman et al., 2023 | [arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050) | Process supervision data |
| DeepSeekMath (GRPO) | Shao et al., 2024 | [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) | GRPO data layout |
| Deduplicating Training Data Makes Language Models Better | Lee et al., 2021 | [arxiv.org/abs/2107.06499](https://arxiv.org/abs/2107.06499) | Exact and near-duplicate removal |

---

**Next:** [Chapter 10 — Pre-Training at Scale](./10_pretraining_at_scale.md) — how a trillion-token run is actually set up and kept alive.

[← Back to Table of Contents](./README.md)
