---
title: "Chapter 1 — Introduction to Language Modelling"
description: "What a language model is, the text-to-prediction pipeline with tensor shapes, a short history, tokenization essentials, and the 2026 model lifecycle from pre-training to test-time compute."
related:
  - { id: machine-learning, why: "The machine-learning foundations this guide assumes." }
  - { id: pytorch-essentials, why: "The PyTorch you need to run every code block here." }
---

[← Back to Table of Contents](./README.md)

# Chapter 1 — Introduction to Language Modelling

A **language model** assigns probabilities to sequences of tokens. At its core the task is deceptively small: given some text, predict what comes next. That single objective — **next-token prediction** — scaled up with enough data, compute and post-training, produces systems that write code, prove theorems, browse the web and operate software.

> **The one-sentence version:** a language model is a learned function from a prefix of tokens to a probability distribution over the next token, and everything else in this guide is about making that function better, cheaper or safer.

## The Core Idea

Every modern large language model (LLM) — GPT-5.x, Claude, Gemini, Llama, DeepSeek, Qwen, Kimi — is trained on some variant of this principle. The model reads a sequence of tokens and learns the conditional distribution of the next one:

$$P(x_{t+1} \mid x_1, x_2, \ldots, x_t)$$

By the chain rule of probability, a model of these conditionals is a model of whole sequences:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

Training minimises the **negative log-likelihood** (cross-entropy) of real text under this product. Generation runs it forwards: predict a distribution, pick a token, append it, repeat.

<div class="diagram">
<div class="diagram-title">The Language Modelling Objective</div>
<div class="flow-h">
  <div class="flow-node accent">The cat sat</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">on <small>p = 0.83</small></div>
</div>
</div>

Whether the result "understands" anything is a real debate (Chapter 38). What is not debated is that the representations a model builds to predict text well are rich enough to support reasoning-like behaviour, and that post-training can shape that behaviour into something useful.

## From Text to Tensors — The Full Pipeline

Before a model can process text, the text must become numbers; before those numbers mean anything, they must become vectors. Each stage below has its own chapter:

<div class="diagram">
<div class="diagram-title">The Language Model Pipeline</div>
<div class="flow">
  <div class="flow-node wide">📝 Raw Text <small>"The cat sat on"</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">🔤 Tokenizer <small>text → token IDs (Appendix B)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">📊 Embedding Layer <small>token IDs → dense vectors (Ch 2)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🧠 Transformer Blocks <small>attention + feed-forward × N (Ch 3–5)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">📈 Output Head <small>hidden states → vocabulary logits</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🎯 Sampling <small>logits → next token (Ch 16)</small></div>
</div>
</div>

And the tensor shapes at each stage — a pattern you will see throughout the guide:

<div class="diagram">
<div class="diagram-title">Tensor Shapes Through the Pipeline</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Token IDs</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-bracket">]</span></div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Embeddings</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d_model</span><span class="ts-bracket">]</span></div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After N blocks</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">d_model</span><span class="ts-bracket">]</span></div>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Logits</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim vocab">V</span><span class="ts-bracket">]</span></div>
  </div>
</div>
</div>

**B** is batch size, **T** sequence length, **d_model** the hidden width (4,096 for Llama 3.1 8B, 7,168 for DeepSeek-V3) and **V** the vocabulary size (32,000 for Llama 2; 128,256 for Llama 3; 151,936 for Qwen3; 262,144 for Gemma 3). Vocabularies have grown about 4–8× since 2023 because bigger vocabularies compress multilingual text and code into fewer tokens, and fewer tokens means less compute per document.

## A Brief History

Language modelling has gone through several paradigm shifts. [Appendix A](./appendix_a_history.md) has the full story; the milestones:

<div class="diagram">
<div class="diagram-title">Key Milestones in Language Modelling</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">1948–1990s</div>
    <div class="timeline-title">Statistical era</div>
    <div class="timeline-desc">Shannon's information theory and n-gram experiments → smoothed n-gram models for speech recognition and translation</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2003</div>
    <div class="timeline-title">Neural language models</div>
    <div class="timeline-desc">Bengio et al. learn word vectors and next-word probabilities jointly in one network</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2013–2015</div>
    <div class="timeline-title">Word vectors and attention</div>
    <div class="timeline-desc">word2vec makes embeddings cheap; Bahdanau attention lets a decoder look back at every input position</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2017</div>
    <div class="timeline-title">The Transformer</div>
    <div class="timeline-desc">"Attention Is All You Need" drops recurrence entirely; training parallelises across the sequence</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2018–2020</div>
    <div class="timeline-title">Pre-training + scaling laws</div>
    <div class="timeline-desc">GPT, BERT, T5, GPT-3; Kaplan et al. show loss falls as a power law in parameters, data and compute</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2022–2023</div>
    <div class="timeline-title">Alignment and open weights</div>
    <div class="timeline-desc">InstructGPT/ChatGPT make RLHF mainstream; Chinchilla resets the data/parameter balance; LLaMA starts the open-weight wave</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2024–2025</div>
    <div class="timeline-title">Reasoning models</div>
    <div class="timeline-desc">o1 and DeepSeek-R1 show that RL on verifiable rewards plus long chains of thought trades inference compute for accuracy; MoE becomes the default at the frontier</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2026</div>
    <div class="timeline-title">Agents and efficient long context</div>
    <div class="timeline-desc">Models are trained as agents that call tools for hours; hybrid and sparse attention (Qwen3.5, DeepSeek-V4) make million-token contexts affordable; FP8/FP4 training and serving go mainstream</div>
  </div>
</div>
</div>

## The Modern Model Lifecycle (2026)

The pipeline above is what a model *is*. What a model *goes through* is a second pipeline, and it has grown stages since the original GPT recipe of "pre-train, then fine-tune":

<div class="diagram">
<div class="diagram-title">How a 2026 Frontier Model Is Made</div>
<div class="flow">
  <div class="flow-node accent wide">🌊 Pre-training <small>10–40T tokens of filtered web, code, books, synthetic data · next-token loss (Ch 9–11)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🧭 Mid-training <small>higher-quality mix, long-context extension, reasoning and agent traces, learning-rate annealing (Ch 12)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🎓 Supervised fine-tuning <small>instruction and tool-use demonstrations, chat template (Ch 13–14)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">🏆 Reinforcement learning <small>preference RL (RLHF/DPO) + RL with verifiable rewards on maths, code and agent tasks (Ch 15, 35)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">🗜️ Compression & serving <small>distillation into smaller models, FP8/FP4 quantization, KV-cache tricks (Ch 17–23, 31)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">⏱️ Test-time compute <small>thinking budgets, tool calls, search, parallel sampling (Ch 16, 35, 37)</small></div>
</div>
</div>

Two shifts define the current era:

1. **RL is now a large slice of compute.** In 2022 post-training was a rounding error next to pre-training. Since o1 and DeepSeek-R1, labs run reinforcement learning on tasks with checkable answers — unit tests, maths solutions, agent environments — for long enough that it becomes a substantial part of the total training budget.
2. **Inference is where capability is bought.** A reasoning model can spend 100 tokens or 100,000 tokens on the same question. Accuracy rises with the budget, so serving efficiency (Chapters 16–18, 31) is no longer just a cost concern but a capability one.

<div class="callout trend">
<span class="callout-title">2026 trend</span>
<p>The line between "model" and "agent" has blurred. Frontier releases are now benchmarked on multi-hour software and research tasks (SWE-bench Verified, Terminal-Bench, METR time-horizon studies) rather than single-turn quizzes, and the models are trained inside the same tool-using harnesses they are deployed in. Chapter 37 covers the machinery.</p>
</div>

## Tokenization Essentials

Before a model sees any text, a **tokenizer** splits it into discrete units called **tokens**. Modern LLMs use **subword tokenization** — between characters and whole words — so that rare words, code and any language can be represented without an unbounded vocabulary.

[Appendix B](./appendix_b_tokenization.md) walks through the algorithms. What you need now:

| Algorithm | Used by | Key idea |
|-----------|---------|----------|
| **Byte-level BPE** | GPT-2 → GPT-5 (tiktoken), Llama 3/4, Qwen, DeepSeek, Mistral (Tekken) | Start from the 256 bytes; repeatedly merge the most frequent adjacent pair. Never produces an unknown token |
| **WordPiece** | BERT, DistilBERT | Like BPE, but picks merges that most increase training-data likelihood |
| **Unigram LM** | T5, ALBERT, XLNet (via SentencePiece) | Start with a large candidate vocabulary and prune the pieces that matter least |

**SentencePiece** is a library, not an algorithm: it implements both BPE and Unigram directly on raw text (spaces included), which is why Llama 2 and T5 both "use SentencePiece" with different algorithms. **tiktoken** is OpenAI's fast byte-level BPE implementation; Llama 3 switched to a tiktoken-style tokenizer with a 128K vocabulary.

The tokenizer fixes the vocabulary size **V**, which sets the size of the embedding table and the output layer. In 2026, V = 128K–262K is typical for new models.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

text = "Language models predict the next token."
ids = tok.encode(text)                       # adds <|begin_of_text|> by default
print(tok.convert_ids_to_tokens(ids))
print(len(ids), "tokens;", tok.vocab_size, "base vocab;", len(tok), "with special tokens")
# ['<|begin_of_text|>', 'Language', 'Ġmodels', 'Ġpredict', 'Ġthe', 'Ġnext', 'Ġtoken', '.']
# 8 tokens; 128000 base vocab; 128256 with special tokens
```

The `Ġ` is how byte-level BPE displays a leading space: the space is part of the token, so `" models"` and `"models"` are different IDs. This is why prompts that differ only in whitespace can behave differently.

<div class="callout note">
<span class="callout-title">Rule of thumb</span>
<p>English prose averages about 0.75 words per token (≈ 1.3 tokens per word) with 100K+ vocabularies. Code, non-Latin scripts and numbers cost more tokens per character — and since price, context length and latency are all counted in tokens, the tokenizer quietly sets the economics of a model.</p>
</div>

## The Simplest Language Model

To build intuition, here is the simplest possible language model — a **bigram model** that predicts the next word from the current one only:

```python
import numpy as np

corpus = "the cat sat on the mat the cat ate the rat"
words = corpus.split()
vocab = sorted(set(words))
w2i = {w: i for i, w in enumerate(vocab)}

# Count bigram frequencies
V = len(vocab)
counts = np.zeros((V, V))
for w1, w2 in zip(words[:-1], words[1:]):
    counts[w2i[w1], w2i[w2]] += 1

# Normalise rows to probabilities (add-one smoothing)
probs = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)

current = "the"
for w, p in sorted(zip(vocab, probs[w2i[current]]), key=lambda x: -x[1]):
    print(f"P({w} | {current}) = {p:.3f}")
# P(cat | the) = 0.273
# P(mat | the) = 0.182
# P(rat | the) = 0.182
# P(ate | the) = 0.091
# P(on | the) = 0.091
# P(sat | the) = 0.091
# P(the | the) = 0.091
```

"the" is followed by "cat" twice, "mat" once and "rat" once in the corpus; add-one smoothing gives every word one extra imaginary count, so the 4 real counts and 7 smoothing counts divide into 11ths.

A bigram model has no memory beyond one token. Try the same idea with a two-word window below, trained on a few hundred sentences of this guide — then notice how quickly it loses the thread. A transformer conditions on the *entire* context window (thousands to millions of tokens) and learns representations rather than counting, but the contract is identical: a distribution over what comes next.

<div class="lab" data-lab="ngram"><noscript>Interactive lab: a trigram language model trained on sentences from this guide, with temperature and top-k controls. Enable JavaScript to use it.</noscript></div>

## How Modern LLMs Differ

Modern LLMs share the same fundamental machinery (the **transformer**, Chapter 3, or a hybrid of attention and linear-time layers, Chapter 34) but differ along these axes:

<div class="diagram">
<div class="diagram-title">What Differentiates Modern LLMs</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📐</div>
    <div class="card-title">Architecture</div>
    <div class="card-desc">Dense vs mixture-of-experts, attention variant (GQA, MLA, sparse, hybrid linear), norm placement, context length</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📊</div>
    <div class="card-title">Data</div>
    <div class="card-desc">Token count, quality filtering, deduplication, domain mix, share of synthetic and reasoning data</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚖️</div>
    <div class="card-title">Scale</div>
    <div class="card-desc">Total vs active parameters, training FLOPs, how far past "compute-optimal" the data is pushed</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🎯</div>
    <div class="card-title">Post-training</div>
    <div class="card-desc">SFT mix, preference optimisation, RL with verifiable rewards, safety training, agent environments</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🔤</div>
    <div class="card-title">Tokenizer</div>
    <div class="card-desc">Vocabulary size, multilingual coverage, special tokens for tools, images and thinking</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">🌐</div>
    <div class="card-title">Modality & interface</div>
    <div class="card-desc">Text-only vs natively multimodal; thinking modes; tool-calling and computer-use formats</div>
  </div>
</div>
</div>

## Putting It All Together — A Dozen Lines

The whole pipeline, from a chat message to a generated reply, with Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Explain language modelling in one sentence."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)  # input_ids: [1, T]
out = model.generate(**inputs, max_new_tokens=60)                                     # [1, T + ≤60]
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Every piece of that — the tokenizer and its chat template, the embedding layer, the transformer blocks, the KV cache that `generate` keeps, the sampling settings — is a chapter in this guide.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Next-token prediction on human text is, in the limit, a model of the processes that produced that text — including the reasoning of the people who wrote it. Why might that limit be unreachable in practice, and which of the lifecycle stages above exists precisely because it is?</p>
</div>

## Key Takeaways

- A language model is a learned conditional distribution $P(x_t \mid x_{<t})$; the chain rule turns it into a model of whole sequences, and cross-entropy is how it is trained.
- Text → token IDs `[B, T]` → embeddings `[B, T, d]` → N blocks → logits `[B, T, V]` → sampling. The shapes are the map for the rest of the guide.
- Byte-level BPE dominates tokenization; vocabularies have grown to 128K–262K to spend fewer tokens on multilingual text and code.
- A 2026 model goes through pre-training, mid-training, SFT, RL (preferences *and* verifiable rewards), compression, and test-time compute — and the last of these is now a capability lever, not only a cost.

## Check Yourself

<details class="check">
<summary>Why is a model of next-token conditionals enough to assign a probability to an entire document?</summary>
<p>The chain rule: $P(x_1,\ldots,x_T) = \prod_t P(x_t \mid x_{&lt;t})$. Multiply the model's probability for each actual next token (or add the log-probabilities) and you have the probability of the whole sequence. Perplexity is the exponential of the average negative log-probability per token.</p>
</details>

<details class="check">
<summary>A model has d_model = 4,096 and V = 128,256. How many parameters are in the embedding table, and how many more if the output head is not tied to it?</summary>
<p>128,256 × 4,096 ≈ 525M parameters for the embedding table. An untied output head is another matrix of the same shape, another ≈ 525M — about 13% of an 8B model between them, which is why small models often tie the two.</p>
</details>

<details class="check">
<summary>Why do "SentencePiece" and "BPE" not belong in the same column of a comparison table?</summary>
<p>BPE is an algorithm (merge the most frequent adjacent pair). SentencePiece is a library that implements BPE and the Unigram algorithm on raw text, treating spaces as ordinary symbols. Llama 2 used SentencePiece-BPE; T5 used SentencePiece-Unigram.</p>
</details>

<details class="check">
<summary>In the bigram code, why is P(cat | the) = 0.273 rather than 2/4 = 0.5?</summary>
<p>Add-one smoothing. "the" is followed by four real bigrams (cat ×2, mat, rat), and smoothing adds one imaginary count for each of the 7 vocabulary words, so the denominator is 4 + 7 = 11 and P(cat | the) = (2 + 1)/11 = 0.273.</p>
</details>

<details class="check">
<summary>What changed between the 2022 recipe (pre-train → SFT → RLHF) and the 2026 recipe?</summary>
<p>A mid-training stage (long context, high-quality and reasoning data, annealing) was inserted; RL expanded from learning human preferences to optimising verifiable outcomes on maths, code and agent tasks at large scale; and inference-time compute (thinking budgets, tools, parallel samples) became a deliberate capability knob.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| A Mathematical Theory of Communication | Shannon, 1948 | [doi.org/10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) | Entropy; n-gram approximations of English |
| A Neural Probabilistic Language Model | Bengio et al., 2003 | [jmlr.org/papers/v3/bengio03a](https://www.jmlr.org/papers/v3/bengio03a.html) | First neural LM with learned word vectors |
| Attention Is All You Need | Vaswani et al., 2017 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) | The Transformer |
| Language Models are Few-Shot Learners | Brown et al., 2020 | [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) | GPT-3, in-context learning |
| Training language models to follow instructions with human feedback | Ouyang et al., 2022 | [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) | SFT + RLHF recipe |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | 128K tokenizer, pre-/post-training stages |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek-AI, 2025 | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) | RL with verifiable rewards producing long reasoning |
| Qwen3 Technical Report | Qwen Team, 2025 | [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) | 151,936-token vocabulary; thinking/non-thinking modes |

---

**Next:** [Chapter 2 — Embeddings & Representations](./02_embeddings.md) — how token IDs become the vectors the transformer actually computes with.

[← Back to Table of Contents](./README.md)
