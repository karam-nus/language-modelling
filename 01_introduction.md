[← Back to Table of Contents](./README.md)

# Chapter 1 — Introduction to Language Modelling

> *"The problem of predicting the next word is, in a sense, the problem of understanding language."*
> — Yoshua Bengio

## The Core Idea

A **language model** is a system that assigns probabilities to sequences of words. At its heart, the task is deceptively simple: given some text, predict what comes next. This single objective — **next-token prediction** — turns out to be powerful enough to produce systems that can write code, answer questions, translate languages, and reason about the world.

Every modern large language model (LLM), from GPT-4 to LLaMA to Gemini, is trained on some variation of this principle. The model reads a sequence of tokens and learns to predict the probability distribution over the next token:

$$P(x_{t+1} \mid x_1, x_2, \ldots, x_t)$$

<div class="diagram">
<div class="diagram-title">The Language Modelling Objective</div>
<div class="flow-h">
  <div class="flow-node accent">The cat sat</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">on (83%)</div>
</div>
</div>

The model doesn't "understand" in a human sense — it builds statistical representations of language so rich that understanding-like behavior emerges from the training process.

## From Text to Tensors — The Full Pipeline

Before a model can process text, the text must become numbers. Before those numbers mean anything, they must be transformed into rich vector representations. The pipeline from raw text to a model's prediction involves several stages, each covered in depth in later chapters:

<div class="diagram">
<div class="diagram-title">The Language Model Pipeline</div>
<div class="flow">
  <div class="flow-node wide">📝 Raw Text <small>"The cat sat on"</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">🔤 Tokenizer <small>text → token IDs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">📊 Embedding Layer <small>token IDs → dense vectors</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🧠 Transformer Blocks <small>self-attention + feed-forward × N</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">📈 Output Head <small>hidden states → vocabulary logits</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🎯 Sampling <small>logits → next token</small></div>
</div>
</div>

And here are the tensor shapes at each stage — a pattern you'll see throughout this guide:

<div class="diagram">
<div class="diagram-title">Tensor Shapes Through the Pipeline</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 100px;">Token IDs</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 100px;">Embeddings</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 100px;">After Transformer</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim feature">d_model</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 100px;">Logits</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-sep">,</span>
      <span class="ts-dim vocab">V</span>
      <span class="ts-bracket">]</span>
    </div>
  </div>
</div>
</div>

Where **B** is batch size, **T** is sequence length, **d_model** is the hidden dimension (e.g., 4096), and **V** is the vocabulary size (e.g., 32000 for LLaMA).

## A Brief History

Language modelling has evolved through several paradigm shifts over six decades. For the full historical deep dive, see [Appendix A](./appendix_a_history.md). Here are the key milestones:

<div class="diagram">
<div class="diagram-title">Key Milestones in Language Modelling</div>
<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">1948–1990s</div>
    <div class="timeline-title">Statistical Era</div>
    <div class="timeline-desc">Shannon's information theory → N-gram models → smoothing techniques</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2003</div>
    <div class="timeline-title">Neural Language Models</div>
    <div class="timeline-desc">Bengio et al. — first neural network that learns word representations and predicts next words</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2013</div>
    <div class="timeline-title">Word2Vec</div>
    <div class="timeline-desc">Mikolov et al. — efficient word embeddings, "king - man + woman = queen"</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2017</div>
    <div class="timeline-title">Transformers</div>
    <div class="timeline-desc">Vaswani et al. — "Attention Is All You Need" — the architecture that changed everything</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2018–2019</div>
    <div class="timeline-title">Pre-training Revolution</div>
    <div class="timeline-desc">GPT, BERT, GPT-2 — large-scale pre-training on internet text</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2020–2023</div>
    <div class="timeline-title">The Scaling Era</div>
    <div class="timeline-desc">GPT-3, Chinchilla, LLaMA, GPT-4 — scaling laws, emergence, open-source explosion</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2024–Present</div>
    <div class="timeline-title">Reasoning & Efficiency</div>
    <div class="timeline-desc">o1, DeepSeek-R1, Mamba, MoE — test-time compute, efficient architectures, open weights</div>
  </div>
</div>
</div>

## Tokenization Essentials

Before a model sees any text, a **tokenizer** breaks it into discrete units called **tokens**. Modern LLMs use **subword tokenization** — a middle ground between character-level and word-level that handles rare words and multiple languages efficiently.

For the complete tokenization deep dive (BPE algorithm walkthrough, training tokenizers from scratch), see [Appendix B](./appendix_b_tokenization.md). Here's what you need to know:

| Method | Used By | Key Idea |
|--------|---------|----------|
| **BPE** (Byte Pair Encoding) | GPT, LLaMA, Mistral | Iteratively merge most frequent byte pairs |
| **WordPiece** | BERT, DistilBERT | Similar to BPE but uses likelihood-based merges |
| **Unigram** | T5, ALBERT | Start with large vocab, prune to target size |
| **SentencePiece** | LLaMA, T5 | Language-agnostic, treats input as raw bytes |

The tokenizer defines the vocabulary size **V**, which determines the final projection layer. A typical modern LLM uses V = 32,000–128,000 tokens.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

text = "Language models predict the next token."
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
print(f"Vocab size: {tokenizer.vocab_size}")
# Text: Language models predict the next token.
# Token IDs: [14658, 4211, 7963, 279, 1828, 4037, 13]
# Tokens: ['Language', ' models', ' predict', ' the', ' next', ' token', '.']
# Vocab size: 128256
```

## The Simplest Language Model

To build intuition, here is the simplest possible language model — a **bigram model** that predicts the next token based only on the current one:

```python
import numpy as np

# A tiny corpus
corpus = "the cat sat on the mat the cat ate the rat"
words = corpus.split()
vocab = sorted(set(words))
w2i = {w: i for i, w in enumerate(vocab)}

# Count bigram frequencies
V = len(vocab)
counts = np.zeros((V, V))
for w1, w2 in zip(words[:-1], words[1:]):
    counts[w2i[w1], w2i[w2]] += 1

# Normalize to probabilities (add-1 smoothing)
probs = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)

# Predict next word
current = "the"
next_probs = probs[w2i[current]]
for w, p in sorted(zip(vocab, next_probs), key=lambda x: -x[1]):
    print(f"  P({w} | {current}) = {p:.3f}")
# P(cat | the) = 0.273
# P(mat | the) = 0.182
# P(rat | the) = 0.182
# ...
```

A bigram model is limited — it has no memory beyond one token. A **transformer-based** language model considers the entire context window (thousands or millions of tokens), learns rich representations, and can capture long-range dependencies. But the core idea is the same: assign probabilities to what comes next.

## How Modern LLMs Differ

Modern LLMs like GPT-4, LLaMA 3, and Gemini all share the same fundamental architecture (the **transformer**, Chapter 3) but differ in:

<div class="diagram">
<div class="diagram-title">What Differentiates Modern LLMs</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📐</div>
    <div class="card-title">Architecture</div>
    <div class="card-desc">Attention type, norm placement, FFN variant, context length</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📊</div>
    <div class="card-title">Training Data</div>
    <div class="card-desc">Corpus size, quality filters, domain mix, deduplication</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚖️</div>
    <div class="card-title">Scale</div>
    <div class="card-desc">Parameter count, compute budget, training tokens</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🎯</div>
    <div class="card-title">Alignment</div>
    <div class="card-desc">RLHF, DPO, safety training, instruction tuning</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🔤</div>
    <div class="card-title">Tokenizer</div>
    <div class="card-desc">Vocabulary size, BPE variant, multilingual support</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">🌐</div>
    <div class="card-title">Modality</div>
    <div class="card-desc">Text-only vs multimodal (vision, audio, code)</div>
  </div>
</div>
</div>

## Putting It All Together — 10 Lines

Here's the entire pipeline — from text to generated response — in 10 lines of Python:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Explain language modelling in one sentence:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # [1, T]
outputs = model.generate(**inputs, max_new_tokens=50)               # [1, T + 50]
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Every component in those 10 lines — the tokenizer, the embedding layer, the transformer blocks, the sampling strategy — is a chapter in this guide.

## What's Next

With the big picture in place, the next chapter dives into **embeddings** — how token IDs become the rich vector representations that transformers actually process.

**Next: [Chapter 2 — Embeddings & Representations →](./02_embeddings.md)**

---

*Last updated: April 2026*
