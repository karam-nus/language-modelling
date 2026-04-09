---
title: "Appendix B — Tokenization Deep Dive"
---

[← Back to Table of Contents](./README.md)

# Appendix B — Tokenization Deep Dive

> *"The tokenizer is the first and last thing that touches your text. Get it wrong, and no amount of model scaling will help."*

## What Exactly Is a Tokenizer?

Before diving into algorithms, it is worth being precise about what a tokenizer *is* — because it is often confused with the model itself.

**A tokenizer is not a neural network.** It is a stateless, deterministic text-preprocessing component that converts raw strings into integer IDs and back. At inference time it runs on the CPU, takes microseconds, and involves no learnable parameters, no matrix multiplications, and no gradients.

<div class="diagram">
<div class="diagram-title">Tokenizer vs. Model — Two Completely Separate Objects</div>
<div class="flow">
  <div class="flow-node wide">Raw text: "The answer is 42."</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Tokenizer (CPU, deterministic)<br><small>vocabulary lookup + merge rules → integer IDs</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Token IDs: [791, 4320, 374, 220, 2983, 13]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Language Model (GPU, learned)<br><small>Transformer forward pass → logit distributions</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Next-token logits: [B, T, vocab_size]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Tokenizer decode (CPU)<br><small>integer IDs → string</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node wide">Generated text: " The answer is 42."</div>
</div>
</div>

### What a Tokenizer Consists Of

A tokenizer object contains exactly **two things**:

1. **Vocabulary** (`vocab`): a dictionary mapping subword strings → integer IDs.  
   e.g., `{"the": 1, "Ġthe": 791, "Ġanswer": 4320, ...}` (size: 32K–256K entries)

2. **Merge rules** (for BPE): an ordered list of character-pair merges learned during training.  
   e.g., `[("Ġ", "t") → "Ġt", ("Ġt", "he") → "Ġthe", ...]`

There are **no weights, no layers, no activations**. The tokenizer is typically serialized as a single JSON or `.model` file (a few MB). In HuggingFace it's a `PreTrainedTokenizer` or `PreTrainedTokenizerFast` object — not a `nn.Module`.

```python
from transformers import AutoTokenizer
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Tokenizer is NOT an nn.Module — it has no parameters()
print(type(tokenizer))                         # PreTrainedTokenizerFast
print(isinstance(tokenizer, nn.Module))        # False
print(f"Vocab size: {tokenizer.vocab_size}")   # 128256

# What it contains:
print(f"Vocab sample: {list(tokenizer.vocab.items())[:3]}")
# [('!', 0), ('"', 1), ('#', 2)]

# Encoding: text → integer IDs  (CPU, ~microseconds)
ids = tokenizer.encode("The answer is 42.")
print(ids)   # [791, 4320, 374, 220, 2983, 13]

# Decoding: integer IDs → text  (deterministic round-trip)
print(tokenizer.decode(ids))   # "The answer is 42."
```

### How Does the Model Use the Token IDs?

The token IDs are the *input* to the language model's embedding layer — an `nn.Embedding` table of shape `[vocab_size, d_model]`. The model looks up each ID as a row index:

```python
import torch.nn as nn

# Inside the Transformer
embedding = nn.Embedding(vocab_size=128256, embedding_dim=4096)

# token_ids: [B, T]  (integers from tokenizer)
# embeddings: [B, T, 4096]  (first layer of the neural network)
embeddings = embedding(token_ids)
```

So the "connection" between tokenizer and model is simply a shared vocabulary size: the tokenizer produces integers in `[0, vocab_size)`, and the model's embedding table has exactly `vocab_size` rows.

### SentencePiece, tiktoken, and Tokenizer Libraries

Different models use different tokenizer *implementations* (though the concepts are the same):

| Library | Used By | Format | Key feature |
|---------|---------|--------|-------------|
| **SentencePiece** | LLaMA-1/2, T5, Gemma | `.model` binary | Treats text as raw bytes, language-agnostic |
| **tiktoken** | GPT-3/4, LLaMA-3 | `.tiktoken` | Rust-based, very fast |
| **HuggingFace tokenizers** | BERT, RoBERTa, most HF models | `tokenizer.json` | Rust backend, universal format |

All three implement the same BPE or Unigram algorithm — just in different languages/formats.

---

## Why Tokenization Matters

The tokenizer determines:
- **Vocabulary size** → embedding table size → model memory
- **Sequence length** → number of tokens per input → compute cost
- **What the model "sees"** → subword units shape what patterns the model can learn

## Tokenization Strategies

<div class="diagram">
<div class="diagram-title">Tokenization Approaches</div>
<div class="layer-stack">
  <div class="layer red">Character-level — ['H','e','l','l','o'] — tiny vocab, very long sequences</div>
  <div class="layer orange">Byte-level — [72,101,108,108,111] — 256 vocab, no UNK, long sequences</div>
  <div class="layer green">BPE (Byte-Pair Encoding) — ['Hello'] or ['Hel','lo'] — balanced, most popular</div>
  <div class="layer accent">WordPiece — ['Hello'] or ['He','##llo'] — BERT-style, ## prefix for continuations</div>
  <div class="layer purple">Unigram — probabilistic, keeps the most likely subword segmentation</div>
  <div class="layer cyan">Word-level — ['Hello'] — simple but can't handle unseen words</div>
</div>
</div>

## BPE: Step by Step

Byte-Pair Encoding (Sennrich et al., 2016) is the most widely used algorithm. It iteratively merges the most frequent pair of adjacent tokens:

### Training BPE

Starting corpus: `"low low low low low newer newer newer newest"`

**Step 0 — Initialize with characters:**
```
Vocabulary: {l, o, w, n, e, r, s, t, _, <space>}
Tokens: l o w · l o w · l o w · l o w · l o w · n e w e r · n e w e r · n e w e r · n e w e s t
```

**Step 1 — Most frequent pair: (l, o) → merge into `lo`:**
```
lo w · lo w · lo w · lo w · lo w · n e w e r · n e w e r · n e w e r · n e w e s t
```

**Step 2 — Most frequent pair: (lo, w) → merge into `low`:**
```
low · low · low · low · low · n e w e r · n e w e r · n e w e r · n e w e s t
```

**Step 3 — Most frequent pair: (e, r) → merge into `er`:**
```
low · low · low · low · low · n e w er · n e w er · n e w er · n e w e s t
```

**Step 4 — Most frequent pair: (n, e) → merge into `ne`:**
```
low · low · low · low · low · ne w er · ne w er · ne w er · ne w e s t
```

**Step 5 — Most frequent pair: (ne, w) → merge into `new`:**
```
low · low · low · low · low · new er · new er · new er · new e s t
```

**Step 6 — Most frequent pair: (new, er) → merge into `newer`:**
```
low · low · low · low · low · newer · newer · newer · new e s t
```

Continue until reaching the desired vocabulary size (typically 32K–256K).

### Tensor Shapes

<div class="tensor-shape">
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Raw text:</span>
    <code>"Hello, world!"</code>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Token IDs:</span>
    <span class="ts-dim seq">[15496, 11, 995, 0]</span>
    <span style="color: var(--text-secondary); margin-left: 0.5rem;">seq_len = 4</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">One-hot (conceptual):</span>
    <span class="ts-dim seq">4</span> × <span class="ts-dim vocab">vocab_size</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After embedding lookup:</span>
    <span class="ts-dim seq">4</span> × <span class="ts-dim feature">d_model</span>
  </div>
</div>
</div>

## Comparison of Algorithms

| Algorithm | Used By | Merge Strategy | UNK Handling | Notes |
|-----------|---------|---------------|-------------|-------|
| **BPE** | GPT-2/3/4, LLaMA, Mistral | Most frequent pair | Byte fallback (no UNK) | Most popular |
| **WordPiece** | BERT, DistilBERT | Maximizes likelihood | [UNK] token | `##` prefix for continuations |
| **Unigram** | T5, ALBERT, XLNet | Remove tokens that least reduce likelihood | Probabilistic segmentation | Multiple valid segmentations |
| **SentencePiece** | LLaMA, T5, Gemma | BPE or Unigram on raw bytes | Language-agnostic | Treats input as raw byte stream |

## Vocabulary Size Tradeoffs

<div class="diagram">
<div class="diagram-title">Vocabulary Size: Smaller vs Larger</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Small Vocab (~32K)</div>
    <ul>
      <li>Smaller embedding table (fewer parameters)</li>
      <li>Longer sequences (more tokens per text)</li>
      <li>More compute per text (more steps)</li>
      <li>Better subword generalization</li>
      <li>Example: LLaMA-1 (32K), GPT-2 (50K)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Large Vocab (~128K+)</div>
    <ul>
      <li>Larger embedding table</li>
      <li>Shorter sequences (fewer tokens per text)</li>
      <li>Less compute per text (fewer steps)</li>
      <li>Better for multilingual (more scripts)</li>
      <li>Example: LLaMA-3 (128K), Gemini (256K)</li>
    </ul>
  </div>
</div>
</div>

| Model | Vocab Size | Tokens for "Hello, world!" | Notes |
|-------|-----------|---------------------------|-------|
| GPT-2 | 50,257 | 4 | BPE |
| LLaMA-1 | 32,000 | 5 | SentencePiece BPE |
| LLaMA-3 | 128,256 | 4 | tiktoken BPE, multilingual |
| Gemma | 256,000 | 3 | SentencePiece, massive vocab |

## Special Tokens

Special tokens serve structural roles that the model learns during training:

| Token | Used By | Purpose |
|-------|---------|---------|
| `[CLS]` | BERT | Classification token (first position) |
| `[SEP]` | BERT | Separates sentence pairs |
| `[MASK]` | BERT | Masked token for MLM |
| `[PAD]` | Most models | Padding for batching |
| `<s>`, `</s>` | LLaMA, Mistral | Begin/end of sequence |
| `<\|endoftext\|>` | GPT-2/3 | End of document |
| `<\|begin_of_text\|>` | LLaMA-3 | Start of text |
| `<\|eot_id\|>` | LLaMA-3 | End of turn in chat |
| `<\|start_header_id\|>` | LLaMA-3 | Chat role marker |

### Chat Templates

Modern models use structured templates to distinguish system, user, and assistant messages:

```
# LLaMA-3 chat format
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is attention?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Attention is a mechanism that allows...
```

```python
# Use tokenizer.apply_chat_template for correct formatting
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is attention?"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(formatted)
```

## Training a BPE Tokenizer from Scratch

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Create a BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# Train on your data
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
    show_progress=True,
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Use it
encoded = tokenizer.encode("Hello, world!")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

# Save/load
tokenizer.save("my_tokenizer.json")
```

## Tokenization Gotchas

1. **Whitespace sensitivity**: `"Hello"` vs `" Hello"` produce different tokens
2. **Numbers**: `"123456"` may tokenize as `["123", "456"]` — the model sees two separate numbers
3. **Code**: indentation and special characters can produce unexpected tokenizations
4. **Multilingual**: non-Latin scripts may require many tokens per word with English-centric tokenizers
5. **Trailing whitespace**: be careful with `add_special_tokens` and `add_generation_prompt`

```python
# Always inspect your tokenizer's behavior
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

text = "The answer is 42."
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
print(f"Text: {text!r}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {ids}")
print(f"Num tokens: {len(ids)}")

# Decode back
print(f"Decoded: {tokenizer.decode(ids)!r}")
```

---

[← Back to Table of Contents](./README.md) · [← Appendix A — A Brief History](./appendix_a_history.md)

---

*Last updated: April 2026*
