---
title: "Appendix B — Tokenization Deep Dive"
description: "What a tokenizer is and is not, byte-level BPE trained step by step (verified), pre-tokenization and why digits split, WordPiece and Unigram, vocabulary-size trade-offs, special tokens and chat formats including thinking and tool tokens, training your own tokenizer, gotchas, and tokenizer-free models."
---

[← Back to Table of Contents](./README.md)

# Appendix B — Tokenization Deep Dive

> **The one-sentence version:** a tokenizer is a fixed, learned dictionary plus merge rules that turns text into integers; it is trained once, separately from the model, and it quietly decides cost, context length and some of what the model can and cannot see.

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
  <div class="flow-node accent wide">Token IDs: [128000, 791, 4320, 374, 220, 2983, 13]</div>
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
print(type(tokenizer).__name__)                # PreTrainedTokenizerFast
print(isinstance(tokenizer, nn.Module))        # False
print(tokenizer.vocab_size, len(tokenizer))    # 128000 128256 — base vocabulary vs incl. special tokens

# Encoding: text → integer IDs  (CPU, microseconds)
ids = tokenizer.encode("The answer is 42.")
print(ids)   # [128000, 791, 4320, 374, 220, 2983, 13] — 128000 is <|begin_of_text|>

# Decoding: integer IDs → text
print(tokenizer.decode(ids, skip_special_tokens=True))   # "The answer is 42."
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

| Library | Used by | Format | Key feature |
|---------|---------|--------|-------------|
| **SentencePiece** | Llama 1/2, T5, Gemma | `.model` | Trains BPE or Unigram directly on raw text (spaces as `▁`) |
| **tiktoken** | GPT-3.5/4 (cl100k), GPT-4o/5 (o200k), gpt-oss | `.tiktoken` | Fast byte-level BPE in Rust |
| **Hugging Face tokenizers** | Most open models incl. Llama 3, Qwen, DeepSeek | `tokenizer.json` | Rust; one file holds normaliser, pre-tokenizer, model and post-processor |

They implement the same algorithms; the differences are file formats, speed and pre-processing details. (Llama 3's tokenizer reuses OpenAI's cl100k vocabulary for its first ~100K entries and adds 28K more — which is why its IDs for common English words match GPT-4's.)

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

Starting corpus: `"low low low low low newer newer newer newest"` — word counts low ×5, newer ×3, newest ×1. (Real byte-level BPE starts from the 256 byte values and keeps a leading space attached to each word; we use plain characters for readability.)

At each step, count every adjacent pair of symbols, weighted by word frequency, and merge the most frequent pair everywhere. The trace below is printed by the trainer that follows it:

```
merge 1: l + o    -> lo     (count 5)   lo w ×5 · n e w e r ×3 · n e w e s t
merge 2: lo + w   -> low    (count 5)   low ×5 · n e w e r ×3 · n e w e s t
merge 3: n + e    -> ne     (count 4)   low ×5 · ne w e r ×3 · ne w e s t
merge 4: ne + w   -> new    (count 4)   low ×5 · new e r ×3 · new e s t
merge 5: new + e  -> newe   (count 4)   low ×5 · newe r ×3 · newe s t
merge 6: newe + r -> newer  (count 3)   low ×5 · newer ×3 · newe s t
merge 7: newe + s -> newes  (count 1)   low ×5 · newer ×3 · newes t
```

Note step 3: once `low` is done, the most frequent pairs are (n, e), (e, w) and (w, e), each appearing 4 times (in "newer" ×3 and "newest" ×1) — more than (e, r) at 3. Ties are broken by a fixed rule (here: first encountered). The vocabulary is the initial symbols plus one new symbol per merge; the ordered **merge list** is what the tokenizer stores.

```python
from collections import Counter

def train_bpe(text, n_merges):
    corpus = {tuple(w): f for w, f in Counter(text.split()).items()}   # word as symbols → frequency
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for symbols, f in corpus.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        (a, b), count = max(pairs.items(), key=lambda kv: kv[1])       # ties: first encountered wins
        merges.append((a, b))
        new = {}
        for symbols, f in corpus.items():                              # apply the merge everywhere
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == (a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(symbols[i]); i += 1
            new[tuple(out)] = f
        corpus = new
    return merges

print(train_bpe("low low low low low newer newer newer newest", 7))
# [('l', 'o'), ('lo', 'w'), ('n', 'e'), ('ne', 'w'), ('new', 'e'), ('newe', 'r'), ('newe', 's')]
```

**Encoding** new text replays the merges in training order: split into bytes, then repeatedly apply the highest-priority merge present. "lower" becomes `low e r` here — `low` is known, but no merge ever joined `e` and `r` before `newe` existed. Try training on your own text:

<div class="lab" data-lab="bpe"><noscript>Interactive lab: train byte-pair encoding on your own text step by step and tokenize new strings with the learned merges. Enable JavaScript to use it.</noscript></div>

Continue until reaching the desired vocabulary size (128K–262K for 2025–26 models).

### Pre-Tokenization: Why Numbers and Spaces Split Oddly

Before BPE runs, a **pre-tokenizer** splits text into chunks that merges may not cross. GPT-4's cl100k and Llama 3 use a regular expression that, among other things, keeps letters with a single leading space, splits **digits into groups of at most three**, and separates punctuation. That is why "The answer is 42." becomes `The · Ġanswer · Ġis · Ġ · 42 · .` — the space before 42 is its own token, because digit groups never absorb a space. Number splitting is deliberate: consistent 1–3-digit chunks make arithmetic patterns learnable, where GPT-2's arbitrary multi-digit tokens did not.

### Tensor Shapes

<div class="tensor-shape">
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Raw text (GPT-2 tokenizer):</span>
    <code>"Hello, world!"</code>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Token IDs:</span>
    <span class="ts-dim seq">[15496, 11, 995, 0]</span>
    <span class="ts-note">seq_len = 4</span>
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
| **Byte-level BPE** | GPT-2 → GPT-5, Llama 3/4, Qwen, DeepSeek, Mistral | Most frequent pair | None needed — every byte is a token | The 2026 default |
| **WordPiece** | BERT, DistilBERT | Pair that most increases likelihood | [UNK] token | `##` prefix for continuations |
| **Unigram LM** | T5, ALBERT, XLNet | Prune tokens that least reduce likelihood | Byte fallback optional | Can sample alternative segmentations (subword regularisation) |

(SentencePiece is a *library* that implements BPE and Unigram on raw text; it is not a separate algorithm.)

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
      <li>Examples: Llama 1/2 (32K), GPT-2 (50K)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Large Vocab (~128K+)</div>
    <ul>
      <li>Larger embedding table</li>
      <li>Shorter sequences (fewer tokens per text)</li>
      <li>Less compute per text (fewer steps)</li>
      <li>Better for multilingual (more scripts)</li>
      <li>Examples: Llama 3 (128K), Qwen3 (152K), o200k (200K), Gemma 3 (262K)</li>
    </ul>
  </div>
</div>
</div>

| Model | Vocabulary | Tokenizer |
|-------|-----------|-------|
| GPT-2 | 50,257 | Byte-level BPE |
| Llama 1/2 | 32,000 | SentencePiece BPE with byte fallback |
| Llama 3 | 128,256 | Byte-level BPE (cl100k + 28K extra) |
| Qwen3 | 151,936 (padded) | Byte-level BPE |
| GPT-4o, GPT-5, gpt-oss | ~200K (o200k) | Byte-level BPE |
| Gemma 3 | 262,144 | SentencePiece |

The vocabulary is a trade-off between the embedding/output matrices (V × d each) and sequence length. For multilingual models, larger vocabularies cut token counts for non-English text substantially — which matters because API prices and context limits are in tokens, and early tokenizers charged some languages several times more tokens than English for the same content.

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
| `<\|eot_id\|>` | Llama 3 | End of turn in chat |
| `<\|start_header_id\|>` | Llama 3 | Chat role marker |
| `<\|im_start\|>`, `<\|im_end\|>` | Qwen, many others (ChatML) | Turn boundaries |
| `<think>`, `</think>` | Qwen3, DeepSeek-R1 and others | Wrap reasoning |
| `<tool_call>`, `<tool_response>` | Qwen and others | Structured tool use |
| `<\|start\|>`, `<\|channel\|>`, `<\|message\|>` | gpt-oss (harmony format) | Roles plus analysis / final channels |
| `<\|fim_prefix\|>`, `<\|fim_suffix\|>`, `<\|fim_middle\|>` | Code models | Fill-in-the-middle |

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

1. **Whitespace sensitivity**: `"Hello"` and `" Hello"` are different tokens; a trailing space in a prompt changes what the model expects next.
2. **Numbers**: `"123456"` is `["123", "456"]` in cl100k-style tokenizers — fine for arithmetic patterns, awkward for character-level tasks.
3. **Counting letters**: "How many r's in strawberry?" is hard partly because the model sees `str · aw · berry`, not letters.
4. **Double BOS**: applying the chat template *and* `add_special_tokens=True` can insert two begin-of-text tokens and subtly degrade output.
5. **Glitch tokens**: tokens that were in the tokenizer's training data but almost never in the model's (the famous "SolidGoldMagikarp") have untrained embeddings and can trigger bizarre behaviour.
6. **Tokenizer–model mismatch**: a fine-tune that adds special tokens must resize the embedding matrix and train the new rows.

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

## Beyond Fixed Tokenizers

A fixed vocabulary is a design decision made before training, never revisited. Research is chipping at it:

- **Byte Latent Transformer** (Meta, 2024): no tokenizer — a small model groups raw bytes into variable-size *patches* based on how predictable the next byte is, spending more compute on hard spans; it matched tokenized models at scale in the paper's experiments.
- **Superword tokens** (2025): letting merges cross spaces ("of the", "in order to") cuts sequence length further for common phrases.
- **Tokenizer transfer**: methods to swap a trained model onto a new tokenizer (e.g., for a new language) without full retraining.

## Check Yourself

<details class="check">
<summary>In the trace above, why is (e, r) not merged at step 3 even though "er" looks like an obvious unit?</summary>
<p>BPE is purely frequency-driven. After "low" is formed, (n, e), (e, w) and (w, e) each occur 4 times (3 in "newer" + 1 in "newest") while (e, r) occurs only 3 times, so a 4-count pair wins.</p>
</details>

<details class="check">
<summary>Why can byte-level BPE never produce an unknown token?</summary>
<p>Its base vocabulary contains all 256 byte values, and any string is a sequence of UTF-8 bytes; in the worst case an unseen word is encoded byte by byte.</p>
</details>

<details class="check">
<summary>Why does "is 42" tokenize as " is", " ", "42" in Llama 3 rather than " is", " 42"?</summary>
<p>The pre-tokenizer regex attaches a leading space to letter runs but not to digit groups, so the space before the digits becomes its own token and "42" is a separate pre-token that merges cannot join to the space.</p>
</details>

<details class="check">
<summary>A model's vocabulary grows from 32K to 256K with d_model = 4,096. How many extra parameters is that in the embedding and an untied output head?</summary>
<p>(256K − 32K) × 4,096 × 2 matrices ≈ 1.84B parameters — significant for a small model, minor for a large one, which is why vocabulary size tends to grow with model scale and multilinguality.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this appendix takes from it |
|---|---|---|---|
| Neural Machine Translation of Rare Words with Subword Units | Sennrich et al., 2015 | [arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) | BPE for NLP |
| Japanese and Korean Voice Search (WordPiece) | Schuster & Nakajima, 2012 | [research.google](https://research.google/pubs/japanese-and-korean-voice-search/) | WordPiece |
| Subword Regularization (Unigram LM) | Kudo, 2018 | [arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959) | Unigram tokenization |
| SentencePiece | Kudo & Richardson, 2018 | [arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226) | Raw-text tokenizer library |
| Language Model Tokenizers Introduce Unfairness Between Languages | Petrov et al., 2023 | [arxiv.org/abs/2305.15425](https://arxiv.org/abs/2305.15425) | Cross-language token cost |
| Byte Latent Transformer | Pagnoni et al. (Meta), 2024 | [arxiv.org/abs/2412.09871](https://arxiv.org/abs/2412.09871) | Tokenizer-free patching |
| tiktoken | OpenAI | [github.com/openai/tiktoken](https://github.com/openai/tiktoken) | cl100k / o200k |
| Hugging Face tokenizers | Hugging Face | [github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Training and serialisation |

---

[← Back to Table of Contents](./README.md) · [← Appendix A — A Brief History](./appendix_a_history.md) · [Appendix C — Model Config Reference →](./appendix_c_model_config.md)
