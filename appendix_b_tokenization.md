[← Back to Table of Contents](./README.md)

# Appendix B — Tokenization Deep Dive

> *"The tokenizer is the first and last thing that touches your text. Get it wrong, and no amount of model scaling will help."*

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
