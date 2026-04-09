---
title: "Chapter 7 — Encoder & Encoder-Decoder Models"
---

[← Back to Table of Contents](./README.md)

# Chapter 7 — Encoder & Encoder-Decoder Models

> *"While decoder-only models dominate generation, encoders remain the workhorses of NLU — classification, retrieval, and structured extraction are still best served by bidirectional context."*

## Encoder-Only: BERT

BERT (Devlin et al., 2018) processes the entire input **bidirectionally** — every token can attend to every other token. No causal mask, no generation — just rich contextual representations.

### Masked Language Modelling (MLM)

BERT's pre-training objective: randomly mask 15% of tokens and predict them from context.

<div class="diagram">
<div class="diagram-title">BERT — Masked Language Modelling</div>
<div class="flow-h">
  <div class="flow-node accent">The</div>
  <div class="flow-node purple">[MASK]</div>
  <div class="flow-node accent">sat</div>
  <div class="flow-node accent">on</div>
  <div class="flow-node accent">the</div>
  <div class="flow-node purple">[MASK]</div>
</div>
<div style="text-align: center; padding: 0.5rem 0; color: var(--text-muted);">↓ Bidirectional Transformer Encoder ↓</div>
<div class="flow-h">
  <div class="flow-node accent" style="opacity: 0.5;">—</div>
  <div class="flow-node green">cat</div>
  <div class="flow-node accent" style="opacity: 0.5;">—</div>
  <div class="flow-node accent" style="opacity: 0.5;">—</div>
  <div class="flow-node accent" style="opacity: 0.5;">—</div>
  <div class="flow-node green">mat</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">BERT Tensor Flow</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Input</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">token IDs</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Embeddings</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">768</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">token + position + segment</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Encoder output</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">768</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">contextual representations</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">[CLS] pooled</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">768</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">sentence-level representation</span>
  </div>
</div>
</div>

### BERT Variants

| Model | Params | Layers | Hidden | Heads | Max Length |
|-------|-------:|-------:|-------:|------:|-----------:|
| BERT-base | 110M | 12 | 768 | 12 | 512 |
| BERT-large | 340M | 24 | 1024 | 16 | 512 |
| RoBERTa | 355M | 24 | 1024 | 16 | 512 |
| DeBERTa-v3 | 304M | 24 | 1024 | 16 | 512 |
| ModernBERT | 395M | 28 | 1024 | 16 | 8192 |

**RoBERTa** improved on BERT by removing the NSP objective, training longer, and using larger batches. **DeBERTa** added disentangled attention (separate content and position). **ModernBERT** (2024) brought modern architectural improvements (RoPE, GeLU, Flash Attention) and longer context.

### BERT for Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,  # binary classification
)

inputs = tokenizer("This movie was fantastic!", return_tensors="pt", padding=True)
# inputs["input_ids"].shape: [1, T]

outputs = model(**inputs)
# outputs.logits.shape: [1, 2]  ← one score per class

prediction = torch.argmax(outputs.logits, dim=-1)
print(f"Predicted class: {prediction.item()}")  # 0 or 1
```

## Encoder-Decoder: T5

**T5** (Raffel et al., 2019) frames every NLP task as text-to-text. The encoder processes the input, and the decoder generates the output autoregressively, using cross-attention to read from encoder representations.

<div class="diagram">
<div class="diagram-title">T5 — Text-to-Text Framework</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Encoder (Bidirectional)</div>
    <ul>
      <li>Input: "translate English to French: Hello world"</li>
      <li>Full bidirectional self-attention</li>
      <li>Output: contextual representations [B, T_enc, d]</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Decoder (Causal)</div>
    <ul>
      <li>Output: "Bonjour le monde"</li>
      <li>Causal self-attention + cross-attention to encoder</li>
      <li>Cross-attn: Q from decoder, K/V from encoder</li>
    </ul>
  </div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Encoder-Decoder Tensor Flow</div>
<div class="flow">
  <div class="flow-node accent wide">Encoder Input: [B, T_enc] → Encoder → [B, T_enc, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Decoder Input: [B, T_dec] → Embed → [B, T_dec, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Decoder Layer: Causal Self-Attention [B, T_dec, d] → Cross-Attention(Q=dec, K/V=enc) → FFN</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Output Logits: [B, T_dec, vocab_size]</div>
</div>
</div>

The crucial difference from decoder-only: the cross-attention layer lets the decoder attend to arbitrary encoder positions (not just preceding tokens), giving it direct access to the full input representation.

### T5 for Summarization

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

text = """summarize: The transformer architecture has revolutionized natural
language processing. Originally proposed for machine translation, it has since
been adapted for virtually every NLP task, from text classification to
open-ended generation."""

inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs.input_ids, max_new_tokens=64)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

## BART: Denoising Autoencoder

**BART** (Lewis et al., 2019) is another encoder-decoder model, pre-trained by corrupting input text (token masking, deletion, permutation, infilling) and learning to reconstruct the original. This makes it particularly strong for **summarization** and **text infilling**.

| Corruption | Description |
|-----------|-------------|
| Token Masking | Replace tokens with `[MASK]` |
| Token Deletion | Randomly delete tokens |
| Text Infilling | Replace spans with single `[MASK]` |
| Sentence Permutation | Shuffle sentence order |

## When to Use Which Architecture

<div class="diagram">
<div class="diagram-title">Architecture Selection Guide</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Encoder-Only</div>
    <div class="card-desc">
      <strong>Best for:</strong> Classification, NER, retrieval, semantic similarity<br>
      <strong>Models:</strong> BERT, RoBERTa, DeBERTa<br>
      <strong>Key:</strong> Bidirectional context, [CLS] pooling
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Encoder-Decoder</div>
    <div class="card-desc">
      <strong>Best for:</strong> Translation, summarization, structured generation<br>
      <strong>Models:</strong> T5, BART, mBART<br>
      <strong>Key:</strong> Cross-attention bridges input and output
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Decoder-Only</div>
    <div class="card-desc">
      <strong>Best for:</strong> Open-ended generation, chat, code, reasoning<br>
      <strong>Models:</strong> GPT, LLaMA, Mistral<br>
      <strong>Key:</strong> Scales best, dominant paradigm
    </div>
  </div>
</div>
</div>

| Task | Best Architecture | Why |
|------|:-----------------:|-----|
| Text classification | Encoder | Bidirectional context captures full meaning |
| Named Entity Recognition | Encoder | Per-token classification needs bidirectional |
| Semantic similarity | Encoder | Sentence embeddings from [CLS] or mean pooling |
| Translation | Encoder-Decoder | Input and output are different languages/structures |
| Summarization | Encoder-Decoder or Decoder | Both work; decoder-only now competitive |
| Open-ended chat | Decoder-only | Autoregressive generation is natural |
| Code generation | Decoder-only | Long-range dependencies, large pre-training |
| Reasoning / math | Decoder-only | Chain-of-thought requires sequential generation |

**Trend**: Decoder-only models are increasingly used for tasks that were previously encoder territory, especially at larger scales. But for **embedding**, **classification**, and **retrieval** at smaller scales, encoders remain more efficient and often more accurate.

## What's Next

Modern LLMs don't just process text — they handle images, audio, and video. The next chapter explores **multimodal models** and how different modalities are projected into a shared representation space.

[← Previous: Chapter 6 — Decoder-Only Models](./06_decoder_only_models.md) · **Next: [Chapter 8 — Multimodal Models →](./08_multimodal_models.md)**

---

*Last updated: April 2026*
