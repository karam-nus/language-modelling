---
title: "Chapter 7 — Encoder & Encoder-Decoder Models"
description: "BERT-style encoders and masked language modelling, the 2024–25 encoder revival (ModernBERT, EuroBERT, Ettin), T5 and BART encoder-decoders, cross-attention, embedding models, and when each architecture still wins."
related:
  - { id: models, why: "Encoders and encoder-decoders in the wider architecture history." }
---

[← Back to Table of Contents](./README.md)

# Chapter 7 — Encoder & Encoder-Decoder Models

> **The one-sentence version:** remove the causal mask and you get an encoder that reads the whole input at once — worse at generating, better per parameter at classifying, tagging and embedding; bolt a decoder onto it with cross-attention and you get a sequence-to-sequence model.

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
<div class="flow-caption">↓ Bidirectional Transformer Encoder ↓</div>
<div class="flow-h">
  <div class="flow-node accent dim">—</div>
  <div class="flow-node green">cat</div>
  <div class="flow-node accent dim">—</div>
  <div class="flow-node accent dim">—</div>
  <div class="flow-node accent dim">—</div>
  <div class="flow-node green">mat</div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">BERT Tensor Flow</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Input</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">token IDs</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Embeddings</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">768</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">token + position + segment</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Encoder output</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">T</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">768</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">contextual representations</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">[CLS] pooled</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">768</span>
      <span class="ts-bracket">]</span>
    </div>
    <span class="ts-note">sentence-level representation</span>
  </div>
</div>
</div>

### BERT Variants

| Model | Params | Layers | Hidden | Heads | Max Length |
|-------|-------:|-------:|-------:|------:|-----------:|
| BERT-base | 110M | 12 | 768 | 12 | 512 |
| BERT-large | 340M | 24 | 1024 | 16 | 512 |
| RoBERTa | 355M | 24 | 1024 | 16 | 512 |
| DeBERTa-v3-large | 304M + 131M embeddings | 24 | 1024 | 16 | 512 |
| ModernBERT-large (2024) | 395M | 28 | 1024 | 16 | 8,192 |

**RoBERTa** improved on BERT by dropping the next-sentence objective, training longer on more data with bigger batches. **DeBERTa** added disentangled attention (separate content and position vectors) and, in v3, an ELECTRA-style replaced-token-detection objective. **ModernBERT** (December 2024) ported the decoder world's improvements back to encoders — RoPE, GeGLU, pre-norm, alternating local/global attention, unpadded sequences with FlashAttention, a 2T-token training run — and became the default fine-tuning backbone for classification and retrieval.

<div class="callout trend">
<span class="callout-title">2025 trend — the encoder revival</span>
<p>After years of neglect, encoders got modern recipes: ModernBERT, NeoBERT and multilingual EuroBERT and mmBERT. Ai2/JHU's <strong>Ettin</strong> suite trained matched encoder and decoder pairs on identical data and found each wins at its own job — encoders at classification and retrieval, decoders at generation — and that converting one into the other with continued training does not close the gap. Small, bidirectional models are not obsolete; they are specialised.</p>
</div>

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
print(f"Predicted class: {prediction.item()}")  # 0 or 1 — meaningless until you fine-tune
```

The classification head (`[d, num_labels]`) is freshly initialised — transformers warns about it — so this prediction is a coin flip until you fine-tune on labelled data. A few thousand labelled examples and a few minutes on one GPU typically suffice for a binary task.

## Encoders as Embedding Models

The biggest production use of encoders is **embeddings** for search and retrieval-augmented generation ([Chapter 36](./36_rag.md)). A *bi-encoder* turns each text into one vector (mean-pooled or [CLS]) so that similar texts land near each other; a *cross-encoder* reads query and document together and outputs one relevance score — slower, more accurate, used to rerank.

| Model (year) | Backbone | Params | Notes |
|---|---|---:|---|
| all-MiniLM-L6-v2 (2021) | BERT-style encoder | 22M | Still the classic CPU default |
| BGE-M3 (2024) | XLM-RoBERTa | 568M | Dense + sparse + multi-vector in one model |
| EmbeddingGemma (2025) | Gemma 3 made bidirectional | 308M | On-device; Matryoshka dimensions 768 → 128 |
| Qwen3-Embedding (2025) | Qwen3 decoder | 0.6B–8B | Decoder-based; top of MTEB multilingual on release |

Note the last two rows: since 2024 many of the best embedding models start from a *decoder* LLM and switch its attention to bidirectional (or pool its last token), then train contrastively. The line between "encoder" and "decoder" is now about the mask and the training objective, not the weights you start from.

## Encoder-Decoder: T5

**T5** (Raffel et al., 2019; JMLR 2020) frames every NLP task as text-to-text. The encoder processes the input, and the decoder generates the output autoregressively, using cross-attention to read from encoder representations.

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

The crucial difference from decoder-only: the cross-attention layer lets the decoder attend to every encoder position, and the encoder itself read the input bidirectionally. The input is processed once, and its encoder states are reused (cached) for every decoding step — efficient when inputs are long and outputs short, as in summarisation, translation and speech recognition.

Encoder-decoders are still the norm where input and output are different *kinds* of sequence: **Whisper** and most speech recognisers (audio in, text out), NLLB-style translation, and many document-understanding models. In 2025 Google's **T5Gemma** showed a decoder-only checkpoint can be *adapted* into an encoder-decoder — initialising both halves from Gemma 2 — and gain quality at the same inference cost.

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
| Token Masking | Replace tokens with `<mask>` |
| Token Deletion | Randomly delete tokens; the model must find where |
| Text Infilling | Replace whole spans (Poisson length, λ = 3) with a single `<mask>` |
| Sentence Permutation | Shuffle sentence order |
| Document Rotation | Start the document at a random token |

Text infilling worked best and became the template for T5's span corruption and, later, the fill-in-the-middle (FIM) objective used to train code models.

## When to Use Which Architecture

<div class="diagram">
<div class="diagram-title">Architecture Selection Guide</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Encoder-Only</div>
    <div class="card-desc">
      <strong>Best for:</strong> Classification, NER, retrieval, semantic similarity<br>
      <strong>Models:</strong> ModernBERT, DeBERTa-v3, EuroBERT, embedding models<br>
      <strong>Key:</strong> Bidirectional context, [CLS] pooling
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Encoder-Decoder</div>
    <div class="card-desc">
      <strong>Best for:</strong> Translation, summarization, structured generation<br>
      <strong>Models:</strong> T5 / T5Gemma, BART, Whisper, NLLB<br>
      <strong>Key:</strong> Cross-attention bridges input and output
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Decoder-Only</div>
    <div class="card-desc">
      <strong>Best for:</strong> Open-ended generation, chat, code, reasoning<br>
      <strong>Models:</strong> GPT, Llama, Qwen, DeepSeek, Gemma<br>
      <strong>Key:</strong> Scales best, dominant paradigm
    </div>
  </div>
</div>
</div>

| Task | Best Architecture | Why |
|------|:-----------------:|-----|
| Text classification | Encoder | Bidirectional context captures full meaning |
| Named Entity Recognition | Encoder | Per-token classification needs bidirectional |
| Semantic similarity / retrieval | Encoder (or bidirectional-ised decoder) | Contrastively trained sentence embeddings |
| Reranking | Cross-encoder | Reads query and document jointly |
| Speech recognition | Encoder-Decoder | Audio frames in, text tokens out |
| Translation | Encoder-Decoder or Decoder | Dedicated MT still often seq2seq; LLMs now competitive |
| Summarization | Encoder-Decoder or Decoder | Both work; decoder-only now competitive |
| Open-ended chat | Decoder-only | Autoregressive generation is natural |
| Code generation | Decoder-only | Long-range dependencies, large pre-training |
| Reasoning / math | Decoder-only | Chain-of-thought requires sequential generation |

**Trend**: decoder-only models absorb more tasks every year, especially at large scale and with prompting. But for **embedding**, **classification** and **retrieval** under a latency or cost budget, encoders remain more efficient and often more accurate per parameter — a 150M ModernBERT fine-tune routinely beats prompting a 70B model on a narrow labelling task, at a thousandth of the cost.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Masked-token prediction gives an encoder a learning signal on only ~15% of positions per sequence, while next-token prediction trains on all of them. Yet encoders win at classification. What does bidirectional context buy that is worth a 6× less dense training signal — and how do diffusion language models (Chapter 38) try to get both?</p>
</div>

## Key Takeaways

- Encoders drop the causal mask and train with masked-token (or replaced-token) objectives; they produce contextual representations, not text.
- ModernBERT, EuroBERT and the Ettin study revived encoders in 2024–25: modern blocks, long context and big training runs still give them the edge on classification and retrieval.
- Embedding models are the encoder's biggest job; many of the best now start from a decoder LLM made bidirectional.
- Encoder-decoders read the input once and reuse it through cross-attention — the natural fit for speech recognition and other "different input and output" tasks.
- Choose by task and budget: generation → decoder; tagging, scoring and embedding at scale → encoder; long-input/short-output transduction → encoder-decoder.

## Check Yourself

<details class="check">
<summary>Why can't BERT generate text the way GPT does?</summary>
<p>It was never trained to predict a token from only its left context: every position sees both directions, and the objective reconstructs masked tokens in place. There is no causal factorisation to sample from left to right (iterative mask-filling is possible but slow and weak).</p>
</details>

<details class="check">
<summary>In an encoder-decoder, which tensors can be cached across decoding steps, and which grow?</summary>
<p>The encoder output — and the cross-attention K and V computed from it — are fixed after one pass and reused at every step. Only the decoder's self-attention KV cache grows with each generated token.</p>
</details>

<details class="check">
<summary>A bi-encoder and a cross-encoder both score query–document relevance. Why use both in one search system?</summary>
<p>Bi-encoder document vectors can be precomputed and searched with approximate nearest neighbours over millions of documents; a cross-encoder must run once per (query, document) pair. So the bi-encoder retrieves the top ~100 candidates and the cross-encoder reranks them accurately.</p>
</details>

<details class="check">
<summary>How do decoder-based embedding models such as Qwen3-Embedding turn a causal LM into an embedder?</summary>
<p>They pool a sequence representation (often the final token's hidden state, sometimes mean pooling with the causal mask removed) and fine-tune contrastively on query–document pairs, so that relevant pairs have high cosine similarity and in-batch negatives low.</p>
</details>

<details class="check">
<summary>Which BART corruption became the basis of fill-in-the-middle training for code models?</summary>
<p>Text infilling: a span is removed and the model regenerates it. FIM reorders a document into prefix, suffix, then middle, so a left-to-right decoder learns to fill a gap given text on both sides.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al., 2018 | [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) | MLM, encoder pre-training |
| RoBERTa | Liu et al., 2019 | [arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) | Better BERT training |
| DeBERTaV3 | He et al., 2021 | [arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543) | Disentangled attention, replaced-token detection |
| Exploring the Limits of Transfer Learning (T5) | Raffel et al., 2019 | [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683) | Text-to-text, span corruption |
| BART | Lewis et al., 2019 | [arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461) | Denoising objectives |
| Smarter, Better, Faster, Longer (ModernBERT) | Warner et al., 2024 | [arxiv.org/abs/2412.13663](https://arxiv.org/abs/2412.13663) | Modern encoder recipe |
| Seq vs Seq: An Open Suite of Paired Encoders and Decoders (Ettin) | Weller et al., 2025 | [arxiv.org/abs/2507.11412](https://arxiv.org/abs/2507.11412) | Encoder vs decoder on equal data |
| Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford et al., 2022 | [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356) | Encoder-decoder for speech |
| Qwen3 Embedding | Zhang et al. (Qwen), 2025 | [arxiv.org/abs/2506.05176](https://arxiv.org/abs/2506.05176) | Decoder-based embedding models |
| Efficient Training of Language Models to Fill in the Middle | Bavarian et al., 2022 | [arxiv.org/abs/2207.14255](https://arxiv.org/abs/2207.14255) | FIM |

---

**Next:** [Chapter 8 — Multimodal Models](./08_multimodal_models.md) — how images, audio and video get turned into tokens a language model can read.

[← Back to Table of Contents](./README.md)
