[← Back to Table of Contents](./README.md)

# Appendix A — A Brief History of Language Modelling

> *"Those who cannot remember the past are condemned to reimplement it."*

## Timeline

<div class="diagram">
<div class="diagram-title">The Arc of Language Modelling</div>
<div class="timeline">
  <div class="timeline-item green">
    <div class="timeline-date">1948</div>
    <div class="timeline-desc">Claude Shannon — "A Mathematical Theory of Communication". Entropy of English, N-gram prediction, Markov chains.</div>
  </div>
  <div class="timeline-item green">
    <div class="timeline-date">1980s</div>
    <div class="timeline-desc">N-gram language models with smoothing (Katz, Kneser-Ney). Dominated NLP for two decades.</div>
  </div>
  <div class="timeline-item accent">
    <div class="timeline-date">2003</div>
    <div class="timeline-desc">Bengio et al. — "A Neural Probabilistic Language Model". First neural LM: learn word embeddings + predict next word with a feedforward network.</div>
  </div>
  <div class="timeline-item accent">
    <div class="timeline-date">2011</div>
    <div class="timeline-desc">RNN language models (Mikolov). Recurrent networks for variable-length sequences.</div>
  </div>
  <div class="timeline-item accent">
    <div class="timeline-date">2013</div>
    <div class="timeline-desc">Word2Vec (Mikolov et al.). Efficient word embeddings via skip-gram and CBOW. "King − Man + Woman = Queen".</div>
  </div>
  <div class="timeline-item purple">
    <div class="timeline-date">2014</div>
    <div class="timeline-desc">Seq2Seq (Sutskever et al.). Encoder-decoder RNNs for machine translation.</div>
  </div>
  <div class="timeline-item purple">
    <div class="timeline-date">2015</div>
    <div class="timeline-desc">Attention mechanism (Bahdanau et al.). Allow decoder to "attend" to relevant encoder states.</div>
  </div>
  <div class="timeline-item orange">
    <div class="timeline-date">2017</div>
    <div class="timeline-desc"><strong>Transformer</strong> (Vaswani et al.) — "Attention Is All You Need". Self-attention replaces recurrence entirely. Parallelizable, scalable.</div>
  </div>
  <div class="timeline-item orange">
    <div class="timeline-date">2018</div>
    <div class="timeline-desc"><strong>GPT-1</strong> (Radford et al.) — decoder-only transformer, 117M params, pre-train then fine-tune. <strong>BERT</strong> (Devlin et al.) — bidirectional encoder, MLM pre-training. Revolutionized NLU.</div>
  </div>
  <div class="timeline-item orange">
    <div class="timeline-date">2019</div>
    <div class="timeline-desc"><strong>GPT-2</strong> (1.5B) — "Language Models are Unsupervised Multitask Learners". Zero-shot task performance. T5 — text-to-text framework.</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">2020</div>
    <div class="timeline-desc"><strong>GPT-3</strong> (175B) — in-context learning emerges at scale. Few-shot prompting. Kaplan scaling laws.</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">2022</div>
    <div class="timeline-desc"><strong>ChatGPT</strong> — RLHF alignment makes LLMs usable as assistants. Chinchilla scaling laws. InstructGPT. PaLM (540B).</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">2023</div>
    <div class="timeline-desc"><strong>GPT-4</strong>, <strong>LLaMA</strong> (open weights revolution), <strong>Mistral-7B</strong>, Mamba (SSMs), Flash Attention 2.</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">2024</div>
    <div class="timeline-desc"><strong>LLaMA-3</strong>, <strong>DeepSeek-V2/V3</strong>, <strong>Qwen-2.5</strong>, o1 (reasoning), Claude 3.5, Gemini 1.5 (2M context). MoE at scale.</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">2025</div>
    <div class="timeline-desc"><strong>DeepSeek-R1</strong> (open reasoning), reasoning models proliferate, agents become practical, efficiency frontier pushed (small models, big data).</div>
  </div>
</div>
</div>

## Key Papers

| Year | Paper | Authors | Contribution |
|------|-------|---------|-------------|
| 1948 | A Mathematical Theory of Communication | Shannon | Information theory, entropy of language |
| 2003 | A Neural Probabilistic Language Model | Bengio et al. | First neural language model |
| 2013 | Efficient Estimation of Word Representations | Mikolov et al. | Word2Vec — skip-gram & CBOW |
| 2014 | Sequence to Sequence Learning | Sutskever et al. | Encoder-decoder for translation |
| 2015 | Neural Machine Translation by Jointly Learning to Align and Translate | Bahdanau et al. | Attention mechanism |
| 2017 | Attention Is All You Need | Vaswani et al. | The Transformer |
| 2018 | Improving Language Understanding by Generative Pre-Training | Radford et al. | GPT-1 |
| 2018 | BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | BERT, MLM |
| 2019 | Language Models are Unsupervised Multitask Learners | Radford et al. | GPT-2 |
| 2020 | Language Models are Few-Shot Learners | Brown et al. | GPT-3, in-context learning |
| 2020 | Scaling Laws for Neural Language Models | Kaplan et al. | Power-law scaling |
| 2021 | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | Parameter-efficient fine-tuning |
| 2022 | Training Compute-Optimal Large Language Models | Hoffmann et al. | Chinchilla scaling laws |
| 2022 | Training language models to follow instructions (InstructGPT) | Ouyang et al. | RLHF for alignment |
| 2022 | Chain-of-Thought Prompting | Wei et al. | Step-by-step reasoning |
| 2023 | LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | Open weights |
| 2023 | FlashAttention-2 | Dao | IO-aware exact attention |
| 2023 | Direct Preference Optimization | Rafailov et al. | DPO — simpler alternative to RLHF |
| 2023 | Mamba: Linear-Time Sequence Modeling | Gu & Dao | Selective state spaces |
| 2024 | The Llama 3 Herd of Models | Meta AI | LLaMA-3 family, 8B–405B |
| 2024 | DeepSeek-V3 Technical Report | DeepSeek | 671B MoE, FP8 training |
| 2025 | DeepSeek-R1 | DeepSeek | RL-trained reasoning, open weights |

## The N-gram Era

Before neural networks, language models counted word sequences:

$$
P(w_n | w_1, \ldots, w_{n-1}) \approx P(w_n | w_{n-k}, \ldots, w_{n-1})
$$

```python
# Simple bigram model in NumPy
import numpy as np
from collections import Counter

def train_bigram(corpus):
    """Train a bigram language model from a list of sentences."""
    bigram_counts = Counter()
    unigram_counts = Counter()
    
    for sentence in corpus:
        tokens = ['<s>'] + sentence.split() + ['</s>']
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i+1])] += 1
            unigram_counts[tokens[i]] += 1
    
    # Convert to probabilities with add-1 smoothing
    vocab = set(unigram_counts.keys())
    V = len(vocab)
    
    def probability(word, context):
        return (bigram_counts[(context, word)] + 1) / (unigram_counts[context] + V)
    
    return probability, vocab

def perplexity(prob_fn, vocab, test_sentence):
    tokens = ['<s>'] + test_sentence.split() + ['</s>']
    log_prob = sum(np.log2(prob_fn(tokens[i+1], tokens[i])) for i in range(len(tokens)-1))
    return 2 ** (-log_prob / (len(tokens) - 1))
```

N-grams were the state of the art for speech recognition and machine translation until ~2014. Their fundamental limitation: they can't generalize beyond seen N-grams and require exponential storage for long contexts.

## The Neural Revolution

Bengio's 2003 paper introduced two ideas that still underpin modern LLMs:
1. **Learned embeddings**: represent words as dense vectors (not one-hot)
2. **Shared parameters**: a neural network that generalizes across contexts

This evolved through RNNs → LSTMs → Attention → Transformers, each solving a limitation of the previous:

| Architecture | Key Limitation Solved | New Limitation |
|-------------|---------------------|---------------|
| Feedforward (2003) | Can't handle variable length | Fixed context window |
| RNN (2011) | Variable-length sequences | Vanishing gradients, sequential |
| LSTM (2014) | Long-range dependencies | Still sequential, slow to train |
| Seq2Seq + Attention (2015) | Encoder bottleneck | Still recurrent |
| **Transformer (2017)** | Fully parallel training | Quadratic attention cost |

See the main chapters for deep dives: [Chapter 3 — The Transformer](./03_the_transformer.md), [Chapter 4 — Attention](./04_attention_sdpa_and_mha.md).

---

[← Back to Table of Contents](./README.md) · [Appendix B — Tokenization Deep Dive →](./appendix_b_tokenization.md)

---

*Last updated: April 2026*
