[← Back to Table of Contents](./README.md)

# Chapter 6 — Decoder-Only Models

> *"The decoder-only transformer, trained with a simple next-token prediction objective on internet-scale data, turned out to be the most powerful architecture humanity has ever built for language understanding and generation."*

## The Causal Language Modelling Objective

Decoder-only models are trained to predict the next token given all previous tokens. The training objective is straightforward — minimize the cross-entropy loss:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

This simple objective, scaled to trillions of tokens, produces models capable of reasoning, code generation, and multilingual understanding.

<div class="diagram">
<div class="diagram-title">Causal Language Modelling — Training vs Inference</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Training (Teacher Forcing)</div>
    <ul>
      <li>All positions processed in parallel</li>
      <li>Causal mask ensures each position only sees past</li>
      <li>Loss computed at every position simultaneously</li>
      <li>One forward pass for the entire sequence</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Inference (Autoregressive)</div>
    <ul>
      <li>Tokens generated one at a time</li>
      <li>Each new token appended to context</li>
      <li>KV-cache avoids recomputation</li>
      <li>Prefill (parallel) → Decode (sequential)</li>
    </ul>
  </div>
</div>
</div>

## Full Decoder Pass — Tensor Shapes

<div class="diagram">
<div class="diagram-title">End-to-End Tensor Flow Through a Decoder Model</div>
<div class="flow">
  <div class="flow-node accent wide">Token IDs: [B, T] <small>integer indices</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Token Embedding: [B, T, d_model] <small>+ positional encoding</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Layer 1: RMSNorm → GQA → residual → RMSNorm → SwiGLU FFN → residual <small>[B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide" style="opacity: 0.7;">Layers 2 … N <small>[B, T, d_model] — shape preserved through all layers</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Final RMSNorm: [B, T, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">LM Head (unembedding): [B, T, vocab_size] <small>logits over vocabulary</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Softmax → next-token probability distribution</div>
</div>
</div>

## The GPT Family

GPT (Generative Pre-trained Transformer) established the decoder-only paradigm:

<div class="diagram">
<div class="diagram-title">GPT Family Evolution</div>
<div class="timeline">
  <div class="timeline-event accent">
    <div class="event-date">2018</div>
    <div class="event-title">GPT-1</div>
    <div class="event-desc">117M params. 12 layers. Showed unsupervised pre-training + supervised fine-tuning works.</div>
  </div>
  <div class="timeline-event purple">
    <div class="event-date">2019</div>
    <div class="event-title">GPT-2</div>
    <div class="event-desc">1.5B params. 48 layers. Zero-shot task transfer. "Too dangerous to release" (at the time).</div>
  </div>
  <div class="timeline-event green">
    <div class="event-date">2020</div>
    <div class="event-title">GPT-3</div>
    <div class="event-desc">175B params. 96 layers, d=12288. In-context learning. Few-shot prompting.</div>
  </div>
  <div class="timeline-event orange">
    <div class="event-date">2023–24</div>
    <div class="event-title">GPT-4 / GPT-4o</div>
    <div class="event-desc">Rumored MoE architecture. Multimodal. RLHF-aligned. State-of-the-art on most benchmarks.</div>
  </div>
</div>
</div>

## The LLaMA Architecture

LLaMA (Meta, 2023–2024) is the most influential open-source decoder model family. Its architectural choices have become the de facto standard:

<div class="diagram">
<div class="diagram-title">LLaMA Architecture — Key Design Choices</div>
<div class="layer-stack">
  <div class="layer accent">Pre-RMSNorm (normalize before attention, not after)</div>
  <div class="layer purple">GQA Attention with RoPE positional encoding</div>
  <div class="layer green">SwiGLU Feed-Forward Network (3 projection matrices)</div>
  <div class="layer orange">No bias terms anywhere (QKV, FFN, RMSNorm)</div>
  <div class="layer cyan">Tied/untied embedding (varies by version)</div>
</div>
</div>

### Model Size Comparison

| Model | Params | Layers | d_model | Heads (Q/KV) | d_k | FFN dim | Context | Vocab |
|-------|-------:|-------:|--------:|:------------:|----:|--------:|--------:|------:|
| LLaMA-1 7B | 6.7B | 32 | 4096 | 32/32 | 128 | 11008 | 2048 | 32000 |
| LLaMA-2 7B | 6.7B | 32 | 4096 | 32/32 | 128 | 11008 | 4096 | 32000 |
| LLaMA-2 70B | 70B | 80 | 8192 | 64/8 | 128 | 28672 | 4096 | 32000 |
| LLaMA-3 8B | 8B | 32 | 4096 | 32/8 | 128 | 14336 | 8192 | 128256 |
| LLaMA-3 70B | 70B | 80 | 8192 | 64/8 | 128 | 28672 | 8192 | 128256 |
| LLaMA-3.1 405B | 405B | 126 | 16384 | 128/8 | 128 | 53248 | 128K | 128256 |

Key evolution: MHA → GQA (LLaMA-2 70B onwards), 32K → 128K vocab (LLaMA-3), 2K → 128K context (LLaMA-3.1).

## The Open-Source Landscape

<div class="diagram">
<div class="diagram-title">Major Open-Weight Decoder-Only Model Families</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">LLaMA (Meta)</div>
    <div class="card-desc">RMSNorm, SwiGLU, RoPE, GQA. 1B–405B. Most forked architecture.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Mistral / Mixtral</div>
    <div class="card-desc">Sliding window attention, GQA, MoE (Mixtral). Strong at 7B scale.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Qwen 2.5 (Alibaba)</div>
    <div class="card-desc">GQA, SwiGLU, RoPE. Strong multilingual. 0.5B–72B.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Gemma 2 (Google)</div>
    <div class="card-desc">Alternating local/global attention. Logit soft-capping. 2B–27B.</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-title">Phi-3/4 (Microsoft)</div>
    <div class="card-desc">High-quality small models. Curated training data. 3.8B–14B.</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-title">DeepSeek (V2/V3)</div>
    <div class="card-desc">MLA attention, MoE. 236B total / 21B active. Cost-efficient.</div>
  </div>
</div>
</div>

## Using Decoder Models with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain attention in one paragraph."}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

# Inspect intermediate shapes
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

print(f"Input:   {input_ids.shape}")                     # [1, T]
print(f"Hidden:  {outputs.hidden_states[-1].shape}")      # [1, T, 4096]
print(f"Logits:  {outputs.logits.shape}")                 # [1, T, 128256]

# Generate
output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Autoregressive Generation Loop

Under the hood, `model.generate()` runs an autoregressive loop. Here's what happens step by step:

```python
def generate_simple(model, input_ids, max_new_tokens, temperature=1.0):
    """Simplified autoregressive generation loop."""
    past_key_values = None  # KV-cache

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values  # cache K, V

        # Sample from the logits of the last position
        logits = outputs.logits[:, -1, :] / temperature  # [B, vocab_size]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == model.config.eos_token_id:
            break

    return input_ids
```

Notice: on the first pass (**prefill**), the entire prompt is processed at once. On subsequent passes (**decode**), only the last token is fed in — the KV-cache stores everything else. See [Chapter 14](./14_kv_cache_mechanics.md) for full details.

## What's Next

Decoder-only models dominate generation, but they're not the only game in town. The next chapter covers **encoder** and **encoder-decoder** architectures that excel at understanding tasks like classification, NER, and translation.

[← Previous: Chapter 5 — GQA, MQA & MLA](./05_attention_gqa_mqa_mla.md) · **Next: [Chapter 7 — Encoder & Seq2Seq Models →](./07_encoder_and_seq2seq_models.md)**

---

*Last updated: April 2026*
