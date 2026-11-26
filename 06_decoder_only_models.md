---
title: "Chapter 6 — Decoder-Only Models"
description: "The causal LM objective, a full decoder pass with shapes, the GPT and Llama lineages, the late-2026 open-weight landscape (DeepSeek-V4, Qwen3.5/3.6, Kimi K3, GLM-5, Gemma 4, gpt-oss, Mistral 3), the closed frontier, and the generation loop."
related:
  - { id: models, why: "Architecture summaries across model generations." }
---

[← Back to Table of Contents](./README.md)

# Chapter 6 — Decoder-Only Models

Decoder-only transformers — a stack of causally masked blocks trained to predict the next token — are the architecture behind essentially every general-purpose LLM shipped since 2020. This chapter assembles the pieces from Chapters 2–5 into complete models and surveys who builds them.

> **The one-sentence version:** a decoder-only model is a causal transformer trained on next-token prediction, and since 2024 the differences between families are mostly in sparsity (MoE), attention variant, data and post-training rather than in the block itself.

## The Causal Language Modelling Objective

Decoder-only models are trained to predict each token from the ones before it, minimising the cross-entropy:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

This simple objective, scaled to tens of trillions of tokens, produces the base models on which all reasoning, coding and agent behaviour is later built.

<div class="diagram">
<div class="diagram-title">Causal Language Modelling — Training vs Inference</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Training (teacher forcing)</div>
    <ul>
      <li>All positions processed in parallel</li>
      <li>Causal mask ensures each position only sees the past</li>
      <li>Loss computed at every position at once</li>
      <li>One forward pass yields T training signals</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Inference (autoregressive)</div>
    <ul>
      <li>Tokens generated one at a time</li>
      <li>Each new token appended to the context</li>
      <li>KV cache avoids recomputing the past</li>
      <li>Prefill (parallel, compute-bound) → decode (sequential, bandwidth-bound)</li>
    </ul>
  </div>
</div>
</div>

Teacher forcing is why decoder-only training is so efficient: a single 8K-token sequence gives 8K next-token predictions from one forward and backward pass. It is also why training and inference behave differently — at inference the model conditions on its *own* earlier outputs, which can drift from anything it saw in training (exposure bias), one motivation for the RL stages in Chapters 15 and 35.

## Full Decoder Pass — Tensor Shapes

<div class="diagram">
<div class="diagram-title">End-to-End Tensor Flow Through a Decoder Model</div>
<div class="flow">
  <div class="flow-node accent wide">Token IDs: [B, T] <small>integer indices</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Token Embedding: [B, T, d_model] <small>positions are added later, inside attention (RoPE)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Layer 1: RMSNorm → GQA/MLA → residual → RMSNorm → SwiGLU FFN or MoE → residual <small>[B, T, d_model]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide dim">Layers 2 … N <small>[B, T, d_model] — shape preserved through all layers</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Final RMSNorm: [B, T, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">LM Head (unembedding): [B, T, V] <small>logits over the vocabulary</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Softmax → next-token distribution <small>only the last position is needed at inference</small></div>
</div>
</div>

## The GPT Lineage

GPT (Generative Pre-trained Transformer) established the decoder-only paradigm. OpenAI stopped publishing architectures after GPT-3, so later entries list only what was disclosed:

<div class="diagram">
<div class="diagram-title">From GPT-1 to GPT-5.5</div>
<div class="timeline">
  <div class="timeline-item accent">
    <div class="timeline-year">2018</div>
    <div class="timeline-title">GPT-1</div>
    <div class="timeline-desc">117M params, 12 layers. Unsupervised pre-training + supervised fine-tuning beats task-specific models.</div>
  </div>
  <div class="timeline-item purple">
    <div class="timeline-year">2019</div>
    <div class="timeline-title">GPT-2</div>
    <div class="timeline-desc">1.5B params, 48 layers, pre-norm. Zero-shot task transfer; staged release over misuse concerns.</div>
  </div>
  <div class="timeline-item green">
    <div class="timeline-year">2020</div>
    <div class="timeline-title">GPT-3</div>
    <div class="timeline-desc">175B params, 96 layers, d = 12,288. In-context learning and few-shot prompting.</div>
  </div>
  <div class="timeline-item orange">
    <div class="timeline-year">2022–24</div>
    <div class="timeline-title">InstructGPT → ChatGPT → GPT-4 → GPT-4o</div>
    <div class="timeline-desc">RLHF makes models follow instructions; GPT-4 (architecture undisclosed) adds vision; GPT-4o is natively omni-modal.</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-year">Sep 2024</div>
    <div class="timeline-title">o1</div>
    <div class="timeline-desc">First widely deployed reasoning model: RL-trained to think in long hidden chains before answering.</div>
  </div>
  <div class="timeline-item cyan">
    <div class="timeline-year">Aug 2025</div>
    <div class="timeline-title">GPT-5 and gpt-oss</div>
    <div class="timeline-desc">GPT-5 routes between fast and thinking modes; gpt-oss-120b/20b are OpenAI's first open-weight LLMs since GPT-2 (Apache 2.0, MoE, MXFP4).</div>
  </div>
  <div class="timeline-item pink">
    <div class="timeline-year">2025–26</div>
    <div class="timeline-title">GPT-5.x cadence</div>
    <div class="timeline-desc">Point releases every few months (5.1, 5.2, 5.4 …); GPT-5.5 (April 2026) is described as the first fully retrained base model since GPT-4.5.</div>
  </div>
</div>
</div>

## The Llama Lineage

Llama (Meta, 2023–2025) was the most influential open-weight family of its era. Its choices — pre-RMSNorm, RoPE, SwiGLU, GQA, no biases — became the default "Llama-style" block that most other labs started from.

<div class="diagram">
<div class="diagram-title">Llama Architecture — Key Design Choices</div>
<div class="layer-stack">
  <div class="layer accent">Pre-RMSNorm (normalise before attention and FFN)</div>
  <div class="layer purple">GQA attention with RoPE</div>
  <div class="layer green">SwiGLU feed-forward network (three projections)</div>
  <div class="layer orange">No bias terms in the linear layers</div>
  <div class="layer cyan">Tied embeddings only in the smallest models (Llama 3.2 1B/3B)</div>
</div>
</div>

### Model Size Comparison

| Model | Params | Layers | d_model | Heads (Q/KV) | d_k | FFN dim | Context | Vocab |
|-------|-------:|-------:|--------:|:------------:|----:|--------:|--------:|------:|
| Llama 1 7B | 6.7B | 32 | 4096 | 32/32 | 128 | 11008 | 2K | 32,000 |
| Llama 2 7B | 6.7B | 32 | 4096 | 32/32 | 128 | 11008 | 4K | 32,000 |
| Llama 2 70B | 69B | 80 | 8192 | 64/8 | 128 | 28672 | 4K | 32,000 |
| Llama 3.1 8B | 8.0B | 32 | 4096 | 32/8 | 128 | 14336 | 128K | 128,256 |
| Llama 3.1 70B | 70.6B | 80 | 8192 | 64/8 | 128 | 28672 | 128K | 128,256 |
| Llama 3.1 405B | 405B | 126 | 16384 | 128/8 | 128 | 53248 | 128K | 128,256 |
| Llama 4 Scout | 109B (17B active) | 48 | 5120 | 40/8 | 128 | MoE, 16 experts | 10M (claimed) | 202,048 |
| Llama 4 Maverick | 400B (17B active) | 48 | 5120 | 40/8 | 128 | MoE, 128 experts | 1M | 202,048 |

The evolution in one line: MHA → GQA, 32K → 128K → 202K vocabulary, 2K → 128K context, then dense → MoE with interleaved RoPE/NoPE layers in Llama 4 (April 2025). Llama 4 was received coolly next to DeepSeek and Qwen, and in April 2026 Meta's new superintelligence lab shipped **Muse Spark** as a closed model; as of late 2026 Meta has said it intends to keep releasing some open weights, but no Llama 5 has shipped.

## The Open-Weight Landscape (late 2026)

Leadership in open weights moved decisively to Chinese labs during 2025, with Google, OpenAI, Mistral and NVIDIA keeping strong entries. The current flagship of each major family:

| Family (lab) | Latest open flagship | Total / active params | Notable architecture | Licence |
|---|---|---|---|---|
| **DeepSeek** | DeepSeek-V4-Pro / V4-Flash (Apr 2026) | 1.6T / 49B; 284B / 13B | MoE, compressed sparse attention, hyper-connections, MTP, 1M context | MIT |
| **Qwen** (Alibaba) | Qwen3.5-397B-A17B (Feb 2026); Qwen3.6-35B-A3B and 27B (Apr 2026) | 397B / 17B | 3:1 Gated DeltaNet : gated attention hybrid, 512 experts, 262K context | Apache 2.0 |
| **Kimi** (Moonshot) | Kimi K3 (Jul 2026) | 2.8T / ~104B | MoE (16 of 896 experts), Kimi Delta Attention + gated MLA hybrid, 1M context | Custom |
| **GLM** (Zhipu) | GLM-5 (Feb 2026) and point releases | 744B / 40B | MoE, DeepSeek Sparse Attention | MIT |
| **Gemma** (Google) | Gemma 4 (Apr 2026): E2B, E4B, 26B MoE, 31B dense | 26B / 3.8B (MoE) | Natively multimodal; up to 256K context | Apache 2.0 |
| **gpt-oss** (OpenAI) | gpt-oss-120b / 20b (Aug 2025) | 117B / 5.1B; 21B / 3.6B | MoE, alternating windowed/full attention, sinks, MXFP4 weights | Apache 2.0 |
| **Mistral** | Mistral Large 3; Ministral 3 (Dec 2025) | 675B / 41B | MoE, multimodal, 256K context | Apache 2.0 |
| **Llama** (Meta) | Llama 4 Scout / Maverick (Apr 2025) | 109B–400B / 17B | MoE, iRoPE | Llama licence |
| **OLMo** (Ai2) | OLMo 3 (Nov 2025) | 7B, 32B dense | Fully open: data, code, checkpoints | Apache 2.0 |
| **Nemotron** (NVIDIA) | Nemotron 3 family (Dec 2025 →) | various | Mamba-Transformer hybrid MoE | NVIDIA open licence |

<div class="callout warn">
<span class="callout-title">Open weights ≠ open source</span>
<p>Most of these release weights and an inference recipe; few release training data or code. OLMo (and Hugging Face's SmolLM) are the exceptions you can fully reproduce. Licences also differ: Apache 2.0 and MIT are permissive; the Llama and Kimi licences add usage conditions. Read the licence before shipping.</p>
</div>

Five trends run through that table:

1. **MoE at the top.** Every open flagship above ~100B is a mixture of experts, and sparsity keeps rising: DeepSeek-V3 activated 37B of 671B (5.5%); Qwen3.5 activates 17B of 397B (4.3%); Kimi K3 about 104B of 2.8T (3.7%). See [Chapter 33](./33_mixture_of_experts.md).
2. **Attention became a menu.** MLA, learned sparse attention and linear-attention hybrids replaced plain GQA in the largest models ([Chapter 5](./05_attention_gqa_mqa_mla.md), [Chapter 34](./34_ssms_and_alternatives.md)).
3. **Thinking is built in.** Qwen3, DeepSeek-V3.1 onward and most 2026 models ship one checkpoint with a thinking and a non-thinking mode, selected by the chat template or a token budget ([Chapter 35](./35_reasoning_models.md)).
4. **Native multimodality.** Gemma 4, Qwen3.5/3.6, Mistral Large 3 and Kimi K3 accept images (and often video) in the base model rather than via a bolted-on adapter ([Chapter 8](./08_multimodal_models.md)).
5. **Trained for low precision.** gpt-oss shipped MXFP4 expert weights; DeepSeek trained V3 in FP8. Checkpoints increasingly arrive in the precision they are meant to be served in ([Chapter 19](./19_data_types_and_precision.md)).

## The Closed Frontier (late 2026)

Closed labs disclose little architecture, but their release cadence sets the bar:

| Lab | Family as of late 2026 | What is publicly known |
|---|---|---|
| **OpenAI** | GPT-5.x (GPT-5, Aug 2025 → GPT-5.5, Apr 2026, and later point releases) | Unified fast + thinking system; heavy RL; agentic coding (Codex) variants |
| **Anthropic** | Claude 5 family (2026): Fable, Opus, Sonnet, Haiku tiers | Hybrid reasoning with adjustable effort; long-horizon agentic coding and computer use |
| **Google DeepMind** | Gemini 3 (Nov 2025) → Gemini 3.5 series (2026) | Natively multimodal, long context, Deep Think parallel reasoning mode |
| **xAI** | Grok 4 series (2025 →) | Large-scale RL; real-time search integration |
| **Meta** | Muse Spark (Apr 2026) | Meta Superintelligence Labs' first model; closed |

These names will be stale within months; the architectural lessons in this guide will not.

## Using Decoder Models with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "Explain attention in one paragraph."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)

# Inspect intermediate shapes
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

print(inputs["input_ids"].shape)        # [1, T]
print(len(out.hidden_states))           # 33 = embedding output + one per layer
print(out.hidden_states[-1].shape)      # [1, T, 4096]
print(out.logits.shape)                 # [1, T, 128256]

# Generate
gen = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Autoregressive Generation Loop

Under the hood, `model.generate()` runs an autoregressive loop:

```python
import torch

@torch.no_grad()
def generate_simple(model, input_ids, max_new_tokens, temperature=1.0, eos_ids=()):
    """Simplified generation for batch size 1."""
    past = None                                  # KV cache (a Cache object in recent transformers)
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids if past is None else input_ids[:, -1:],
                    past_key_values=past, use_cache=True)
        past = out.past_key_values

        logits = out.logits[:, -1, :] / temperature           # [1, V]
        next_token = torch.multinomial(torch.softmax(logits, -1), 1)  # [1, 1]
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() in eos_ids:
            break
    return input_ids

eos = model.generation_config.eos_token_id
eos_ids = set(eos if isinstance(eos, list) else [eos])     # Llama 3.1 Instruct has three
```

On the first call (**prefill**) the whole prompt is processed at once; afterwards (**decode**) only the newest token is fed in and the KV cache supplies the rest. Note `eos_ids`: Llama 3.1 Instruct ends turns with `<|eot_id|>`, not `<|end_of_text|>`, and its generation config lists several stop IDs — checking only `config.eos_token_id` is a classic cause of models that never stop talking. [Chapter 17](./17_kv_cache_mechanics.md) covers the cache in detail.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>The open-weight frontier now sits at 1–3 trillion parameters with 3–5% active per token. What does a model like that cost to <em>serve</em> versus to <em>train</em>, and who can actually run the weights they are given? Keep the question in mind for Chapters 31 and 33.</p>
</div>

## Key Takeaways

- Decoder-only = causal mask + next-token loss; teacher forcing gives T training signals per sequence.
- The Llama block (pre-RMSNorm, RoPE, SwiGLU, GQA, no bias) is the common ancestor of most open models.
- As of late 2026 the open frontier is MoE (DeepSeek-V4, Qwen3.5, Kimi K3, GLM-5, Mistral Large 3) with sparse or hybrid attention, built-in thinking modes and native multimodality; Meta has moved its flagship to a closed model.
- Closed frontier families (GPT-5.x, Claude 5, Gemini 3.x) release every few months; treat any specific model name as perishable.
- In code: use the chat template, decode only the generated slice, and stop on every EOS ID the model defines.

## Check Yourself

<details class="check">
<summary>Why does one training sequence of 4,096 tokens give 4,096 loss terms but generating 4,096 tokens takes 4,096 forward passes?</summary>
<p>In training the true tokens are known, so all positions are predicted in parallel under the causal mask (teacher forcing). In generation each token depends on the previous <em>generated</em> token, so the passes are inherently sequential; the KV cache only makes each pass cheaper.</p>
</details>

<details class="check">
<summary>What fraction of its parameters does each model activate per token: DeepSeek-V4-Flash, Qwen3.5-397B-A17B, gpt-oss-120b?</summary>
<p>13/284 ≈ 4.6%; 17/397 ≈ 4.3%; 5.1/117 ≈ 4.4%. All three cluster around 4–5% — the compute of a ~15B dense model with the knowledge capacity of a much larger one.</p>
</details>

<details class="check">
<summary>A fine-tuned Llama 3.1 Instruct model keeps generating new "user" turns after its answer. What is the likely bug?</summary>
<p>Generation stops only on <code>&lt;|end_of_text|&gt;</code> while the model ends its turn with <code>&lt;|eot_id|&gt;</code>. Pass all of the generation config's EOS IDs (or the chat template's end-of-turn token) as stop tokens.</p>
</details>

<details class="check">
<summary>Name three architectural differences between Llama 3.1 70B and DeepSeek-V4-Pro.</summary>
<p>Dense FFN vs mixture of experts; GQA vs compressed sparse attention (MLA lineage); single residual stream vs manifold-constrained hyper-connections — plus multi-token prediction and a 1M-token context.</p>
</details>

<details class="check">
<summary>Why is "open weights" not the same as "open source" for a model like Qwen3.5?</summary>
<p>The weights and inference code are published, but the training data, data pipeline and full training code are not, so the model cannot be reproduced or audited end to end. Fully open releases such as OLMo publish all of those.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Improving Language Understanding by Generative Pre-Training | Radford et al. (OpenAI), 2018 | [cdn.openai.com](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | GPT-1 |
| Language Models are Unsupervised Multitask Learners | Radford et al. (OpenAI), 2019 | [cdn.openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | GPT-2 |
| Language Models are Few-Shot Learners | Brown et al., 2020 | [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) | GPT-3 |
| LLaMA: Open and Efficient Foundation Language Models | Touvron et al., 2023 | [arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971) | Llama block |
| The Llama 3 Herd of Models | Meta AI, 2024 | [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783) | Llama 3.1 sizes |
| The Llama 4 herd | Meta AI, 2025 | [ai.meta.com/blog/llama-4-multimodal-intelligence](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Llama 4 Scout/Maverick |
| DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437) | MoE + MLA flagship |
| DeepSeek V4 Preview Release | DeepSeek, 2026 | [api-docs.deepseek.com/news/news260424](https://api-docs.deepseek.com/news/news260424) | V4-Pro / V4-Flash sizes, 1M context |
| Qwen3.5: Towards Native Multimodal Agents | Qwen Team, 2026 | [simonwillison.net/2026/Feb/17/qwen35](https://simonwillison.net/2026/Feb/17/qwen35/) | Qwen3.5 release summary |
| Kimi K3 | Moonshot AI, 2026 | [github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) | K3 size and hybrid attention |
| GLM-5: from Vibe Coding to Agentic Engineering | Zhipu AI, 2026 | [arxiv.org/abs/2602.15763](https://arxiv.org/abs/2602.15763) | GLM-5 |
| Gemma 4 announcement | Google, 2026 | [blog.google](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | Gemma 4 sizes and licence |
| gpt-oss model card | OpenAI, 2025 | [openai.com/index/gpt-oss-model-card](https://openai.com/index/gpt-oss-model-card/) | gpt-oss architecture |
| Introducing Mistral 3 | Mistral AI, 2025 | [mistral.ai/news/mistral-3](https://mistral.ai/news/mistral-3) | Mistral Large 3 |
| With Muse Spark, Meta pivots away from open weights | DeepLearning.AI, 2026 | [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/with-muse-spark-meta-pivots-away-from-its-open-weights-llama-strategy) | Meta's 2026 strategy change |

---

**Next:** [Chapter 7 — Encoder & Seq2Seq Models](./07_encoder_and_seq2seq_models.md) — the bidirectional and encoder-decoder architectures that still power embeddings, classifiers and speech recognition.

[← Back to Table of Contents](./README.md)
