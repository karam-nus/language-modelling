---
title: "Chapter 29 — The transformers Library Internals"
description: "How transformers is put together: config → Auto mapping → model classes, the Llama module tree as printed today, the forward pass and loss, the Cache API, attention back-ends and the attention interface, modular model files, tensor-parallel plans, registering a custom architecture correctly, and debugging a port."
related:
  - { id: huggingface, why: "The Hugging Face ecosystem." }
  - { id: pytorch-essentials, why: "The PyTorch underneath transformers." }
---

[← Back to Table of Contents](./README.md)

# Chapter 29 — The transformers Library Internals

Knowing how `transformers` is organised lets you read any model's source in minutes, find where a behaviour lives (RoPE scaling, attention masking, cache handling), add an architecture, and debug numerical mismatches between implementations.

> **The one-sentence version:** a `config.json` names a model type, the Auto classes map it to a PyTorch module tree defined in one readable modeling file, and pluggable interfaces — attention back-ends, cache classes, generation strategies, parallel plans — keep that file simple.

## Class Hierarchy

<div class="diagram">
<div class="diagram-title">transformers Class Architecture</div>
<div class="layer-stack">
  <div class="layer orange">PreTrainedModel — from_pretrained / save_pretrained, weight init, gradient checkpointing, device and dtype handling</div>
  <div class="layer accent">LlamaModel (QwenModel, MistralModel, …) — embeddings + N decoder layers + final norm (+ rotary embedding module)</div>
  <div class="layer green">LlamaForCausalLM — adds lm_head and the loss; inherits GenerationMixin</div>
  <div class="layer purple">GenerationMixin — generate(): greedy, sampling, beam search, assisted (speculative) decoding, cache management</div>
</div>
</div>

```python
# What AutoModelForCausalLM.from_pretrained(repo) does, in outline:
# 1. read config.json → "model_type": "llama"            → LlamaConfig
# 2. look up the causal-LM mapping: "llama" → LlamaForCausalLM
# 3. build the module tree on the "meta" device (no memory allocated)
# 4. stream safetensors shards straight into place, in the requested dtype and device map
```

## Model Anatomy: LlamaForCausalLM

`print(model)` for Llama 3.1 8B in a recent version:

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
```

Two changes from older versions are worth knowing: there is one `LlamaAttention` class (the eager/SDPA/Flash variants were merged behind an attention interface), and the rotary embedding is computed once in `LlamaModel` and passed to every layer as `(cos, sin)`.

## Forward Pass Walkthrough

<div class="diagram">
<div class="diagram-title">Shapes (Llama 3.1 8B)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row"><span class="ts-label">input_ids</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">embed_tokens</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">4096</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">32 × decoder layer</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim feature">4096</span><span class="ts-bracket">]</span></div></div>
  <div class="tensor-flow-row"><span class="ts-label">norm → lm_head</span><div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">T</span><span class="ts-sep">,</span><span class="ts-dim vocab">128256</span><span class="ts-bracket">]</span></div></div>
</div>
</div>

```python
# Simplified LlamaForCausalLM.forward
def forward(self, input_ids, attention_mask=None, position_ids=None,
            past_key_values=None, labels=None, logits_to_keep=0, **kwargs):
    h = self.model.embed_tokens(input_ids)                         # [B, T, d]
    cos, sin = self.model.rotary_emb(h, position_ids)              # computed once, shared by all layers
    mask = create_causal_mask(...)                                 # causal + padding (+ sliding window)
    for layer in self.model.layers:
        h = layer(h, attention_mask=mask, position_embeddings=(cos, sin),
                  past_key_values=past_key_values, **kwargs)
    h = self.model.norm(h)
    logits = self.lm_head(h[:, -logits_to_keep:, :])               # generate() keeps only the last position
    loss = None
    if labels is not None:                                         # shift inside; ignore_index = -100
        loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)
```

Note `logits_to_keep`: during generation only the final position's logits are computed, avoiding a `[B, T, V]` tensor at prefill — at V = 128K and T = 32K that alone would be 8 GB in BF16 for one sequence.

## The Cache API

```python
from transformers import DynamicCache

cache = DynamicCache()                                   # grows as tokens are added
out = model(**inputs, past_key_values=cache, use_cache=True)       # prefill
for _ in range(32):                                                # decode
    nxt = out.logits[:, -1:].argmax(-1)
    out = model(nxt, past_key_values=cache, use_cache=True)

print(cache.get_seq_length())               # prompt length + 32
k = cache.layers[0].keys                    # [B, n_kv_heads, T_cached, head_dim]
```

Other cache classes cover other needs: `StaticCache` (pre-allocated, so `torch.compile` can capture fixed shapes), sliding-window and hybrid caches (Gemma's local/global layers, gpt-oss), quantized caches, offloaded caches, and recurrent-state caches for hybrid SSM models. `generate()` picks a suitable one from the config unless you pass your own.

## Attention Back-Ends

| `attn_implementation` | What runs | Notes |
|---------|-----|-------|
| `"eager"` | Plain PyTorch matmul + softmax | Reference; returns attention weights; O(T²) memory |
| `"sdpa"` | `F.scaled_dot_product_attention` | Default; picks Flash / memory-efficient / math kernels |
| `"flash_attention_2"` / `"flash_attention_3"` | Dao's kernels | Fast; variable-length packing without padding |
| `"flex_attention"` | PyTorch FlexAttention | Custom masks and score mods compiled into one kernel |
| `"kernels-community/…"` | A kernel downloaded from the Hub via `kernels` | Newest kernels without compiling packages |

All of them plug into the same **attention interface**: a model's attention module calls `ALL_ATTENTION_FUNCTIONS[config._attn_implementation](...)`. You can register your own function — for example a custom sparse attention — and select it by name.

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             attn_implementation="sdpa", dtype="auto")
```

## Reading Modern Model Code: Modular Files

New architectures are often near-copies of old ones (Qwen3 ≈ Llama + QK-norm). To avoid thousands of duplicated lines, the library uses **modular files**: `modular_qwen3.py` declares only what differs, inheriting from Llama classes, and a converter generates the full, standalone `modeling_qwen3.py`. When reading a model, start from its modular file to see what is actually new; when debugging, read the generated modeling file, which is what runs.

## Tensor Parallelism in from_pretrained

Models carry a **tensor-parallel plan** in their config (which projections are column- or row-parallel). Launch with `torchrun` and pass `tp_plan="auto"` to shard a model across GPUs without writing any parallel code:

```python
# torchrun --nproc-per-node 4 infer.py
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct",
                                             tp_plan="auto", dtype=torch.bfloat16)
```

It is convenient for experiments; for production serving, vLLM and SGLang (Chapter 31) remain faster.

## Adding a Custom Architecture

For local use, register the config and model with the Auto classes; for Hub distribution with `trust_remote_code`, use `register_for_auto_class`:

```python
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, GenerationMixin,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

class TinyConfig(PretrainedConfig):
    model_type = "tiny_lm"
    def __init__(self, vocab_size=32000, d_model=512, n_layers=8, n_heads=8, **kw):
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        super().__init__(vocab_size=vocab_size, **kw)

class TinyForCausalLM(PreTrainedModel, GenerationMixin):      # GenerationMixin gives .generate()
    config_class = TinyConfig
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()                                      # weight init, tying, etc.
    def forward(self, input_ids, labels=None, **kw):
        h = self.embed(input_ids)
        for b in self.blocks:
            h = b(h)
        logits = self.lm_head(self.norm(h))
        loss = None if labels is None else self.loss_function(logits, labels, self.config.vocab_size)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

AutoConfig.register("tiny_lm", TinyConfig)                   # local registration
AutoModelForCausalLM.register(TinyConfig, TinyForCausalLM)
# For the Hub: TinyConfig.register_for_auto_class(); TinyForCausalLM.register_for_auto_class("AutoModelForCausalLM")
```

(This toy model has no KV cache, so `generate()` recomputes the whole prefix each step — fine for experiments.)

## Debugging a Port

When your implementation (or a fast engine) disagrees with the reference:

1. **Compare layer by layer** on the same input in FP32: embeddings, then each layer's output, using forward hooks; find the first layer whose output diverges beyond ~1e-5 relative error.
2. **Check config semantics** ([Appendix C](./appendix_c_model_config.md)): `rope_theta` and `rope_scaling`, `head_dim` that is not `hidden_size / num_heads`, `rms_norm_eps`, `tie_word_embeddings`, sliding-window and layer-type lists, `attention_bias`, partial rotary factors.
3. **Check conventions**: RoPE pairing (interleaved vs rotate-half, [Chapter 2](./02_embeddings.md)), whether norms add 1 to their weight (Gemma's `(1 + w)`), attention scaling, logit softcapping, sink terms.
4. **Check the tokenizer and template**: BOS added twice is a classic silent bug.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>transformers prioritises one readable file per model over shared abstractions, even at the cost of duplication. For a library that serves as the specification other engines port from, why might readability beat reuse — and what does the modular-file system trade to get both?</p>
</div>

## Key Takeaways

- config.json → model_type → Auto mapping → a module tree built on the meta device and filled from safetensors.
- Modern Llama code: one attention class behind an attention interface; rotary embeddings computed once per forward; `logits_to_keep` avoids full-sequence logits during generation.
- The Cache API (`DynamicCache`, `StaticCache`, sliding/hybrid/quantized caches) replaced tuples of tensors.
- Modular files show what a new architecture changes; `tp_plan="auto"` gives quick tensor parallelism.
- Register custom models with `AutoConfig.register` / `AutoModelForCausalLM.register`, inherit `GenerationMixin`, and debug ports layer by layer.

## Check Yourself

<details class="check">
<summary>Why does generate() pass <code>logits_to_keep=1</code> during prefill?</summary>
<p>Only the last position's logits are needed to sample the next token. Computing all T positions would materialise a [B, T, V] tensor — gigabytes at long context with a 128K+ vocabulary — and waste a large matmul.</p>
</details>

<details class="check">
<summary>What does building the model on the "meta" device buy during from_pretrained?</summary>
<p>Modules are created with shapes but no storage, so a 70B model is not first allocated with random weights and then overwritten; weights stream directly from safetensors into their final device and dtype, halving peak memory and load time.</p>
</details>

<details class="check">
<summary>Why would you choose StaticCache over DynamicCache?</summary>
<p>A static, pre-allocated cache keeps tensor shapes fixed across decode steps, so <code>torch.compile</code> (and CUDA graphs) can capture one graph and replay it, removing per-step overhead — at the cost of reserving memory for the maximum length up front.</p>
</details>

<details class="check">
<summary>Your port of a Gemma model matches Llama code perfectly but outputs garbage. Name two Gemma-specific conventions to check.</summary>
<p>RMSNorm scales by (1 + weight) rather than weight; embeddings are multiplied by √d_model; plus query pre-attention scaling, local/global layer patterns with different RoPE bases, and (Gemma 2) logit softcapping.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Transformers documentation: model internals, attention interface, cache | Hugging Face, 2025–26 | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) | Current APIs |
| Modular transformers | Hugging Face | [huggingface.co/docs/transformers/modular_transformers](https://huggingface.co/docs/transformers/modular_transformers) | Modular model files |
| Caching in transformers | Hugging Face | [huggingface.co/docs/transformers/cache_explanation](https://huggingface.co/docs/transformers/cache_explanation) | Cache classes |
| Tensor parallelism in transformers | Hugging Face | [huggingface.co/docs/transformers/perf_infer_gpu_multi](https://huggingface.co/docs/transformers/perf_infer_gpu_multi) | tp_plan |
| modeling_llama.py | Hugging Face | [github.com/huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | Reference implementation |

---

**Next:** [Chapter 30 — Evaluation & Benchmarks](./30_evaluation_and_benchmarks.md) — how to tell whether any of this produced a good model.

[← Back to Table of Contents](./README.md)
