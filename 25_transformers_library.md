[← Back to Table of Contents](./README.md)

# Chapter 25 — The transformers Library Internals

> *"Understanding how transformers (the library) is structured lets you read any model's source, add custom architectures, and debug generation issues."*

## Class Hierarchy

<div class="diagram">
<div class="diagram-title">transformers Class Architecture</div>
<div class="layer-stack">
  <div class="layer orange">PreTrainedModel — Base class: from_pretrained(), save_pretrained(), generate(), gradient_checkpointing</div>
  <div class="layer accent">LlamaModel (or GPT2Model, MistralModel, ...) — Core transformer: embeddings + N decoder layers + final norm</div>
  <div class="layer green">LlamaForCausalLM — Adds lm_head (linear projection to vocab). Computes cross-entropy loss.</div>
  <div class="layer purple">GenerationMixin — generate() method: sampling, beam search, contrastive, speculative decode</div>
</div>
</div>

Every model in the library follows this pattern. The `Auto*` classes simply look up the correct class from the model config:

```python
# What AutoModelForCausalLM.from_pretrained() does internally:
# 1. Download config.json → read "model_type": "llama"
# 2. Look up MODEL_FOR_CAUSAL_LM_MAPPING: "llama" → LlamaForCausalLM
# 3. Instantiate LlamaForCausalLM(config)
# 4. Load safetensors weights into the model
```

## Model Anatomy: LlamaForCausalLM

```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)           # token embeddings
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(4096, 4096, bias=False)     # Q projection
          (k_proj): Linear(4096, 1024, bias=False)     # K projection (GQA: fewer heads)
          (v_proj): Linear(4096, 1024, bias=False)     # V projection (GQA)
          (o_proj): Linear(4096, 4096, bias=False)     # output projection
          (rotary_emb): LlamaRotaryEmbedding()         # RoPE
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(4096, 14336, bias=False) # SwiGLU gate
          (up_proj): Linear(4096, 14336, bias=False)   # SwiGLU up
          (down_proj): Linear(14336, 4096, bias=False) # SwiGLU down
        )
        (input_layernorm): LlamaRMSNorm(4096)          # pre-attention norm
        (post_attention_layernorm): LlamaRMSNorm(4096) # pre-FFN norm
      )
    )
    (norm): LlamaRMSNorm(4096)                         # final norm
  )
  (lm_head): Linear(4096, 128256, bias=False)          # maps to vocab logits
)
```

## Forward Pass Walkthrough

<div class="tensor-shape">
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Input token IDs:</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After embed_tokens:</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim feature">4096</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">× 32 LlamaDecoderLayers:</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim feature">4096</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After final norm:</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim feature">4096</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After lm_head:</span>
    <span class="ts-dim batch">B</span> × <span class="ts-dim seq">T</span> × <span class="ts-dim vocab">128256</span>
  </div>
</div>
</div>

```python
# Simplified forward of LlamaForCausalLM
def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None):
    # 1. Embeddings
    hidden_states = self.model.embed_tokens(input_ids)  # [B, T] → [B, T, d]
    
    # 2. Decoder layers
    for layer in self.model.layers:
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    
    # 3. Final norm + lm_head
    hidden_states = self.model.norm(hidden_states)       # [B, T, d]
    logits = self.lm_head(hidden_states)                 # [B, T, vocab_size]
    
    # 4. Loss (if labels provided)
    loss = None
    if labels is not None:
        # Shift: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )
    
    return CausalLMOutputWithPast(loss=loss, logits=logits,
                                   past_key_values=past_key_values)
```

## KV-Cache in transformers

During generation, the model caches key/value tensors to avoid recomputation:

```python
# First call (prefill): process full prompt
outputs = model(input_ids, use_cache=True)
past_key_values = outputs.past_key_values  # DynamicCache object

# Subsequent calls (decode): process one new token at a time
for _ in range(max_new_tokens):
    outputs = model(
        next_token_id.unsqueeze(0),         # [1, 1] — single token
        past_key_values=past_key_values,    # reuse cached K, V
        use_cache=True,
    )
    past_key_values = outputs.past_key_values  # updated cache
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
```

The `past_key_values` is a `DynamicCache` containing per-layer K and V tensors:

```python
# Inspect cache structure
cache = outputs.past_key_values
print(len(cache))              # 32 (one per layer)
print(cache[0][0].shape)       # K: [B, n_kv_heads, T_cached, head_dim]
print(cache[0][1].shape)       # V: [B, n_kv_heads, T_cached, head_dim]
```

## Extending transformers: Custom Models

Register a custom architecture:

```python
from transformers import PreTrainedModel, PretrainedConfig

class MyConfig(PretrainedConfig):
    model_type = "my_transformer"
    def __init__(self, d_model=1024, n_layers=12, n_heads=16, vocab_size=32000, **kwargs):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        super().__init__(vocab_size=vocab_size, **kwargs)

class MyModel(PreTrainedModel):
    config_class = MyConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

# Register and use
MyConfig.register_for_auto_class()
MyModel.register_for_auto_class("AutoModelForCausalLM")

# Now AutoModelForCausalLM.from_pretrained works with your model
```

## Attention Implementations

transformers supports multiple attention backends, selectable via `attn_implementation`:

| Backend | Key | Speed | Memory | Requirements |
|---------|-----|-------|--------|-------------|
| `"eager"` | Manual PyTorch | Baseline | O(T²) | Always available |
| `"sdpa"` | `F.scaled_dot_product_attention` | 2× | Depends on backend | PyTorch ≥ 2.0 |
| `"flash_attention_2"` | Flash Attention 2 | 2–4× | O(T) | flash-attn package |

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    attn_implementation="flash_attention_2",  # or "sdpa", "eager"
    torch_dtype=torch.bfloat16,
)
```

## What's Next

How do we know if a model is actually good? The next chapter covers **evaluation and benchmarks** — the metrics, datasets, and leaderboards used to measure LLM quality.

[← Previous: Chapter 24 — The Hugging Face Ecosystem](./24_huggingface_ecosystem.md) · **Next: [Chapter 26 — Evaluation & Benchmarks →](./26_evaluation_and_benchmarks.md)**

---

*Last updated: April 2026*
