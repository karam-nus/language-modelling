---
title: "Chapter 13 — Fine-Tuning & Adaptation"
description: "Supervised fine-tuning that turns a base model into an assistant: chat templates and assistant-only loss, how much data you need, reasoning SFT by distillation, full fine-tuning vs LoRA/QLoRA with honest memory numbers, LoRA's 2025 'without regret' findings, model merging, and a working TRL recipe."
---

[← Back to Table of Contents](./README.md)

# Chapter 13 — Fine-Tuning & Adaptation

A base model continues text. An assistant answers questions, follows formats, uses tools and knows when to stop. **Supervised fine-tuning (SFT)** is the first and cheapest step between the two.

> **The one-sentence version:** SFT is next-token training on example conversations with the loss restricted to the assistant's turns; it mostly teaches *format and behaviour* — which is why a few thousand excellent examples can matter more than a million mediocre ones.

## Supervised Fine-Tuning (SFT)

SFT trains a base model on conversations rendered through a **chat template**, with the cross-entropy loss computed **only on the assistant's tokens**: the model should learn to produce answers, not to imitate users or system prompts.

<div class="diagram">
<div class="diagram-title">SFT Loss Masking</div>
<div class="flow-h">
  <div class="flow-node purple">System: You are a helpful assistant</div>
  <div class="flow-node orange">User: What is attention?</div>
  <div class="flow-node green">Assistant: Attention is a mechanism that…</div>
</div>
<div class="flow-caption">↓ loss computation ↓</div>
<div class="flow-h">
  <div class="flow-node purple faint">masked (label −100)</div>
  <div class="flow-node orange faint">masked (label −100)</div>
  <div class="flow-node green">loss computed here ✓</div>
</div>
</div>

### Chat Templates

Every instruct model defines a template — special tokens marking roles and turn boundaries — stored in its tokenizer config as a Jinja template. Training and inference must use the same one:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
msgs = [{"role": "user", "content": "What is attention?"},
        {"role": "assistant", "content": "A weighted average over tokens."}]
print(tok.apply_chat_template(msgs, tokenize=False))
# <|im_start|>user
# What is attention?<|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# A weighted average over tokens.<|im_end|>
```

Note the empty `<think>` block: Qwen3's template inserts it for assistant turns without reasoning, so that the same model can be trained and run with or without a visible thinking segment. Templates now also define how **tool calls** and **tool results** are serialised; a mismatch between the template used in fine-tuning and the one used by your serving stack is one of the most common causes of "the fine-tune works in my notebook and not in production".

<div class="callout warn">
<span class="callout-title">Check what your trainer masks</span>
<p>Libraries differ in their defaults. In recent TRL versions, prompt–completion datasets mask the prompt automatically, but conversational <code>messages</code> datasets need <code>assistant_only_loss=True</code> (and a template that marks assistant spans) — otherwise the loss includes user and system tokens. Print one batch's labels and confirm the −100s are where you expect.</p>
</div>

### How Much Data, and What Kind?

| Finding | Evidence | Implication |
|---|---|---|
| A little high-quality data goes far | LIMA (2023): 1,000 curated examples | Spend effort on quality and diversity, not volume |
| Behaviour is mostly *elicited*, not taught | "Superficial alignment hypothesis"; base-model knowledge sets the ceiling | SFT cannot add much knowledge; use CPT or RAG for that |
| Reasoning can be distilled | DeepSeek-R1 distilled 800K reasoning samples into Qwen and Llama bases; s1 (2025): 1,000 long traces + "budget forcing" | Long chain-of-thought SFT from a strong teacher is the cheapest way to make a small reasoning model |
| Mixing matters | Tulu 3, Llama 3 post-training | Balance chat, maths, code, safety, tool use; deduplicate and decontaminate |

Representative SFT datasets are listed in [Chapter 9](./09_data_for_llms.md). In 2026 almost all of them are synthetic or model-filtered, and many include reasoning traces and tool-call transcripts.

## Full Fine-Tuning vs Parameter-Efficient Fine-Tuning

Full fine-tuning updates every weight and needs the full optimizer state. **LoRA** (Hu et al., 2021) freezes the weights and learns a low-rank update for selected matrices instead:

$$W' = W + \frac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r},\; r \ll \min(d, k)$$

B starts at zero, so training begins exactly at the base model. [Chapter 14](./14_peft_lora_and_variants.md) has the full mathematics and every variant; here is what matters for choosing.

<div class="diagram">
<div class="diagram-title">LoRA on one projection (Llama 3.1 8B W_q, r = 16)</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Full fine-tuning</div>
    <ul>
      <li>Update W directly: [4096, 4096]</li>
      <li>16.8M trainable parameters</li>
      <li>Optimizer state for all of them</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">LoRA (r = 16)</div>
    <ul>
      <li>Freeze W; train A [16, 4096] and B [4096, 16]</li>
      <li>65,536 + 65,536 = 131,072 parameters</li>
      <li>128× fewer for this matrix</li>
    </ul>
  </div>
</div>
</div>

### LoRA Size on a Real Model

Applying LoRA to **all seven linear projections** in every layer of Llama 3.1 8B (q, k, v, o, gate, up, down), each layer contributes r × (sum of in + out dimensions) = r × 81,920 parameters:

| Rank r | Trainable params (all linear layers, 32 layers) | Share of 8B |
|:---:|:---:|:---:|
| 8 | 21.0M | 0.26% |
| 16 | 41.9M | 0.52% |
| 32 | 83.9M | 1.0% |
| 64 | 167.8M | 2.1% |
| 128 | 335.5M | 4.2% |

### Memory, Honestly

| Setup (Llama 3.1 8B) | Weights | Grads + optimizer | Typical total with activations (2K tokens, checkpointing) |
|---|---|---|---|
| Full FT, AdamW, mixed precision | 16 GB BF16 + 32 GB FP32 master | 16 GB grads + 64 GB m, v | **~130 GB** — multiple GPUs with FSDP |
| LoRA r = 16 on BF16 base | 16 GB | ~0.6 GB (adapters, grads, Adam) | **~20–24 GB** — one 24–48 GB GPU |
| QLoRA r = 16 on NF4 base | ~5.6 GB (linear layers NF4; embedding and head stay BF16) | ~0.6 GB | **~9–12 GB** — fits a 16–24 GB consumer GPU |

Estimate your own configuration — model, method, sequence length, batch size, checkpointing:

<div class="lab" data-lab="vram"><noscript>Interactive lab: GPU memory calculator for full fine-tuning, LoRA and QLoRA, with weights, gradients, optimizer state and activations broken out. Enable JavaScript to use it.</noscript></div>

### Does LoRA Cost Quality?

Two careful studies frame the answer:

- **"LoRA Learns Less and Forgets Less"** (Biderman et al., 2024): on large-scale continued pre-training for code and maths, LoRA underperformed full fine-tuning — but also degraded the base model's other abilities less.
- **"LoRA Without Regret"** (Thinking Machines, 2025): for typical post-training dataset sizes, LoRA matched full fine-tuning *when applied to all layers including the MLPs* and run with a learning rate about 10× higher than full fine-tuning's; attention-only LoRA underperformed. For RL, even rank 1 matched full fine-tuning, because policy-gradient updates carry very few bits per episode.

| Factor | Full fine-tuning | LoRA / QLoRA |
|--------|:---:|:---:|
| Trainable parameters | 100% | 0.3–4% (all-linear) |
| Memory (8B) | ~130 GB | ~10–24 GB |
| Quality on post-training-sized data | Baseline | ≈ baseline with all-layer LoRA and tuned LR |
| Quality on CPT-sized data (billions of tokens) | Best | Lower capacity, but less forgetting |
| Serving many variants | One full model each | Hot-swap adapters on one base (multi-LoRA serving) |
| Merge into base? | N/A | Yes: W ← W + (α/r)BA |

## Model Merging

Fine-tuned models that share a base can often be **merged** by arithmetic on their weights, without further training:

- **Weight averaging / model soups**: average checkpoints or fine-tunes of the same base; Llama 3 averaged models from different post-training runs.
- **Task arithmetic, TIES, DARE**: add the *differences* from the base ("task vectors"), resolving sign conflicts (TIES) or randomly dropping and rescaling deltas (DARE) to reduce interference.
- Labs use merging routinely to combine specialist post-trained models and to recover capabilities lost in one training stage; the open community uses it to combine fine-tunes on the Hub.

## Fine-Tuning with TRL

[TRL](https://github.com/huggingface/trl) is Hugging Face's library for SFT, preference optimisation and RL. A QLoRA SFT run:

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

base = "meta-llama/Llama-3.1-8B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
tok = AutoTokenizer.from_pretrained(base)

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules="all-linear")          # all projections, as LoRA-without-regret advises

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")   # "messages" column

cfg = SFTConfig(
    output_dir="./llama-qlora-sft",
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=2e-4,                  # LoRA LRs are ~10x full-FT LRs
    num_train_epochs=1, max_length=2048,
    packing=True,                        # pack short conversations (use a FlashAttention-style backend so packed
                                         # examples cannot attend to each other)
    assistant_only_loss=True,            # loss on assistant turns only; requires a chat template with
                                         # generation markers (a Jinja "generation" block) — else use prompt/completion data
    gradient_checkpointing=True, bf16=True,
    logging_steps=10, save_steps=500,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=peft_cfg)
trainer.train()
trainer.save_model("./llama-qlora-sft/adapter")              # saves only the adapter (~170 MB)
```

To ship a single checkpoint, reload the base in BF16, attach the adapter and call `merge_and_unload()`; merging directly into 4-bit weights loses precision.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>If SFT mostly elicits behaviour the base model already has, why does distilling 800K reasoning traces into a 7B model make it dramatically better at maths? Is that new knowledge, a new <em>procedure</em>, or just a new <em>format</em> for knowledge it already had?</p>
</div>

## Key Takeaways

- SFT = next-token loss on assistant tokens of templated conversations; templates (including thinking and tool-call formats) must match between training and serving.
- Quality and diversity beat volume; SFT shapes behaviour but adds little knowledge. Long-CoT distillation is the cheap path to small reasoning models.
- Full fine-tuning of an 8B model needs ~130 GB; LoRA ~20 GB; QLoRA ~10 GB.
- Apply LoRA to all linear layers with a higher learning rate; it then matches full fine-tuning on post-training-sized data (and in RL even at tiny rank), while forgetting less.
- Model merging (averaging, TIES, DARE) combines fine-tunes of a shared base without retraining.

## Check Yourself

<details class="check">
<summary>LoRA with r = 32 on all linear layers of Llama 3.1 8B: how many trainable parameters, and how much BF16 storage for the adapter?</summary>
<p>32 × 81,920 × 32 layers = 83.9M parameters; × 2 bytes ≈ 168 MB.</p>
</details>

<details class="check">
<summary>Why is B initialised to zero in LoRA?</summary>
<p>So that BA = 0 at step 0 and the adapted model is exactly the base model. A is random, so gradients flow to B immediately; if both were zero, the gradient to each would be zero and nothing would train.</p>
</details>

<details class="check">
<summary>You fine-tuned with the loss on all tokens, including user turns. What behaviour might you see?</summary>
<p>The model partly learns to imitate users: it may continue a conversation by writing the next user turn, echo system-prompt phrasing, or spend capacity on predicting prompts rather than answers. Masking prompt tokens avoids this.</p>
</details>

<details class="check">
<summary>Why does QLoRA keep the embedding table and LM head in BF16 even though they are large?</summary>
<p>They are not ordinary linear layers in the quantization pass (embeddings are lookups; the head is usually excluded), and they are sensitive: quantization error in the output projection directly distorts the logits. For Llama 3.1 8B they add ~2.1 GB in BF16.</p>
</details>

<details class="check">
<summary>When would you choose full fine-tuning over LoRA in 2026?</summary>
<p>When training on billions of tokens (continued pre-training or very large SFT/distillation) where LoRA's capacity limits quality; when you need to change the model deeply (e.g., new language or modality); or when you have the compute and want one fully owned checkpoint. For typical instruction or RL fine-tunes, all-layer LoRA is usually as good and far cheaper.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al., 2021 | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) | LoRA |
| QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers et al., 2023 | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314) | NF4, double quantization, paged optimizers |
| LIMA: Less Is More for Alignment | Zhou et al., 2023 | [arxiv.org/abs/2305.11206](https://arxiv.org/abs/2305.11206) | Data quality over quantity |
| LoRA Learns Less and Forgets Less | Biderman et al., 2024 | [arxiv.org/abs/2405.09673](https://arxiv.org/abs/2405.09673) | LoRA vs full FT at CPT scale |
| LoRA Without Regret | Schulman et al. (Thinking Machines), 2025 | [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) | All-layer LoRA, LR scaling, RL at low rank |
| DeepSeek-R1 | DeepSeek-AI, 2025 | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) | Reasoning distillation into small models |
| s1: Simple test-time scaling | Muennighoff et al., 2025 | [arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393) | 1,000-example reasoning SFT |
| Editing Models with Task Arithmetic | Ilharco et al., 2022 | [arxiv.org/abs/2212.04089](https://arxiv.org/abs/2212.04089) | Task vectors |
| TIES-Merging | Yadav et al., 2023 | [arxiv.org/abs/2306.01708](https://arxiv.org/abs/2306.01708) | Interference-aware merging |
| TRL documentation: SFTTrainer | Hugging Face, 2025 | [huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer) | Assistant-only loss, packing |

---

**Next:** [Chapter 14 — PEFT: LoRA, QLoRA & Variants](./14_peft_lora_and_variants.md) — the mathematics of low-rank adaptation and its many descendants.

[← Back to Table of Contents](./README.md)
