---
title: "Chapter 28 — The Hugging Face Ecosystem"
description: "The Hub (models, datasets, Spaces, safetensors, GGUF, Xet storage, gated repos, inference providers), transformers v5 as the PyTorch model-definition layer, tokenizers, datasets, PEFT, TRL, Accelerate, kernels, smolagents and lighteval — with current, working code."
---

[← Back to Table of Contents](./README.md)

# Chapter 28 — The Hugging Face Ecosystem

Most open models, datasets and fine-tunes pass through Hugging Face at some point: published on its Hub, defined in its `transformers` library, trained with its PEFT and TRL libraries, and served by engines that read its checkpoint format. Knowing the pieces saves a great deal of glue code.

> **The one-sentence version:** the Hub stores and versions everything, `transformers` defines the models, and a ring of focused libraries — tokenizers, datasets, PEFT, TRL, Accelerate, kernels — handles each stage of the lifecycle around them.

## Ecosystem Overview

<div class="diagram">
<div class="diagram-title">Hugging Face Stack (2026)</div>
<div class="layer-stack">
  <div class="layer orange">🤗 Hub — millions of models and hundreds of thousands of datasets; Spaces for demos; Xet-backed large-file storage</div>
  <div class="layer accent">transformers — model definitions for hundreds of architectures; PyTorch-only from v5</div>
  <div class="layer green">tokenizers — fast Rust BPE / WordPiece / Unigram</div>
  <div class="layer purple">datasets — Arrow-backed, memory-mapped, streaming</div>
  <div class="layer red">PEFT — LoRA and its variants, multi-adapter loading</div>
  <div class="layer cyan">TRL — SFT, DPO, GRPO and online RL trainers</div>
  <div class="layer yellow">Accelerate — device placement, multi-GPU, FSDP/DeepSpeed glue</div>
  <div class="layer teal">kernels · lighteval · smolagents · LeRobot · Gradio · TGI and integrations with vLLM / SGLang / llama.cpp / MLX</div>
</div>
</div>

## The Hub

```bash
hf auth login                                                  # the `hf` CLI (formerly huggingface-cli)
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-instruct
hf upload my-org/my-model ./my_model                           # create or update a repo
```

```python
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(search="qwen3", sort="downloads", limit=5):
    print(m.id, m.downloads)
```

Key Hub features:
- **Model cards** — training data, intended use, evaluation, licence; the first thing to read.
- **Gated models** — accept a licence before download (Llama, Gemma and others).
- **safetensors** — the default weight format: no arbitrary code on load (unlike pickle), memory-mappable, fast. Prefer it; be careful with repos that require `trust_remote_code=True`, which runs Python from the repo.
- **GGUF, MLX and quantized checkpoints** — llama.cpp, Apple MLX and compressed-tensors/GPTQ/AWQ variants live alongside the originals.
- **Xet storage** — chunk-level deduplicated storage (rolled out 2025), so uploading a fine-tune that shares most bytes with its base moves only what changed.
- **Inference Providers** — call many hosted models through one OpenAI-compatible API.

## transformers

`transformers` is the reference implementation for most open architectures — the definition other engines (vLLM, SGLang, llama.cpp converters, MLX) port from or call into. **Version 5** (2025–26) made it PyTorch-only, dropped the legacy TensorFlow and Flax back-ends, and modernised defaults (`dtype=` instead of `torch_dtype=`, fast tokenizers everywhere, a standard attention interface and cache API).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto",      # spread layers over available GPUs
    attn_implementation="sdpa",                             # or "flash_attention_2", "flex_attention"
)

messages = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain attention in one paragraph."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Without `add_generation_prompt=True` the prompt ends after the user turn and the model may continue *as the user* — the most common chat-template mistake.

### Auto Classes

| Auto class | Purpose | Example models |
|-----------|---------|---------------|
| `AutoModel` | Base model, no head | Feature extraction, embeddings |
| `AutoModelForCausalLM` | Decoder-only generation | Llama, Qwen, Mistral, gpt-oss |
| `AutoModelForSeq2SeqLM` | Encoder-decoder | T5, BART |
| `AutoModelForSequenceClassification` | Classification / reward models | ModernBERT + head, reward models |
| `AutoModelForTokenClassification` | NER, tagging | BERT-style + token head |
| `AutoModelForImageTextToText` | Vision-language models | Qwen2.5-VL, Gemma 3, Llama 4 |
| `AutoModelForSpeechSeq2Seq` | Speech recognition | Whisper |

### Generation Config

Every model ships a `generation_config.json` with the lab's recommended defaults; override per call:

```python
from transformers import GenerationConfig

cfg = GenerationConfig(max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                       stop_strings=["</answer>"])
out = model.generate(**inputs, generation_config=cfg, tokenizer=tok)   # stop_strings needs the tokenizer
```

## Pipelines

```python
from transformers import pipeline

chat = pipeline("text-generation", model="HuggingFaceTB/SmolLM3-3B", dtype="auto", device_map="auto")
print(chat([{"role": "user", "content": "One sentence on why attention works."}],
           max_new_tokens=60)[0]["generated_text"][-1]["content"])

ner        = pipeline("token-classification", aggregation_strategy="simple")
asr        = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
```

Pipelines accept chat messages directly and apply the chat template for you.

## Trainer

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./checkpoints", num_train_epochs=3,
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-5, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=500,
    save_strategy="steps", save_steps=500, report_to="wandb",
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator, processing_class=tok)        # `tokenizer=` was renamed
trainer.train()
```

## Companion Libraries

### PEFT

```python
from peft import LoraConfig, get_peft_model

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196
```

### TRL

```python
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer

SFTTrainer(model=model, args=SFTConfig(output_dir="sft"), train_dataset=chat_ds).train()
DPOTrainer(model=model, args=DPOConfig(output_dir="dpo", beta=0.1),          # beta lives in the config
           train_dataset=pref_ds, processing_class=tok).train()
GRPOTrainer(model=model, reward_funcs=[my_reward], args=GRPOConfig(output_dir="grpo"),
            train_dataset=prompts_ds).train()
```

Chapters 13–15 have complete recipes.

### Accelerate

```python
from accelerate import Accelerator

acc = Accelerator(mixed_precision="bf16")
model, optimizer, loader = acc.prepare(model, optimizer, loader)
for batch in loader:
    loss = model(**batch).loss
    acc.backward(loss)
    optimizer.step(); optimizer.zero_grad()
# accelerate launch --num_processes 8 train.py   (FSDP or DeepSpeed via `accelerate config`)
```

### datasets

```python
from datasets import load_dataset

web = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)   # no full download
for ex in web.take(3):
    print(len(tok(ex["text"]).input_ids))
```

### And More

| Library | What it does |
|---|---|
| **kernels** (2025) | Load optimised compute kernels (attention, MoE, quantized GEMMs) from the Hub at runtime instead of compiling them |
| **lighteval** | Fast evaluation across many benchmarks, with vLLM/SGLang back-ends |
| **smolagents** | Minimal agents that act by writing Python code (Chapter 37) |
| **LeRobot** | Datasets, policies (including VLAs) and tools for real robots |
| **Gradio / Spaces** | Demos and apps; ZeroGPU Spaces lend shared GPUs on demand |
| **TGI** | Hugging Face's serving engine; Inference Endpoints also offer vLLM and SGLang (Chapter 31) |

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>When vLLM, SGLang, llama.cpp and MLX all port their model code from `transformers`, the library becomes the de facto specification of each architecture. What happens when an implementation detail — a RoPE convention, an attention-sink term, a norm epsilon — differs between the "spec" and a fast engine? How would you catch it?</p>
</div>

## Key Takeaways

- The Hub is the distribution layer: read model cards, prefer safetensors, beware `trust_remote_code`; Xet storage makes derived uploads cheap.
- transformers v5 is PyTorch-only and the reference model definition for the ecosystem; use `dtype=`, `processing_class=` and `add_generation_prompt=True`.
- PEFT (LoRA), TRL (SFT/DPO/GRPO), Accelerate (distribution) and datasets (streaming) cover fine-tuning end to end.
- Newer pieces — kernels, lighteval, smolagents, LeRobot — extend the stack to kernels, evaluation, agents and robotics.

## Check Yourself

<details class="check">
<summary>Why is safetensors safer than PyTorch's .bin checkpoints?</summary>
<p>.bin files are Python pickles, which can execute arbitrary code when loaded. safetensors stores only a JSON header and raw tensor bytes, so loading cannot run code; it is also memory-mappable for fast, lazy loading.</p>
</details>

<details class="check">
<summary>A chat model loaded with transformers answers by writing another user question. What did the code probably miss?</summary>
<p><code>add_generation_prompt=True</code> in <code>apply_chat_template</code>, so the prompt did not end with the assistant header and the model continued the user's turn.</p>
</details>

<details class="check">
<summary>What does <code>device_map="auto"</code> do, and when is it the wrong tool?</summary>
<p>It places layers across available GPUs (and CPU/disk if needed) so a large model fits, running them sequentially — pipeline placement with only one GPU active at a time. It is convenient for inference experiments but slow; for serving use vLLM/SGLang tensor parallelism, and for training use FSDP or DeepSpeed.</p>
</details>

<details class="check">
<summary>Why does <code>stop_strings</code> in <code>generate</code> require passing the tokenizer?</summary>
<p>Stopping is checked on token IDs during generation, but a stop string can be split across tokens in many ways; transformers uses the tokenizer to precompute which token sequences can complete the string.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Transformers: State-of-the-Art Natural Language Processing | Wolf et al., 2020 | [aclanthology.org/2020.emnlp-demos.6](https://aclanthology.org/2020.emnlp-demos.6/) | The library's design |
| Transformers documentation | Hugging Face | [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) | v5 APIs, chat templates, generation |
| Hub documentation | Hugging Face | [huggingface.co/docs/hub](https://huggingface.co/docs/hub) | Model cards, gating, Xet storage |
| safetensors | Hugging Face | [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors) | Weight format |
| PEFT · TRL · Accelerate · datasets docs | Hugging Face | [huggingface.co/docs](https://huggingface.co/docs) | Library APIs |
| kernels | Hugging Face, 2025 | [github.com/huggingface/kernels](https://github.com/huggingface/kernels) | Hub-distributed kernels |

---

**Next:** [Chapter 29 — Transformers Library Deep Dive](./29_transformers_library.md) — inside the library: configs, model classes, attention back-ends and caches.

[← Back to Table of Contents](./README.md)
