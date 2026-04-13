---
title: "Language Modelling — The Complete Guide"
permalink: /
---

# 🧠 Language Modelling — The Complete Guide

> **From Tokens to Frontier**: Understanding, building, and deploying large language models. A comprehensive learning path for ML practitioners navigating the generative AI landscape.

## Who This Is For

You're comfortable with Python and basic machine learning. You've heard about transformers, GPT, and LLMs — maybe even used them — but you want to truly understand **how they work under the hood**: from tokenization and attention mechanisms to quantization, CUDA kernels, and production deployment. This guide takes you from first principles all the way to the research frontier.

## 📋 Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| **Foundations** | | |
| 1 | [Introduction to Language Modelling](./01_introduction.md) | The full pipeline — text to tokens to tensors to predictions. Brief history, tokenization essentials, and how everything connects |
| 2 | [Embeddings & Representations](./02_embeddings.md) | From one-hot to dense vectors, positional encodings (sinusoidal, RoPE, ALiBi), and the geometry of meaning |
| 3 | [The Transformer](./03_the_transformer.md) | Complete architecture walkthrough with tensor shapes at every layer — residuals, norms, feed-forward networks |
| 4 | [Attention — SDPA & Multi-Head](./04_attention_sdpa_and_mha.md) | Q/K/V math, scaled dot-product attention, multi-head mechanism, causal masking, Flash Attention |
| **Attention & Architecture** | | |
| 5 | [Attention Variants — GQA, MQA & MLA](./05_attention_gqa_mqa_mla.md) | The KV bottleneck, grouped-query attention, multi-latent attention, sliding window — with tensor shape comparisons |
| 6 | [Decoder-Only Models](./06_decoder_only_models.md) | GPT family, LLaMA architecture, model size tables, open-source landscape |
| 7 | [Encoder & Seq2Seq Models](./07_encoder_and_seq2seq_models.md) | BERT, T5, BART — when to use encoder vs decoder vs encoder-decoder |
| 8 | [Multimodal Models](./08_multimodal_models.md) | Vision-language models, ViT, Whisper, modality fusion, image-to-token pipelines |
| **Training** | | |
| 9 | [Data for LLMs & VLMs](./09_data_for_llms.md) | Pre-training, mid-training, SFT, RLHF, RL data — formats, tensor shapes, and representative datasets for every phase |
| 10 | [Pre-Training at Scale](./10_pretraining_at_scale.md) | Data curation, training objectives, learning rate schedules, loss curves, datasets |
| 11 | [Optimizers & Loss Functions](./11_optimizers_and_loss_functions.md) | Cross-entropy, KD loss, DPO loss; AdamW, Adam-mini, Muon, SOAP — the machinery that drives LLM training |
| 12 | [Mid-Training & Continued Pre-Training](./12_mid_training.md) | Long-context extension, domain adaptation, data mixing, annealing — the bridge between pre-training and fine-tuning |
| 13 | [Fine-Tuning & Adaptation](./13_finetuning_and_adaptation.md) | SFT, LoRA, QLoRA, DoRA — parameter-efficient methods with rank decomposition math |
| 14 | [PEFT: LoRA, QLoRA & Variants](./14_peft_lora_and_variants.md) | Deep dive into LoRA math, QLoRA NF4 quantisation, DoRA, rsLoRA, VeRA, Adapters, Prefix Tuning, IA³ — with tensor shapes and selection guide |
| 15 | [Alignment — RLHF & Beyond](./15_alignment_rlhf_and_beyond.md) | Reward models, PPO, DPO, RLAIF, Constitutional AI — making models helpful and safe |
| **Inference & Generation** | | |
| 16 | [Inference & Sampling Strategies](./16_inference_and_sampling.md) | Temperature, top-k/p, beam search, speculative decoding, structured output |
| 17 | [KV-Cache — Mechanics & Memory](./17_kv_cache_mechanics.md) | How KV-cache works, tensor shapes through generation, memory calculations, worked examples for real models |
| 18 | [KV-Cache — Optimization Strategies](./18_kv_cache_optimization.md) | PagedAttention, continuous batching, eviction policies, KV-cache quantization, prefix caching |
| **Numerical Precision & Quantization** | | |
| 19 | [Data Types & Numerical Precision](./19_data_types_and_precision.md) | FP32 to FP4 — bit layouts, range vs precision, mixed precision training, BF16 vs FP16 |
| 20 | [Quantization Fundamentals](./20_quantization_fundamentals.md) | Affine math, symmetric vs asymmetric, calibration, weight-only vs weight-activation vs KV-cache quantization |
| 21 | [Quantization Techniques — Full Landscape](./21_quantization_techniques.md) | Every major method: GPTQ, AWQ, SmoothQuant, KIVI, BitNet + master comparison table |
| 22 | [Quantization Benchmarks & Selection](./22_quantization_benchmarks.md) | Perplexity vs bits, throughput benchmarks, decision flowchart, GGUF guide |
| 23 | [Knowledge Distillation & QAD](./23_knowledge_distillation.md) | Response/feature/relation distillation, sequence-level distillation, quantisation-aware distillation, fake quantisation with STE |
| **Hardware & Kernels** | | |
| 24 | [GPU Architecture for ML](./24_gpu_architecture.md) | SMs, Tensor Cores, memory hierarchy, roofline model, A100/H100/B200 comparison |
| 25 | [CUDA & Kernel Development](./25_cuda_and_kernel_development.md) | CUDA programming model, Triton kernels, Flash Attention design, fused operations |
| 26 | [Distributed Training](./26_distributed_training.md) | DDP, tensor/pipeline parallelism, FSDP, DeepSpeed ZeRO, 3D parallelism |
| 27 | [ASICs & Specialized Accelerators](./27_asics_and_accelerators.md) | TPUs, Groq, Trainium, Apple Silicon — when GPUs aren't the answer |
| **Ecosystem & Tooling** | | |
| 28 | [The Hugging Face Ecosystem](./28_huggingface_ecosystem.md) | Hub, datasets, tokenizers, Spaces — the open-source ML platform |
| 29 | [Transformers Library Deep Dive](./29_transformers_library.md) | AutoModel, config, Trainer, Pipeline — internals and patterns |
| 30 | [Evaluation & Benchmarks](./30_evaluation_and_benchmarks.md) | Perplexity, MMLU, HumanEval, lm-eval-harness, Chatbot Arena |
| 31 | [Serving & Deployment](./31_serving_and_deployment.md) | vLLM, TGI, TensorRT-LLM, Ollama, llama.cpp — from prototype to production |
| **Advanced Topics** | | |
| 32 | [Scaling Laws & Emergent Abilities](./32_scaling_laws.md) | Kaplan, Chinchilla, emergence debate — the science of "how big" |
| 33 | [Mixture of Experts](./33_mixture_of_experts.md) | Sparse routing, load balancing, Mixtral, DeepSeek-MoE — more params, same compute |
| 34 | [SSMs & Beyond Transformers](./34_ssms_and_alternatives.md) | Mamba, RWKV, Jamba — O(T) alternatives to quadratic attention |
| 35 | [Reasoning Models](./35_reasoning_models.md) | Chain-of-Thought, o1-style reasoning, process rewards, test-time compute scaling |
| **Applications & Frontier** | | |
| 36 | [Retrieval-Augmented Generation](./36_rag.md) | RAG pipeline, embedding models, vector databases, advanced retrieval |
| 37 | [Agents & Tool Use](./37_agents_and_tool_use.md) | Function calling, coding agents, computer use, MCP protocol |
| 38 | [The Frontier](./38_frontier.md) | Long context, world models, safety, interpretability — open problems |
| **Appendices** | | |
| A | [History of Language Modelling](./appendix_a_history.md) | Full timeline from Shannon to GPT-4 — every milestone, paper, and breakthrough |
| B | [Tokenization Deep Dive](./appendix_b_tokenization.md) | BPE step-by-step, WordPiece, SentencePiece, training tokenizers from scratch |
| C | [HuggingFace Model Config Field Reference](./appendix_c_model_config.md) | Every `config.json` parameter explained — values, formulae, caveats, VLM/VLA configs, and kernel pitfall checklist |

## 🗺️ Learning Path

<div class="diagram">
<div class="diagram-title">Recommended Learning Path</div>
<div class="flow">
  <div class="flow-node accent wide">📖 Ch 1–4: Foundations & Transformer</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🧩 Ch 5–8: Attention Variants & Architectures</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🏋️ Ch 9–15: Training Pipeline</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">⚡ Ch 16–18: Inference & KV-Cache</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">🔢 Ch 19–23: Precision, Quantization & Distillation</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">🖥️ Ch 24–27: Hardware & Kernels</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node yellow wide">🛠️ Ch 28–31: Ecosystem & Deployment</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🚀 Ch 32–38: Advanced & Frontier</div>
</div>
</div>

## ⚡ Quick Start Paths

### Path A: "I want to understand transformers" (5 chapters)

1. [01 — Introduction](./01_introduction.md) — the big picture
2. [03 — The Transformer](./03_the_transformer.md) — architecture deep dive
3. [04 — SDPA & Multi-Head Attention](./04_attention_sdpa_and_mha.md) — attention mechanics
4. [06 — Decoder-Only Models](./06_decoder_only_models.md) — GPT & LLaMA
5. [16 — Inference & Sampling](./16_inference_and_sampling.md) — how generation works

### Path B: "I want to train / fine-tune models" (7 chapters)

1. [09 — Data for LLMs & VLMs](./09_data_for_llms.md) — data across all training phases
2. [10 — Pre-Training at Scale](./10_pretraining_at_scale.md) — data and objectives
3. [11 — Optimizers & Loss Functions](./11_optimizers_and_loss_functions.md) — AdamW, schedules, losses
4. [12 — Mid-Training](./12_mid_training.md) — continued pre-training
5. [13 — Fine-Tuning & Adaptation](./13_finetuning_and_adaptation.md) — SFT & LoRA
6. [14 — PEFT: LoRA, QLoRA & Variants](./14_peft_lora_and_variants.md) — parameter-efficient methods
7. [26 — Distributed Training](./26_distributed_training.md) — scaling to multi-GPU

### Path C: "I want to deploy and optimize" (7 chapters)

1. [17 — KV-Cache Mechanics](./17_kv_cache_mechanics.md) — memory bottleneck
2. [18 — KV-Cache Optimization](./18_kv_cache_optimization.md) — PagedAttention & more
3. [19 — Data Types](./19_data_types_and_precision.md) — precision tradeoffs
4. [21 — Quantization Techniques](./21_quantization_techniques.md) — the full landscape
5. [23 — Knowledge Distillation & QAD](./23_knowledge_distillation.md) — compress with distillation
6. [29 — Transformers Library](./29_transformers_library.md) — practical tooling
7. [31 — Serving & Deployment](./31_serving_and_deployment.md) — vLLM, TGI, Ollama

### Path D: "I want deep understanding" (full guide)

Read chapters 1 through 38 in order. Each builds on the previous. Appendices A and B provide additional depth on history and tokenization.

## 📚 Prerequisites

Before diving in, you should be comfortable with:

- **Python** — NumPy, basic PyTorch tensor operations
- **Linear algebra** — matrix multiplication, dot products, transposes
- **ML basics** — loss functions, gradient descent, backpropagation
- **Probability** — softmax, cross-entropy, sampling from distributions
- **Command line** — terminal, pip/conda, environment variables

## 📝 Changelog

| Date | Changes |
|------|---------|
| April 2026 | Added Appendix C — HuggingFace Model Config Field Reference (LLM, VLM, VLA configs, kernel pitfall checklist) |
| April 2026 | Added Ch 9 (Data), Ch 11 (Optimizers & Loss Functions), Ch 14 (PEFT Deep Dive), Ch 23 (Knowledge Distillation & QAD) — renumbered all chapters accordingly |
| April 2026 | Initial release — 34 chapters + 2 appendices |

---

*Last updated: April 2026*
