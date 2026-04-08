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
| 9 | [Pre-Training at Scale](./09_pretraining_at_scale.md) | Data curation, training objectives, learning rate schedules, loss curves, datasets |
| 10 | [Mid-Training & Continued Pre-Training](./10_mid_training.md) | Long-context extension, domain adaptation, data mixing, annealing — the bridge between pre-training and fine-tuning |
| 11 | [Fine-Tuning & Adaptation](./11_finetuning_and_adaptation.md) | SFT, LoRA, QLoRA, DoRA — parameter-efficient methods with rank decomposition math |
| 12 | [Alignment — RLHF & Beyond](./12_alignment_rlhf_and_beyond.md) | Reward models, PPO, DPO, RLAIF, Constitutional AI — making models helpful and safe |
| **Inference & Generation** | | |
| 13 | [Inference & Sampling Strategies](./13_inference_and_sampling.md) | Temperature, top-k/p, beam search, speculative decoding, structured output |
| 14 | [KV-Cache — Mechanics & Memory](./14_kv_cache_mechanics.md) | How KV-cache works, tensor shapes through generation, memory calculations, worked examples for real models |
| 15 | [KV-Cache — Optimization Strategies](./15_kv_cache_optimization.md) | PagedAttention, continuous batching, eviction policies, KV-cache quantization, prefix caching |
| **Numerical Precision & Quantization** | | |
| 16 | [Data Types & Numerical Precision](./16_data_types_and_precision.md) | FP32 to FP4 — bit layouts, range vs precision, mixed precision training, BF16 vs FP16 |
| 17 | [Quantization Fundamentals](./17_quantization_fundamentals.md) | Affine math, symmetric vs asymmetric, calibration, weight-only vs weight-activation vs KV-cache quantization |
| 18 | [Quantization Techniques — Full Landscape](./18_quantization_techniques.md) | Every major method: GPTQ, AWQ, SmoothQuant, KIVI, BitNet + master comparison table |
| 19 | [Quantization Benchmarks & Selection](./19_quantization_benchmarks.md) | Perplexity vs bits, throughput benchmarks, decision flowchart, GGUF guide |
| **Hardware & Kernels** | | |
| 20 | [GPU Architecture for ML](./20_gpu_architecture.md) | SMs, Tensor Cores, memory hierarchy, roofline model, A100/H100/B200 comparison |
| 21 | [CUDA & Kernel Development](./21_cuda_and_kernel_development.md) | CUDA programming model, Triton kernels, Flash Attention design, fused operations |
| 22 | [Distributed Training](./22_distributed_training.md) | DDP, tensor/pipeline parallelism, FSDP, DeepSpeed ZeRO, 3D parallelism |
| 23 | [ASICs & Specialized Accelerators](./23_asics_and_accelerators.md) | TPUs, Groq, Trainium, Apple Silicon — when GPUs aren't the answer |
| **Ecosystem & Tooling** | | |
| 24 | [The Hugging Face Ecosystem](./24_hugging_face_ecosystem.md) | Hub, datasets, tokenizers, Spaces — the open-source ML platform |
| 25 | [Transformers Library Deep Dive](./25_transformers_library.md) | AutoModel, config, Trainer, Pipeline — internals and patterns |
| 26 | [Evaluation & Benchmarks](./26_evaluation_and_benchmarks.md) | Perplexity, MMLU, HumanEval, lm-eval-harness, Chatbot Arena |
| 27 | [Serving & Deployment](./27_serving_and_deployment.md) | vLLM, TGI, TensorRT-LLM, Ollama, llama.cpp — from prototype to production |
| **Advanced Topics** | | |
| 28 | [Scaling Laws & Emergent Abilities](./28_scaling_laws.md) | Kaplan, Chinchilla, emergence debate — the science of "how big" |
| 29 | [Mixture of Experts](./29_mixture_of_experts.md) | Sparse routing, load balancing, Mixtral, DeepSeek-MoE — more params, same compute |
| 30 | [SSMs & Beyond Transformers](./30_ssms_and_alternatives.md) | Mamba, RWKV, Jamba — O(T) alternatives to quadratic attention |
| 31 | [Reasoning Models](./31_reasoning_models.md) | Chain-of-Thought, o1-style reasoning, process rewards, test-time compute scaling |
| **Applications & Frontier** | | |
| 32 | [Retrieval-Augmented Generation](./32_retrieval_augmented_generation.md) | RAG pipeline, embedding models, vector databases, advanced retrieval |
| 33 | [Agents & Tool Use](./33_agents_and_tool_use.md) | Function calling, coding agents, computer use, MCP protocol |
| 34 | [The Frontier](./34_the_frontier.md) | Long context, world models, safety, interpretability — open problems |
| **Appendices** | | |
| A | [History of Language Modelling](./appendix_a_history.md) | Full timeline from Shannon to GPT-4 — every milestone, paper, and breakthrough |
| B | [Tokenization Deep Dive](./appendix_b_tokenization.md) | BPE step-by-step, WordPiece, SentencePiece, training tokenizers from scratch |

## 🗺️ Learning Path

<div class="diagram">
<div class="diagram-title">Recommended Learning Path</div>
<div class="flow">
  <div class="flow-node accent wide">📖 Ch 1–4: Foundations & Transformer</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🧩 Ch 5–8: Attention Variants & Architectures</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🏋️ Ch 9–12: Training Pipeline</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">⚡ Ch 13–15: Inference & KV-Cache</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">🔢 Ch 16–19: Precision & Quantization</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">🖥️ Ch 20–23: Hardware & Kernels</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node yellow wide">🛠️ Ch 24–27: Ecosystem & Deployment</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🚀 Ch 28–34: Advanced & Frontier</div>
</div>
</div>

## ⚡ Quick Start Paths

### Path A: "I want to understand transformers" (5 chapters)

1. [01 — Introduction](./01_introduction.md) — the big picture
2. [03 — The Transformer](./03_the_transformer.md) — architecture deep dive
3. [04 — SDPA & Multi-Head Attention](./04_attention_sdpa_and_mha.md) — attention mechanics
4. [06 — Decoder-Only Models](./06_decoder_only_models.md) — GPT & LLaMA
5. [13 — Inference & Sampling](./13_inference_and_sampling.md) — how generation works

### Path B: "I want to train / fine-tune models" (5 chapters)

1. [09 — Pre-Training at Scale](./09_pretraining_at_scale.md) — data and objectives
2. [10 — Mid-Training](./10_mid_training.md) — continued pre-training
3. [11 — Fine-Tuning & Adaptation](./11_finetuning_and_adaptation.md) — LoRA & friends
4. [12 — Alignment](./12_alignment_rlhf_and_beyond.md) — RLHF & DPO
5. [22 — Distributed Training](./22_distributed_training.md) — scaling to multi-GPU

### Path C: "I want to deploy and optimize" (6 chapters)

1. [14 — KV-Cache Mechanics](./14_kv_cache_mechanics.md) — memory bottleneck
2. [15 — KV-Cache Optimization](./15_kv_cache_optimization.md) — PagedAttention & more
3. [16 — Data Types](./16_data_types_and_precision.md) — precision tradeoffs
4. [18 — Quantization Techniques](./18_quantization_techniques.md) — the full landscape
5. [25 — Transformers Library](./25_transformers_library.md) — practical tooling
6. [27 — Serving & Deployment](./27_serving_and_deployment.md) — vLLM, TGI, Ollama

### Path D: "I want deep understanding" (full guide)

Read chapters 1 through 34 in order. Each builds on the previous. Appendices A and B provide additional depth on history and tokenization.

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
| April 2026 | Initial release — all 34 chapters + 2 appendices |

---

*Last updated: April 2026*
# language-modelling
LM 101
