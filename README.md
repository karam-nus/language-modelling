---
title: "Home"
description: "From tokens to the frontier: a first-principles guide to how large language models are built, trained, aligned, compressed and served — updated for late 2026, with tensor shapes on every page, verified code and 23 interactive labs."
permalink: /
---

# Language Modelling — The Complete Guide

<div class="home-hero">
<div class="hh-left">
<div class="hh-kicker">A field guide · edition of November 2026</div>
<h1 class="hh-title" aria-label="Language Modelling"><span class="ln ln-1">Language</span><span class="ln ln-2">Modelling</span><span class="ghost" aria-hidden="true"><span class="ln ln-1">Language</span><span class="ln ln-2">Modelling</span></span></h1>
<p class="hh-dek">How a machine that only ever predicts the next token became the most general tool we have built — taken apart layer by layer, from the tokenizer to the training cluster, with every tensor shape written down and every number checked.</p>
<div class="hh-cta">
<a class="primary" href="01_introduction.html">Start at token zero →</a>
<a href="labs.html">Open the lab bench</a>
<a href="#the-guide-drawn-as-a-model">See the map</a>
</div>
<div class="hh-stats">
<div class="hh-stat"><b>38</b><span>chapters</span></div>
<div class="hh-stat"><b>3</b><span>appendices</span></div>
<div class="hh-stat"><b>23</b><span>live labs</span></div>
<div class="hh-stat"><b>192</b><span>check-yourself Qs</span></div>
</div>
</div>
<div class="lab" data-lab="ngram" data-variant="hero"><noscript>Interactive lab: a tiny trigram language model trained on sentences from this guide, generating text one token at a time. Enable JavaScript to use it.</noscript></div>
</div>

<div class="home-section" markdown="0">
<div class="hs-head"><h2 id="who-this-is-for">Who this is for</h2><p>prerequisites: Python, a little linear algebra, curiosity</p></div>
<div class="newlist">
<div><b>You use LLMs</b>and want to know what actually happens between your prompt and the first token — and why the second token is so much cheaper than the first.</div>
<div><b>You train or fine-tune them</b>and want the memory, compute and data arithmetic before you rent the GPUs, plus the 2025–26 recipes: RL with verifiable rewards, Muon, FP8, mid-training.</div>
<div><b>You serve them</b>and need to reason about KV caches, quantization, batching, speculative decoding and $/million tokens from first principles, not folklore.</div>
</div>
</div>

<div class="home-section" markdown="0">
<div class="hs-head"><h2 id="the-guide-drawn-as-a-model">The guide, drawn as a model</h2><p>nine blocks on a residual stream · ✓ marks chapters you have read on this device</p></div>
<div class="modelmap">
<div class="mm-residual" aria-hidden="true"></div>
<div class="mm-io"><span>input tokens</span><div class="tokens-row"><span>you</span><span>know</span><span>Python</span><span>and</span><span>a</span><span>little</span><span>ML</span></div></div>
<div class="mm-block blue"><div class="mm-tag"><b>I</b><span>Foundations</span></div><div class="mm-body"><div class="mm-name">From text to tensors to the next token</div><div class="mm-chips"><a href="01_introduction.html" data-path="/01_introduction"><b>01</b>Introduction</a><a href="02_embeddings.html" data-path="/02_embeddings"><b>02</b>Embeddings &amp; positions</a><a href="03_the_transformer.html" data-path="/03_the_transformer"><b>03</b>The transformer</a><a href="04_attention_sdpa_and_mha.html" data-path="/04_attention_sdpa_and_mha"><b>04</b>Attention: SDPA &amp; MHA</a></div></div></div>
<div class="mm-block purple"><div class="mm-tag"><b>II</b><span>Architectures</span></div><div class="mm-body"><div class="mm-name">The shapes models actually ship in</div><div class="mm-chips"><a href="05_attention_gqa_mqa_mla.html" data-path="/05_attention_gqa_mqa_mla"><b>05</b>GQA, MQA, MLA &amp; sparse</a><a href="06_decoder_only_models.html" data-path="/06_decoder_only_models"><b>06</b>Decoder-only models</a><a href="07_encoder_and_seq2seq_models.html" data-path="/07_encoder_and_seq2seq_models"><b>07</b>Encoders &amp; seq2seq</a><a href="08_multimodal_models.html" data-path="/08_multimodal_models"><b>08</b>Multimodal</a></div></div></div>
<div class="mm-block green"><div class="mm-tag"><b>III</b><span>Training</span></div><div class="mm-body"><div class="mm-name">Data, pre-training, post-training, RL</div><div class="mm-chips"><a href="09_data_for_llms.html" data-path="/09_data_for_llms"><b>09</b>Data</a><a href="10_pretraining_at_scale.html" data-path="/10_pretraining_at_scale"><b>10</b>Pre-training</a><a href="11_optimizers_and_loss_functions.html" data-path="/11_optimizers_and_loss_functions"><b>11</b>Optimizers &amp; losses</a><a href="12_mid_training.html" data-path="/12_mid_training"><b>12</b>Mid-training</a><a href="13_finetuning_and_adaptation.html" data-path="/13_finetuning_and_adaptation"><b>13</b>Fine-tuning</a><a href="14_peft_lora_and_variants.html" data-path="/14_peft_lora_and_variants"><b>14</b>LoRA &amp; PEFT</a><a href="15_alignment_rlhf_and_beyond.html" data-path="/15_alignment_rlhf_and_beyond"><b>15</b>Alignment &amp; RL</a></div></div></div>
<div class="mm-block accent"><div class="mm-tag"><b>IV</b><span>Inference</span></div><div class="mm-body"><div class="mm-name">One token at a time, as fast as memory allows</div><div class="mm-chips"><a href="16_inference_and_sampling.html" data-path="/16_inference_and_sampling"><b>16</b>Sampling &amp; speculation</a><a href="17_kv_cache_mechanics.html" data-path="/17_kv_cache_mechanics"><b>17</b>KV cache: mechanics</a><a href="18_kv_cache_optimization.html" data-path="/18_kv_cache_optimization"><b>18</b>KV cache: optimization</a></div></div></div>
<div class="mm-block orange"><div class="mm-tag"><b>V</b><span>Precision</span></div><div class="mm-body"><div class="mm-name">Fewer bits, same answers</div><div class="mm-chips"><a href="19_data_types_and_precision.html" data-path="/19_data_types_and_precision"><b>19</b>Data types</a><a href="20_quantization_fundamentals.html" data-path="/20_quantization_fundamentals"><b>20</b>Quantization basics</a><a href="21_quantization_techniques.html" data-path="/21_quantization_techniques"><b>21</b>Techniques</a><a href="22_quantization_benchmarks.html" data-path="/22_quantization_benchmarks"><b>22</b>Choosing one</a><a href="23_knowledge_distillation.html" data-path="/23_knowledge_distillation"><b>23</b>Distillation</a></div></div></div>
<div class="mm-block teal"><div class="mm-tag"><b>VI</b><span>Hardware</span></div><div class="mm-body"><div class="mm-name">The silicon and the cluster underneath</div><div class="mm-chips"><a href="24_gpu_architecture.html" data-path="/24_gpu_architecture"><b>24</b>GPU architecture</a><a href="25_cuda_and_kernel_development.html" data-path="/25_cuda_and_kernel_development"><b>25</b>CUDA &amp; kernels</a><a href="26_distributed_training.html" data-path="/26_distributed_training"><b>26</b>Distributed training</a><a href="27_asics_and_accelerators.html" data-path="/27_asics_and_accelerators"><b>27</b>ASICs &amp; accelerators</a></div></div></div>
<div class="mm-block pink"><div class="mm-tag"><b>VII</b><span>Ecosystem</span></div><div class="mm-body"><div class="mm-name">Tools, evaluation and production</div><div class="mm-chips"><a href="28_huggingface_ecosystem.html" data-path="/28_huggingface_ecosystem"><b>28</b>Hugging Face</a><a href="29_transformers_library.html" data-path="/29_transformers_library"><b>29</b>transformers internals</a><a href="30_evaluation_and_benchmarks.html" data-path="/30_evaluation_and_benchmarks"><b>30</b>Evaluation</a><a href="31_serving_and_deployment.html" data-path="/31_serving_and_deployment"><b>31</b>Serving</a></div></div></div>
<div class="mm-block yellow"><div class="mm-tag"><b>VIII</b><span>Scale</span></div><div class="mm-body"><div class="mm-name">How big, how sparse, how linear</div><div class="mm-chips"><a href="32_scaling_laws.html" data-path="/32_scaling_laws"><b>32</b>Scaling laws</a><a href="33_mixture_of_experts.html" data-path="/33_mixture_of_experts"><b>33</b>Mixture of experts</a><a href="34_ssms_and_alternatives.html" data-path="/34_ssms_and_alternatives"><b>34</b>SSMs &amp; hybrids</a></div></div></div>
<div class="mm-block red"><div class="mm-tag"><b>IX</b><span>Reasoning</span></div><div class="mm-body"><div class="mm-name">Thinking, retrieving, acting</div><div class="mm-chips"><a href="35_reasoning_models.html" data-path="/35_reasoning_models"><b>35</b>Reasoning models</a><a href="36_rag.html" data-path="/36_rag"><b>36</b>RAG</a><a href="37_agents_and_tool_use.html" data-path="/37_agents_and_tool_use"><b>37</b>Agents &amp; tools</a><a href="38_frontier.html" data-path="/38_frontier"><b>38</b>The frontier</a></div></div></div>
<div class="mm-block teal"><div class="mm-tag"><b>A–C</b><span>Appendices</span></div><div class="mm-body"><div class="mm-name">History, tokenizers and the config file</div><div class="mm-chips"><a href="appendix_a_history.html" data-path="/appendix_a_history"><b>A</b>History</a><a href="appendix_b_tokenization.html" data-path="/appendix_b_tokenization"><b>B</b>Tokenization</a><a href="appendix_c_model_config.html" data-path="/appendix_c_model_config"><b>C</b>Model config reference</a></div></div></div>
<div class="mm-io"><span>output</span><div class="tokens-row"><span>p(next idea | everything you have read)</span></div></div>
</div>
</div>

<div class="home-section" markdown="0">
<div class="hs-head"><h2 id="the-lab-bench">The lab bench</h2><p>23 instruments · arithmetic checked against the chapters by tools/test_labs.js</p></div>
{% include labgrid.html %}
</div>

<div class="home-section" markdown="0">
<div class="hs-head"><h2 id="new-in-this-edition">New in this edition</h2><p>every chapter re-checked and dated November 2026</p></div>
<div class="newlist">
<div><b>Reasoning by reinforcement</b>RL with verifiable rewards, GRPO and its 2025 repairs (Dr. GRPO, DAPO, GSPO), thinking budgets and test-time compute — <a href="15_alignment_rlhf_and_beyond.html">Ch 15</a>, <a href="35_reasoning_models.html">35</a>.</div>
<div><b>Hybrid attention goes mainstream</b>Gated DeltaNet and Kimi Delta Attention mixed 3 : 1 with full attention, DeepSeek's sparse attention, and what a fixed state forgets — <a href="05_attention_gqa_mqa_mla.html">Ch 5</a>, <a href="34_ssms_and_alternatives.html">34</a>.</div>
<div><b>The 2026 open frontier</b>DeepSeek-V4, Kimi K3, Qwen3.5, GLM-5, Gemma 4 and gpt-oss, side by side with real configs — <a href="06_decoder_only_models.html">Ch 6</a>, <a href="33_mixture_of_experts.html">33</a>.</div>
<div><b>Four-bit everything</b>MXFP4 and NVFP4 for weights, activations, KV caches and even pre-training — <a href="19_data_types_and_precision.html">Ch 19</a>–<a href="22_quantization_benchmarks.html">22</a>.</div>
<div><b>New optimizers at scale</b>Muon and its Newton–Schulz iteration, with the evidence on when it beats AdamW — <a href="11_optimizers_and_loss_functions.html">Ch 11</a>.</div>
<div><b>Disaggregated serving</b>Prefill/decode split, KV-aware routing, llm-d and Dynamo, and the cost model behind $/M tokens — <a href="18_kv_cache_optimization.html">Ch 18</a>, <a href="31_serving_and_deployment.html">31</a>.</div>
<div><b>Agents that work for hours</b>Tool calling, MCP, context engineering, coding agents and how they are evaluated — <a href="37_agents_and_tool_use.html">Ch 37</a>, <a href="30_evaluation_and_benchmarks.html">30</a>.</div>
<div><b>Every page, the same spine</b>A one-sentence version, a "ponder" prompt, key takeaways, check-yourself questions with worked answers, and dated references.</div>
</div>
</div>

<div class="home-section" markdown="0">
<div class="hs-head"><h2 id="paths-through-the-guide">Paths through the guide</h2><p>read in order, or follow one route</p></div>
<div class="paths">
<div class="path"><h3>Understand how an LLM works</h3><ol><li><a href="01_introduction.html">Introduction</a></li><li><a href="02_embeddings.html">Embeddings &amp; positions</a></li><li><a href="03_the_transformer.html">The transformer</a></li><li><a href="04_attention_sdpa_and_mha.html">Attention</a></li><li><a href="16_inference_and_sampling.html">Sampling</a></li><li><a href="17_kv_cache_mechanics.html">KV cache</a></li></ol></div>
<div class="path"><h3>Fine-tune and align a model</h3><ol><li><a href="09_data_for_llms.html">Data</a></li><li><a href="11_optimizers_and_loss_functions.html">Optimizers &amp; losses</a></li><li><a href="13_finetuning_and_adaptation.html">Fine-tuning</a></li><li><a href="14_peft_lora_and_variants.html">LoRA &amp; PEFT</a></li><li><a href="15_alignment_rlhf_and_beyond.html">Alignment &amp; RL</a></li><li><a href="35_reasoning_models.html">Reasoning models</a></li></ol></div>
<div class="path"><h3>Serve a model cheaply</h3><ol><li><a href="17_kv_cache_mechanics.html">KV cache</a></li><li><a href="18_kv_cache_optimization.html">KV optimization</a></li><li><a href="20_quantization_fundamentals.html">Quantization</a></li><li><a href="22_quantization_benchmarks.html">Choosing a format</a></li><li><a href="24_gpu_architecture.html">GPU architecture</a></li><li><a href="31_serving_and_deployment.html">Serving</a></li></ol></div>
<div class="path"><h3>Train at scale</h3><ol><li><a href="10_pretraining_at_scale.html">Pre-training</a></li><li><a href="24_gpu_architecture.html">GPU architecture</a></li><li><a href="25_cuda_and_kernel_development.html">Kernels</a></li><li><a href="26_distributed_training.html">Distributed training</a></li><li><a href="32_scaling_laws.html">Scaling laws</a></li><li><a href="33_mixture_of_experts.html">Mixture of experts</a></li></ol></div>
</div>
</div>

## Contents

| # | Chapter | What you will learn |
|:-:|---------|---------------------|
| | **I · Foundations** | |
| 01 | [Introduction to Language Modelling](./01_introduction.md) | The whole pipeline — text → tokens → tensors → a distribution over the next token — and how today's models are built and trained |
| 02 | [Embeddings & Positional Encodings](./02_embeddings.md) | Embedding tables, sinusoidal positions, RoPE and its long-context extensions (NTK, YaRN, partial RoPE, NoPE layers) |
| 03 | [The Transformer](./03_the_transformer.md) | A modern pre-norm decoder block with every tensor shape: RMSNorm, SwiGLU, residuals, and where the parameters live |
| 04 | [Attention: SDPA & Multi-Head](./04_attention_sdpa_and_mha.md) | Q/K/V, scaled dot-product attention, masking, multi-head, FlashAttention, gated attention |
| | **II · Architectures** | |
| 05 | [GQA, MQA, MLA & Sparse Attention](./05_attention_gqa_mqa_mla.md) | Shrinking the KV cache: grouped and multi-latent attention, sliding windows, sparse and hybrid attention |
| 06 | [Decoder-Only Models](./06_decoder_only_models.md) | The model families of 2026, open and closed, with configs and the design choices that distinguish them |
| 07 | [Encoders & Seq2Seq](./07_encoder_and_seq2seq_models.md) | BERT to ModernBERT, T5, embedding and reranking models — when bidirectional still wins |
| 08 | [Multimodal Models](./08_multimodal_models.md) | Vision encoders, image tokens, M-RoPE, audio and speech, native multimodality |
| | **III · Training** | |
| 09 | [Data for LLMs & VLMs](./09_data_for_llms.md) | Every training phase's data, with formats, tensor shapes and representative datasets |
| 10 | [Pre-Training at Scale](./10_pretraining_at_scale.md) | Objectives, schedules (WSD), stability, FP8 training, and the compute budget |
| 11 | [Optimizers & Loss Functions](./11_optimizers_and_loss_functions.md) | Cross-entropy, distillation and preference losses; AdamW, Muon and friends |
| 12 | [Mid-Training](./12_mid_training.md) | Long-context extension, annealing, domain and reasoning mid-training |
| 13 | [Fine-Tuning & Adaptation](./13_finetuning_and_adaptation.md) | SFT, chat templates, loss masking, and the memory arithmetic of full vs LoRA vs QLoRA |
| 14 | [PEFT: LoRA & Variants](./14_peft_lora_and_variants.md) | LoRA's maths and practice, QLoRA, DoRA and the rest, and when rank matters |
| 15 | [Alignment: RLHF & Beyond](./15_alignment_rlhf_and_beyond.md) | Reward models, PPO, DPO, RL with verifiable rewards, GRPO and its successors |
| | **IV · Inference** | |
| 16 | [Inference & Sampling](./16_inference_and_sampling.md) | Prefill vs decode, sampling methods, structured output, speculative decoding |
| 17 | [KV Cache: Mechanics](./17_kv_cache_mechanics.md) | Shapes through generation and exact memory for real models |
| 18 | [KV Cache: Optimization](./18_kv_cache_optimization.md) | PagedAttention, prefix caching, quantized and offloaded caches, disaggregation |
| | **V · Precision & Compression** | |
| 19 | [Data Types & Precision](./19_data_types_and_precision.md) | FP32 to FP4, microscaling formats, mixed precision |
| 20 | [Quantization Fundamentals](./20_quantization_fundamentals.md) | Scales, zero points, granularity, outliers, and what to quantize |
| 21 | [Quantization Techniques](./21_quantization_techniques.md) | GPTQ, AWQ, SmoothQuant, rotations, codebooks, QAT and native low-bit models |
| 22 | [Choosing a Quantization](./22_quantization_benchmarks.md) | Quality vs bits, throughput, GGUF, and a decision guide |
| 23 | [Distillation & QAD](./23_knowledge_distillation.md) | Logit, sequence and on-policy distillation; quantization-aware distillation |
| | **VI · Hardware & Systems** | |
| 24 | [GPU Architecture](./24_gpu_architecture.md) | SMs, tensor cores, the memory hierarchy, the roofline, Hopper to Rubin |
| 25 | [CUDA & Kernels](./25_cuda_and_kernel_development.md) | The CUDA model, Triton and newer DSLs, FlashAttention's design, fusion |
| 26 | [Distributed Training](./26_distributed_training.md) | Data, tensor, pipeline, context and expert parallelism; ZeRO and FSDP |
| 27 | [ASICs & Accelerators](./27_asics_and_accelerators.md) | TPUs, Trainium, wafer-scale and SRAM machines, AMD, Apple silicon |
| | **VII · Ecosystem & Serving** | |
| 28 | [The Hugging Face Ecosystem](./28_huggingface_ecosystem.md) | Hub, datasets, tokenizers, TRL, and the wider open tooling |
| 29 | [transformers Internals](./29_transformers_library.md) | Configs, auto classes, attention backends, generation, caches |
| 30 | [Evaluation & Benchmarks](./30_evaluation_and_benchmarks.md) | What the 2026 benchmarks measure, contamination, arenas, agentic evals |
| 31 | [Serving & Deployment](./31_serving_and_deployment.md) | vLLM, SGLang, TensorRT-LLM, llama.cpp; batching, routing and cost |
| | **VIII · Scale & New Architectures** | |
| 32 | [Scaling Laws](./32_scaling_laws.md) | Kaplan, Chinchilla and its refit, over-training, inference-aware scaling |
| 33 | [Mixture of Experts](./33_mixture_of_experts.md) | Routing, balancing, fine-grained and shared experts, expert parallelism |
| 34 | [SSMs & Hybrids](./34_ssms_and_alternatives.md) | Mamba, linear attention, the delta rule, the hybrid wave, diffusion LMs |
| | **IX · Reasoning, Retrieval & Agents** | |
| 35 | [Reasoning Models](./35_reasoning_models.md) | Chain of thought, RL-trained thinking, test-time compute, faithfulness |
| 36 | [Retrieval-Augmented Generation](./36_rag.md) | Embeddings, hybrid search, reranking, long context vs retrieval |
| 37 | [Agents & Tool Use](./37_agents_and_tool_use.md) | Tool calling, MCP, coding and computer-use agents, safety |
| 38 | [The Frontier](./38_frontier.md) | Where the field stands in late 2026 and the open problems |
| | **Interactive & Appendices** | |
| ⚗ | [The Lab Bench](./labs.md) | All 23 interactive labs on one page |
| A | [History of Language Modelling](./appendix_a_history.md) | From Shannon to 2026 |
| B | [Tokenization Deep Dive](./appendix_b_tokenization.md) | BPE, byte-level BPE, Unigram, tokenizer quirks — with a trainer |
| C | [Model Config Reference](./appendix_c_model_config.md) | Every field of a modern `config.json`, explained |

## How to read this guide

- **Every chapter has the same spine**: a one-sentence version, the material with tensor shapes, a *Ponder* prompt, Key Takeaways, Check Yourself questions with worked answers, and a dated References table.
- **Numbers are dense and dated**: parameter counts, byte counts and FLOPs are computed, not quoted from memory, and the code that produced them is on the page.
- **Labs are instruments, not toys**: their arithmetic lives in small pure functions tested against the chapters' own numbers.
- **Press <kbd>/</kbd> or <kbd>⌘K</kbd> anywhere** to search every chapter, section and lab. The left rail is the residual stream; chapters you finish are marked.
- A companion guide, [Chips — The Complete Guide](https://karam-nus.github.io/chips/), goes below the CUDA line into the silicon itself; the accelerator data in the labs comes from it.
