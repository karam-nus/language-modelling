---
title: "The Lab Bench"
description: "All 23 interactive labs from the guide on one bench: n-grams, RoPE, attention, KV caches, training compute, fine-tuning memory, GRPO, sampling, speculative decoding, number formats, quantization, rooflines, parallelism, serving cost, scaling laws, MoE routing, fixed-state memory, test-time voting and BPE."
---

# The Lab Bench

> **The one-sentence version:** every instrument from the guide on one bench — each one a few small, tested functions behind sliders, so you can check a chapter's arithmetic with your own numbers.

Each lab also lives inside its chapter, next to the text that explains it; the link under each title takes you there. Their arithmetic is in [`assets/js/lm-labs.js`](https://github.com/karam-nus/language-modelling/blob/main/assets/js/lm-labs.js) as pure functions, and [`tools/test_labs.js`](https://github.com/karam-nus/language-modelling/blob/main/tools/test_labs.js) checks them against the numbers quoted in the chapters (`node tools/test_labs.js`). Accelerator specifications are shared with the companion [Chips guide](https://karam-nus.github.io/chips/).

{% include labgrid.html local=true %}

## Foundations

<p class="lab-from"><a href="01_introduction.html">From Chapter 1 →</a></p>
<div class="lab" data-lab="ngram"></div>
<p class="lab-from"><a href="02_embeddings.html">From Chapter 2 →</a></p>
<div class="lab" data-lab="rope"></div>
<p class="lab-from"><a href="03_the_transformer.html">From Chapter 3 →</a></p>
<div class="lab" data-lab="params"></div>
<p class="lab-from"><a href="04_attention_sdpa_and_mha.html">From Chapter 4 →</a></p>
<div class="lab" data-lab="attention"></div>

## Architecture

<p class="lab-from"><a href="05_attention_gqa_mqa_mla.html">From Chapter 5 →</a></p>
<div class="lab" data-lab="kvcache"></div>

## Training

<p class="lab-from"><a href="10_pretraining_at_scale.html">From Chapter 10 →</a></p>
<div class="lab" data-lab="compute"></div>
<p class="lab-from"><a href="13_finetuning_and_adaptation.html">From Chapter 13 →</a></p>
<div class="lab" data-lab="vram"></div>
<p class="lab-from"><a href="14_peft_lora_and_variants.html">From Chapter 14 →</a></p>
<div class="lab" data-lab="lora"></div>
<p class="lab-from"><a href="15_alignment_rlhf_and_beyond.html">From Chapter 15 →</a></p>
<div class="lab" data-lab="grpo"></div>

## Inference

<p class="lab-from"><a href="16_inference_and_sampling.html">From Chapter 16 →</a></p>
<div class="lab" data-lab="sampling"></div>
<p class="lab-from"><a href="16_inference_and_sampling.html">From Chapter 16 →</a></p>
<div class="lab" data-lab="specdec"></div>
<p class="lab-from"><a href="18_kv_cache_optimization.html">From Chapter 18 →</a></p>
<div class="lab" data-lab="paged"></div>

## Precision

<p class="lab-from"><a href="19_data_types_and_precision.html">From Chapter 19 →</a></p>
<div class="lab" data-lab="bits"></div>
<p class="lab-from"><a href="20_quantization_fundamentals.html">From Chapter 20 →</a></p>
<div class="lab" data-lab="quant"></div>
<p class="lab-from"><a href="22_quantization_benchmarks.html">From Chapter 22 →</a></p>
<div class="lab" data-lab="fit"></div>

## Hardware

<p class="lab-from"><a href="24_gpu_architecture.html">From Chapter 24 →</a></p>
<div class="lab" data-lab="roofline"></div>
<p class="lab-from"><a href="26_distributed_training.html">From Chapter 26 →</a></p>
<div class="lab" data-lab="parallel"></div>

## Serving

<p class="lab-from"><a href="31_serving_and_deployment.html">From Chapter 31 →</a></p>
<div class="lab" data-lab="servecost"></div>

## Scale

<p class="lab-from"><a href="32_scaling_laws.html">From Chapter 32 →</a></p>
<div class="lab" data-lab="scaling"></div>
<p class="lab-from"><a href="33_mixture_of_experts.html">From Chapter 33 →</a></p>
<div class="lab" data-lab="moe"></div>
<p class="lab-from"><a href="34_ssms_and_alternatives.html">From Chapter 34 →</a></p>
<div class="lab" data-lab="ssm"></div>

## Reasoning

<p class="lab-from"><a href="35_reasoning_models.html">From Chapter 35 →</a></p>
<div class="lab" data-lab="tts"></div>

## Tokenization

<p class="lab-from"><a href="appendix_b_tokenization.html">From Appendix B →</a></p>
<div class="lab" data-lab="bpe"></div>

---

**Next:** [Chapter 1 — Introduction](./01_introduction.md), where the next-token machine is explained.
