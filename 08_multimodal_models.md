---
title: "Chapter 8 — Multimodal Models"
---

[← Back to Table of Contents](./README.md)

# Chapter 8 — Multimodal Models

> *"A language model that can see, hear, and reason across modalities isn't just a better chatbot — it's a step toward general-purpose intelligence."*

## The Multimodal Idea

Modern LLMs are no longer text-only. **Vision-Language Models (VLMs)** process images alongside text, audio models handle speech, and emerging architectures handle video, 3D, and more. The key challenge is projecting each modality into the shared representation space of the language model.

<div class="diagram">
<div class="diagram-title">Multimodal LLM — General Architecture</div>
<div class="flow">
  <div class="flow-node orange wide">Image: [B, C, H, W] <small>e.g., [1, 3, 336, 336]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Vision Encoder (ViT/SigLIP): [B, N_patches, d_vision] <small>e.g., [1, 576, 1024]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Projection Layer: [B, N_patches, d_model] <small>map vision → LLM space</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Concatenate: [B, N_patches + T_text, d_model] <small>vision tokens + text tokens</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">LLM Decoder: process combined sequence → generate text</div>
</div>
</div>

## Vision Encoders

Vision Transformers (ViT) convert images into sequences of patch embeddings, making them directly compatible with transformer architectures.

### How ViT Works

<div class="diagram">
<div class="diagram-title">Vision Transformer — Image to Patch Tokens</div>
<div class="flow-h">
  <div class="flow-node accent">Image<br><small>[3, 336, 336]</small></div>
  <div class="flow-node purple">Split into patches<br><small>14×14 → 576 patches</small></div>
  <div class="flow-node green">Linear embed<br><small>[576, d_vision]</small></div>
  <div class="flow-node orange">+ position embed<br><small>[576, d_vision]</small></div>
  <div class="flow-node cyan">ViT Encoder<br><small>[576, d_vision]</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Vision Encoder Tensor Shapes</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Raw image</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">3</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">336</span><span class="ts-sep">,</span>
      <span class="ts-dim generic">336</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">RGB pixels</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">Patches (14×14 px)</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">576</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">588</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">14×14×3 = 588 per patch</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">After ViT encoder</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">576</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">1024</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">d_vision (SigLIP-L)</span>
  </div>
  <div class="tensor-flow-row">
    <span style="color: var(--text-muted); font-size: 0.75rem; min-width: 160px;">After projection</span>
    <div class="tensor-shape">
      <span class="ts-bracket">[</span>
      <span class="ts-dim batch">B</span><span class="ts-sep">,</span>
      <span class="ts-dim seq">576</span><span class="ts-sep">,</span>
      <span class="ts-dim feature">4096</span>
      <span class="ts-bracket">]</span>
    </div>
    <span style="color: var(--text-muted); font-size: 0.6875rem; margin-left: 0.5rem;">mapped to LLM d_model</span>
  </div>
</div>
</div>

### Common Vision Encoders

| Encoder | Training | Resolution | Patches | d_vision | Used By |
|---------|----------|:----------:|:-------:|:--------:|---------|
| CLIP ViT-L/14 | Contrastive (image-text pairs) | 224 | 256 | 1024 | LLaVA 1.0 |
| SigLIP SO400M | Sigmoid contrastive | 384 | 729 | 1152 | LLaVA-NeXT, PaliGemma |
| InternViT-6B | Contrastive + generative | 448 | 1024 | 3200 | InternVL 2 |
| DINOv2 ViT-L | Self-supervised | 518 | 1369 | 1024 | Various |

## Fusion Strategies

How vision and language tokens interact defines the model architecture:

<div class="diagram">
<div class="diagram-title">Modality Fusion Approaches</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-title">Early Fusion</div>
    <div class="card-desc">Concatenate vision + text tokens before feeding to LLM. Vision tokens attend to text and vice versa in every layer. <strong>Most common.</strong></div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Cross-Attention Fusion</div>
    <div class="card-desc">Add cross-attention layers where text queries attend to vision keys/values. Keeps modalities partially separate. Used by Flamingo.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Late Fusion</div>
    <div class="card-desc">Process modalities independently, combine only at decision time. Simplest but weakest interaction.</div>
  </div>
</div>
</div>

## Key VLM Architectures

### LLaVA (Visual Instruction Tuning)

The most influential open-source VLM architecture — elegantly simple.

<div class="diagram">
<div class="diagram-title">LLaVA Architecture</div>
<div class="flow">
  <div class="flow-node orange wide">Image → Vision Encoder (CLIP/SigLIP) → [B, N, d_vision]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">MLP Projector: [B, N, d_vision] → [B, N, d_model] <small>2-layer MLP</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Interleave: [system tokens] [image tokens] [user text tokens]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">LLM (Vicuna / LLaMA): process combined sequence → generate response</div>
</div>
</div>

Training: (1) **Pre-train projector** on image-caption pairs (freeze vision encoder + LLM), then (2) **instruction-tune** the LLM + projector on visual QA data.

### Other Architectures

| Model | Vision Encoder | Projection | LLM | Key Innovation |
|-------|---------------|-----------|-----|----------------|
| **LLaVA-NeXT** | SigLIP | 2-layer MLP | LLaMA/Vicuna | AnyRes (dynamic resolution) |
| **GPT-4V/4o** | Proprietary | Proprietary | GPT-4 | Seamless multimodal reasoning |
| **Gemini** | Natively multimodal | Shared encoder | Gemini | Trained multimodal from scratch |
| **PaliGemma** | SigLIP | Linear | Gemma | Small, efficient, compositional |
| **Qwen2-VL** | ViT + M-RoPE | MLP | Qwen2 | Multimodal RoPE, dynamic resolution |
| **Pixtral** | Custom 400M ViT | — | Mistral | Variable resolution, no padding |

## Audio Models: Whisper

**Whisper** (OpenAI, 2022) is an encoder-decoder model for speech recognition:

<div class="diagram">
<div class="diagram-title">Whisper Architecture</div>
<div class="flow">
  <div class="flow-node accent wide">Audio → Mel Spectrogram: [B, 80, 3000] <small>80 mel bins, 30s max</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">2× Conv1D → [B, 1500, d_model] <small>downsample 2×</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Transformer Encoder (bidirectional) → [B, 1500, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Transformer Decoder (causal + cross-attention) → text tokens</div>
</div>
</div>

| Whisper Model | Params | Layers (Enc/Dec) | d_model | Heads |
|--------------|-------:|:----------------:|--------:|------:|
| tiny | 39M | 4/4 | 384 | 6 |
| base | 74M | 6/6 | 512 | 8 |
| small | 244M | 12/12 | 768 | 12 |
| medium | 769M | 24/24 | 1024 | 16 |
| large-v3 | 1.5B | 32/32 | 1280 | 20 |

## Using a VLM with Transformers

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

image = Image.open("example.jpg")
prompt = "<image>\nDescribe this image in detail."

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
# inputs["pixel_values"].shape: [1, 3, 336, 336]
# inputs["input_ids"].shape:    [1, T_text]  (includes <image> placeholder)

output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0], skip_special_tokens=True))
```

## The Token Budget Problem

Each image becomes hundreds of tokens (576 for a 336×336 image with 14×14 patches). With multiple images or higher resolution, image tokens can dominate the context window:

| Resolution | Patch Size | Tokens per Image |
|:----------:|:----------:|:----------------:|
| 224×224 | 14×14 | 256 |
| 336×336 | 14×14 | 576 |
| 448×448 | 14×14 | 1024 |
| Dynamic (4 tiles) | 14×14 | ~2304 |

Solutions: **token compression** (average pooling, learned resampling), **dynamic resolution** (process at native aspect ratio, tile into sub-images), and **early fusion with pooling** (Qwen2-VL reduces tokens with a perceiver-style resampler).

## What's Next

With architectures covered — text-only (encoder, decoder, encoder-decoder) and multimodal — we now shift to **how these models are trained**. The next chapter covers pre-training at scale.

[← Previous: Chapter 7 — Encoder & Seq2Seq Models](./07_encoder_and_seq2seq_models.md) · **Next: [Chapter 9 — Data for LLMs →](./09_data_for_llms.md)**

---

*Last updated: April 2026*
