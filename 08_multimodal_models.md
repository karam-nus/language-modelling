---
title: "Chapter 8 — Multimodal Models"
description: "How images, video and audio become tokens: ViT and SigLIP encoders, projectors and patch merging, unified-embedding vs cross-attention vs native early fusion, dynamic resolution and token budgets, Whisper and omni models, image generation inside LLMs, and vision-language-action models."
---

[← Back to Table of Contents](./README.md)

# Chapter 8 — Multimodal Models

In 2023 a "multimodal LLM" meant a text model with a vision adapter bolted on. By 2026 the frontier models — Gemini 3, GPT-5.x, Claude 5, Qwen3.5, Gemma 4, Kimi K3 — are trained on images, video and often audio from early in pre-training, and the open models ship with vision built in.

> **The one-sentence version:** every modality is turned into a sequence of vectors the size of the LLM's hidden state — image patches, video frames, audio frames — and from then on the transformer treats them as tokens.

## The Multimodal Idea

<div class="diagram">
<div class="diagram-title">Multimodal LLM — General Architecture</div>
<div class="flow">
  <div class="flow-node orange wide">Image: [B, C, H, W] <small>e.g., [1, 3, 336, 336]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Vision encoder (ViT / SigLIP): [B, N_patches, d_vision] <small>e.g., [1, 576, 1024]</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Projector / patch merger: [B, N_tokens, d_model] <small>map vision → LLM space, often 4× fewer tokens</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Interleave: [B, N_tokens + T_text, d_model] <small>image tokens sit where the image appeared in the prompt</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">LLM decoder: process the combined sequence → generate text</div>
</div>
</div>

## Vision Encoders

A Vision Transformer (ViT) cuts an image into fixed-size patches, flattens each patch, projects it linearly and runs a transformer encoder over the patch sequence.

### How ViT Works

<div class="diagram">
<div class="diagram-title">Vision Transformer — Image to Patch Tokens</div>
<div class="flow-h">
  <div class="flow-node accent">Image<br><small>[3, 336, 336]</small></div>
  <div class="flow-node purple">Split into patches<br><small>14×14 px → 24×24 = 576</small></div>
  <div class="flow-node green">Linear embed<br><small>[576, d_vision]</small></div>
  <div class="flow-node orange">+ position embed<br><small>[576, d_vision]</small></div>
  <div class="flow-node cyan">ViT encoder<br><small>[576, d_vision]</small></div>
</div>
</div>

<div class="diagram">
<div class="diagram-title">Vision Encoder Tensor Shapes (LLaVA-1.5)</div>
<div class="tensor-flow">
  <div class="tensor-flow-row">
    <span class="ts-label">Raw image</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim generic">3</span><span class="ts-sep">,</span><span class="ts-dim generic">336</span><span class="ts-sep">,</span><span class="ts-dim generic">336</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">RGB pixels</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">Patches (14×14 px)</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">576</span><span class="ts-sep">,</span><span class="ts-dim feature">588</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">14×14×3 = 588 values per patch</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After ViT encoder</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">576</span><span class="ts-sep">,</span><span class="ts-dim feature">1024</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">d_vision (CLIP ViT-L/14-336)</span>
  </div>
  <div class="tensor-flow-row">
    <span class="ts-label">After projection</span>
    <div class="tensor-shape"><span class="ts-bracket">[</span><span class="ts-dim batch">B</span><span class="ts-sep">,</span><span class="ts-dim seq">576</span><span class="ts-sep">,</span><span class="ts-dim feature">4096</span><span class="ts-bracket">]</span></div>
    <span class="ts-note">mapped to the LLM's d_model</span>
  </div>
</div>
</div>

### Common Vision Encoders

| Encoder | Training | Resolution | Patches | d_vision | Used by |
|---------|----------|:----------:|:-------:|:--------:|---------|
| CLIP ViT-L/14 | Contrastive (softmax over image–text pairs) | 224 / 336 | 256 / 576 | 1024 | LLaVA 1.0 / 1.5 |
| SigLIP SO400M/14 | Sigmoid contrastive | 384 | 729 | 1152 | LLaVA-OneVision, PaliGemma, Gemma 3 (at 896) |
| SigLIP 2 (2025) | Sigmoid + captioning + self-distillation | 256–512, NaFlex native aspect | varies | 768–1536 | Many 2025–26 VLMs |
| InternViT-6B | Contrastive + generative | 448 | 1024 | 3200 | InternVL 2/3 |
| Qwen2.5-VL ViT | Trained with the LLM | Native resolution | varies | 1280 | Qwen2.5-VL |
| DINOv2 ViT-L/14 | Self-supervised (no text) | 518 | 1369 | 1024 | Robotics, dense features |

Two shifts since 2024: encoders now accept **native resolution and aspect ratio** (rather than squashing every image to a square), and the best VLMs **train the encoder jointly** with the LLM instead of freezing a CLIP checkpoint.

## Fusion Strategies

How vision and language tokens interact defines the architecture:

<div class="diagram">
<div class="diagram-title">Modality Fusion Approaches</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Unified embedding (LLaVA-style)</div>
    <div class="card-desc">Project encoder outputs into the LLM's input space and interleave them with text tokens. Image and text tokens attend to each other in every layer. <strong>The most common open design</strong> — Qwen-VL, InternVL, Gemma 3, Pixtral.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Cross-attention</div>
    <div class="card-desc">Keep image features outside the sequence; insert cross-attention layers in which text queries read image keys and values. The LLM's text path can stay frozen. Flamingo, Llama 3.2 Vision.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Native early fusion</div>
    <div class="card-desc">Train on interleaved image and text from the start of pre-training, sometimes with images as discrete tokens in a shared vocabulary. Chameleon, Llama 4, Gemini; the 2026 frontier default.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Late fusion</div>
    <div class="card-desc">Process modalities independently and combine only at the end (e.g. CLIP-score reranking). Simple; weakest interaction.</div>
  </div>
</div>
</div>

## Key VLM Architectures

### LLaVA (Visual Instruction Tuning)

The most influential open VLM recipe — strikingly simple:

<div class="diagram">
<div class="diagram-title">LLaVA Architecture</div>
<div class="flow">
  <div class="flow-node orange wide">Image → vision encoder (CLIP / SigLIP) → [B, N, d_vision]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">MLP projector: [B, N, d_vision] → [B, N, d_model] <small>2-layer MLP</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Interleave: [system tokens] [image tokens] [user text tokens]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">LLM (Vicuna / Llama): process the combined sequence → generate a response</div>
</div>
</div>

Training: (1) **align** — train only the projector on image–caption pairs, with encoder and LLM frozen; (2) **instruction-tune** the LLM and projector on visual question-answering and conversation data. Modern recipes add a third, larger stage in between — full multimodal pre-training on interleaved web documents, OCR, charts and video — which is where most of the capability now comes from.

### The Landscape

| Model (year) | Vision side | Connector | LLM | Key idea |
|-------|---------------|-----------|-----|----------------|
| **LLaVA-OneVision** (2024) | SigLIP | 2-layer MLP | Qwen2 | One model for single image, multi-image and video |
| **Qwen2-VL / 2.5-VL** (2024–25) | ViT at native resolution, M-RoPE | Merge 2×2 patches + MLP | Qwen2 / 2.5 | Dynamic resolution; absolute-time video positions |
| **Qwen3-VL** (2025) | ViT; features from several encoder depths ("DeepStack") | MLP merger | Qwen3 dense & MoE | Interleaved M-RoPE, 256K context, long video |
| **InternVL3 / 3.5** (2025) | InternViT | MLP + pixel shuffle | Qwen / InternLM | Native multimodal pre-training |
| **Gemma 3** (2025) | SigLIP 400M at 896² | Pool to 256 tokens | Gemma 3 | "Pan & scan" crops for odd aspect ratios |
| **Llama 3.2 Vision** (2024) | ViT | Cross-attention layers | Llama 3.1 | Text weights frozen, vision added alongside |
| **Llama 4** (2025) | MetaCLIP-based encoder | Early fusion | Llama 4 MoE | Pre-trained jointly on text, images, video |
| **Pixtral / Mistral Large 3** | Mistral ViT, variable resolution | MLP | Mistral | Row-break tokens preserve 2D layout |
| **Gemma 4, Qwen3.5/3.6, Kimi K3** (2026) | Built-in encoders | Native | Same model | Vision (and video) in the base model by default |

## The Token Budget Problem

Every image becomes hundreds or thousands of tokens, and attention cost grows with them:

| Scheme | Image | Tokens |
|:----------|:----------:|:----------------:|
| CLIP-L/14 at 224 (LLaVA 1.0) | 224×224 | 256 |
| CLIP-L/14 at 336 (LLaVA-1.5) | 336×336 | 576 |
| AnyRes, 4 tiles + overview (LLaVA-NeXT) | 672×672 | ~2,880 |
| Gemma 3 (pooled) | one 896×896 crop | 256 |
| Qwen2-VL (14-px patches, 2×2 merge) | 1036×1036 | (1036/28)² = 1,369 |

Qwen2-VL's rule is worth memorising: after resizing each side to a multiple of 28, **tokens ≈ (H/28) × (W/28)** — so a 1080p screenshot costs about 2,700 tokens. Video multiplies that by the number of sampled frames (Qwen2-VL also merges pairs of consecutive frames). The standard tools:

- **Merge or pool**: Qwen-VL's 2×2 patch merger, Gemma 3's pooling to 256, pixel-shuffle in InternVL.
- **Cap the pixels**: VLM processors take `min_pixels` / `max_pixels` so you trade detail for tokens explicitly.
- **Sample frames smartly**: fixed frames per second plus per-frame token caps for long video.
- **Drop tokens late**: prune visually redundant tokens after a few LLM layers (a 2025–26 research staple).

<div class="callout note">
<span class="callout-title">Why OCR got good</span>
<p>Much of the 2024–26 jump in document, chart and screenshot understanding came not from bigger LLMs but from native-resolution encoders and enough tokens per image to read small text — plus huge synthetic OCR and chart datasets in pre-training. It is also what made computer-use agents (Chapter 37) possible.</p>
</div>

## Using a VLM with Transformers

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://example.com/chart.png"},
    {"type": "text",  "text": "What is the highest value in this chart?"},
]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
# inputs["pixel_values"]:   [N_patches, 1176]  — flattened 2×14×14×3 patches, not [1, 3, H, W]
# inputs["image_grid_thw"]: [[1, H/14, W/14]]  — the patch grid M-RoPE needs
# inputs["input_ids"]:      [1, T]             — includes one <|image_pad|> per merged token

out = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Note the shapes: native-resolution VLMs pass a flat list of patches plus a grid description, because images in one batch no longer share a size. (Each Qwen patch is 1,176 values because it spans 2 time steps × 14 × 14 pixels × 3 channels — a still image is duplicated in time.)

## Audio: Whisper and Omni Models

**Whisper** (OpenAI, 2022) is the encoder-decoder workhorse for speech recognition:

<div class="diagram">
<div class="diagram-title">Whisper Architecture</div>
<div class="flow">
  <div class="flow-node accent wide">Audio → log-mel spectrogram: [B, 80 or 128, 3000] <small>10 ms hop, 30 s window; large-v3 uses 128 mel bins</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">2 × Conv1D (second has stride 2) → [B, 1500, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Transformer encoder (bidirectional) → [B, 1500, d_model]</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Transformer decoder (causal + cross-attention) → text tokens</div>
</div>
</div>

| Whisper model | Params | Layers (enc/dec) | d_model | Heads |
|--------------|-------:|:----------------:|--------:|------:|
| tiny | 39M | 4/4 | 384 | 6 |
| base | 74M | 6/6 | 512 | 8 |
| small | 244M | 12/12 | 768 | 12 |
| medium | 769M | 24/24 | 1024 | 16 |
| large-v3 | 1.55B | 32/32 | 1280 | 20 |
| large-v3-turbo | 809M | 32/4 | 1280 | 20 |

Turbo shows where the cost lives: the decoder runs once per output token, so cutting it from 32 to 4 layers makes transcription several times faster for a small accuracy loss.

**Omni models** go further: audio in *and* out, in one network. GPT-4o (2024) popularised real-time voice; open designs such as Qwen2.5-Omni and Qwen3-Omni split the work into a *Thinker* (the LLM, reading text, image, audio and video) and a *Talker* (which streams speech tokens), while Kyutai's Moshi models two audio streams at once for full-duplex conversation. Speech is represented as discrete **codec tokens** (from neural audio codecs such as EnCodec or Mimi) — exactly as text is represented by BPE tokens.

## Generating Images and Actions

The same token machinery runs in reverse:

- **Image generation inside the LLM.** Since GPT-4o's image generation (March 2025) and Gemini's native image models (2025), frontier assistants produce images from the same model that reads them. Open "unified" models (Janus-Pro, BAGEL, Show-o) combine an autoregressive LLM with discrete image tokens or a diffusion head. The appeal is control: the model that understood the conversation draws the picture.
- **Vision-language-action (VLA) models.** A VLM fine-tuned to emit robot actions — joint targets or end-effector deltas — as tokens or via a flow-matching "action expert". RT-2 (2023) started the line; π0 / π0.5, NVIDIA GR00T N1 and Gemini Robotics (2025) are current examples. [Appendix C](./appendix_c_model_config.md) covers their config files.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>A 1080p screenshot costs ~2,700 tokens in Qwen2-VL's scheme, but the same information as accessible-tree text might cost 500. Computer-use agents choose between pixels and structure every step. When is the expensive modality worth it?</p>
</div>

## Key Takeaways

- A modality enters an LLM as a sequence of vectors in d_model: ViT patches → projector (often with 2×2 merging) → interleaved with text.
- Fusion designs: unified embedding (most open VLMs), cross-attention (Flamingo, Llama 3.2 Vision), native early fusion (Llama 4, Gemini, 2026 open models).
- Native-resolution encoders and token budgets are the practical levers: Qwen2-VL spends ≈ (H/28)(W/28) tokens per image.
- Whisper-style encoder-decoders still dominate ASR; omni models add speech output via codec tokens.
- The token interface also runs outwards: unified models generate images, and VLAs generate robot actions.

## Check Yourself

<details class="check">
<summary>How many tokens does a 448×448 image produce with a 14-pixel ViT and no merging? With Qwen-style 2×2 merging?</summary>
<p>(448/14)² = 32² = 1,024 patches. Merging each 2×2 block gives 16² = 256 tokens.</p>
</details>

<details class="check">
<summary>Why does Llama 3.2 Vision use cross-attention rather than putting image tokens in the sequence?</summary>
<p>It lets Meta keep the Llama 3.1 text weights frozen — text-only behaviour is untouched — and image features are not counted against the context or recomputed in every self-attention layer. The cost is that images and text interact only at the inserted cross-attention layers.</p>
</details>

<details class="check">
<summary>Whisper's encoder always sees a 30-second window. How many encoder positions does that become, and why?</summary>
<p>A 10 ms hop gives 3,000 spectrogram frames; the second convolution has stride 2, so the encoder sees 1,500 positions (one per 20 ms).</p>
</details>

<details class="check">
<summary>Why did large-v3-turbo prune decoder layers rather than encoder layers?</summary>
<p>The encoder runs once per 30-second window; the decoder runs once per output token. Decoder depth therefore dominates latency, while encoder depth matters most for acoustic accuracy.</p>
</details>

<details class="check">
<summary>What does <code>image_grid_thw</code> carry in a Qwen-VL batch, and what would break without it?</summary>
<p>The temporal, height and width patch counts of each image or video. The model needs them to un-flatten patches for the 2×2 merge and to assign M-RoPE (t, h, w) positions; without them it cannot tell where one row of patches ends and the next begins.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| An Image is Worth 16x16 Words (ViT) | Dosovitskiy et al., 2020 | [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929) | Vision Transformer |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP) | Radford et al., 2021 | [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) | Contrastive image–text encoders |
| Sigmoid Loss for Language Image Pre-Training (SigLIP) | Zhai et al., 2023 | [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343) | SigLIP |
| SigLIP 2 | Tschannen et al., 2025 | [arxiv.org/abs/2502.14786](https://arxiv.org/abs/2502.14786) | NaFlex native-resolution encoders |
| Flamingo | Alayrac et al., 2022 | [arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198) | Cross-attention fusion |
| Visual Instruction Tuning (LLaVA) | Liu et al., 2023 | [arxiv.org/abs/2304.08485](https://arxiv.org/abs/2304.08485) | Projector + instruction tuning |
| Qwen2-VL | Wang et al., 2024 | [arxiv.org/abs/2409.12191](https://arxiv.org/abs/2409.12191) | Dynamic resolution, patch merging, M-RoPE |
| Qwen2.5-VL Technical Report | Qwen Team, 2025 | [arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923) | Window-attention ViT, absolute time |
| Chameleon: Mixed-Modal Early-Fusion Foundation Models | Chameleon Team (Meta), 2024 | [arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818) | Native early fusion |
| Robust Speech Recognition via Large-Scale Weak Supervision | Radford et al., 2022 | [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356) | Whisper |
| π0: A Vision-Language-Action Flow Model | Black et al. (Physical Intelligence), 2024 | [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164) | VLA with an action expert |

---

**Next:** [Chapter 9 — Data for LLMs & VLMs](./09_data_for_llms.md) — architecture is only half of a model; the other half is what it is trained on.

[← Back to Table of Contents](./README.md)
