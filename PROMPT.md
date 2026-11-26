# Authoring Spec — Language Modelling Guide

This file describes how the guide is written and built, so that new chapters and later edits stay consistent with the rest of the site and with the companion [Chips guide](https://github.com/karam-nus/chips). It is not published on the site (`_config.yml` excludes it).

## Audience and voice

- Reader: an ML practitioner fluent in Python, comfortable with tensors and basic linear algebra, who wants to know how large language models are actually built, trained, compressed and served — not just how to call them.
- Build every idea from first principles, write down the tensor shape at every step, and tie it back to a concrete decision (memory, compute, latency, quality).
- Plain, direct prose. Numbers over adjectives. Name the model, the precision, the context length and the date whenever you quote a figure.

## Accuracy rules

1. **Compute numbers, do not recall them.** Parameter counts, KV-cache bytes, FLOPs, GPU-hours and memory budgets are derived from the model's `config.json` and shown with the arithmetic (or the code that produced them).
2. **Re-run code.** Every printed output in a code block must be what the code prints. If it needs a GPU, PyTorch or a model download that is unavailable, run an equivalent NumPy version (`ml_dtypes` for low-precision formats) and say so.
3. **Dense throughput only.** H100 SXM = 989 TFLOP/s BF16 dense, 1,979 FP8 dense. Never quote a 2:4-sparse figure as dense. Accelerator specs come from `assets/data/accelerators.json`, shared with the Chips guide.
4. **Date-stamp fast-moving facts.** Model releases, leaderboard positions, prices and "current best" claims carry a month and year ("as of November 2026"). Closed models: only what the developer has published; never guess parameter counts.
5. **Primary sources first.** Papers (arXiv ID checked against the title), model cards and configs, official docs. Every chapter ends with a dated References table.
6. **Labs and text agree.** Every number a lab shows by default must match the chapter; `tools/test_labs.js` asserts the important ones.

## Chapter skeleton

```
---
title: "Chapter NN — Title"
description: "One or two sentences for search, social cards and the command palette."
---

[← Back to Table of Contents](./README.md)

# Chapter NN — Title

> **The one-sentence version:** …            becomes the yellow-shadowed dek

## Sections …                                  h2s are numbered §1, §2 … and become KV-strip cells
(diagrams, tensor shapes, code, tables, labs)

<div class="callout ponder"> … </div>          one per chapter: an open question

## Key Takeaways                               bullet list → numbered takeaway cards
## Check Yourself                              4–6 <details class="check"> with worked answers
## References                                  | Source | Author / Org, Year | Link | What this chapter takes from it |

---

**Next:** [Chapter NN+1 — Title](./NN+1_slug.md) — one sentence on why it follows.

[← Back to Table of Contents](./README.md)
```

Navigation lives in `_config.yml` under `navigation:`; each entry has `title`, `url` (without `.html`) and `n` (the number shown in the residual-stream rail). Entries with `url: ""` are part headers.

## Components (styled in `assets/css/style.css`)

| Component | Markup | Use for |
|---|---|---|
| Callouts | `<div class="callout note\|warn\|tip\|field\|myth\|trend\|ponder">` + `<span class="callout-title">` | Asides; `trend` for 2025–26 developments; `ponder` once per chapter |
| Self-check | `<details class="check"><summary>Q</summary><p>A</p></details>` | Check Yourself (auto-numbered Q1, Q2 …) |
| Plates | `<div class="diagram"><div class="diagram-title">…</div> … </div>` | Every diagram; gets crop marks and an automatic "Plate N.k" number |
| Diagram parts | `.flow` / `.flow-h` of `.flow-node` (+ colour class) and `.flow-arrow`; `.flow-caption`; `.diagram-grid.cols-2/3/4` of `.diagram-card`; `.compare`; `.layer-stack`; `.timeline`; `.cycle` | Concept maps |
| Tensor shapes | `.tensor-flow` > `.tensor-flow-row` > `.ts-label` + `.tensor-shape` of `.ts-dim` (`.batch`, `.seq`, `.hidden`, `.heads`, `.vocab`, …) + `.ts-note` | Shapes through a layer |
| Colour classes | `.accent .blue .green .purple .teal .orange .pink .red .yellow` | Set `--c` on any component |
| Figures | `<div class="img-caption"><img src="{{ '/assets/images/X.svg' \| relative_url }}" alt="…"><figcaption>…</figcaption></div>` | SVG figures; numbered "Fig. N.k" |
| Labs | `<div class="lab" data-lab="NAME"><noscript>What it does.</noscript></div>` | See below |

No inline `style=""` attributes in chapters — add a class instead.

Jekyll/kramdown gotchas:
- Raw HTML blocks must not contain blank lines.
- `jekyll-relative-links` rewrites `.md` links in Markdown but **not** inside raw HTML: use `.html` in HTML `href`s.
- Markdown emphasis does not render inside HTML blocks; use `<em>` / `<strong>`.
- Never write `{{` or `{%` in prose or code without `{% raw %}` … `{% endraw %}`.

## Labs

All 23 labs live in `assets/js/lm-labs.js`. Each is two layers:

1. **Pure functions** (`countParams`, `kvBytes`, `finetuneMem`, `fpEncode`, `quantize`, `chinOptimal`, `simulateRouting`, `recallError`, `bpeTrain`, …) exported via `module.exports` for Node.
2. **A renderer** registered in `LABS.NAME` that builds controls (`slider`, `select`, `check`), readouts and charts (`chart`, `barList`) from the lab UI kit.

To add one: write and export the pure function, add assertions to `tools/test_labs.js` against the numbers in the chapter, register the renderer, add an entry to `_data/labs.yml` (feeds the home page, the Lab Bench page and the search index), and drop the `div.lab` into the chapter and into `labs.md`.

```bash
node tools/test_labs.js      # must print N/N lab checks passed
```

## Figures

Process diagrams are drawn by `assets/js/lm-figs.js` as theme-aware SVG (ink tokens such as `var(--blue)`, so they follow the paper/night theme). Put one inside a plate:

```html
<div class="diagram">
<div class="diagram-title">Ring All-Reduce, Step by Step</div>
<div class="fig" data-fig="ringallreduce" data-n="4"><noscript>Text description of the figure.</noscript></div>
<div class="flow-caption">One-line caption.</div>
</div>
```

Available: `heads` (MHA/GQA/MQA/MLA), `masks` (`data-set` of causal, bidirectional, prefix, sliding, chunked, sink, dilated, nsa, topk), `linstate`, `flash`, `gated`, `collectives`, `ringallreduce`, `ddp`, `zero`, `fsdp`, `tp` (`mlp`/`attn`), `sp`, `pipeline` (`gpipe`/`1f1b`/`interleaved`/`zb` with `data-p`, `data-m`, `data-v`), `ep`, `ring`, `ulysses`, `mesh`, `loopstack`. Figures that show a process get a ◀ ▷ ▶ stepper. Any arithmetic a figure shows (pipeline bubbles, ring all-reduce states, masks) comes from the tested pure functions in `lm-labs.js` (`window.LMPure`); add a test when you add one.

## Design system — "Latent Press"

A riso-printed manuscript, not a dashboard.

- **Paper** theme (warm off-white, ink black, vermilion `--verm` for attention) and **Night** theme (deep ink, screen-blended inks). The theme follows the OS until the reader toggles it (`localStorage['lm-theme']`).
- **Inks**: vermilion, blue, teal, green, purple, pink, orange, yellow, red — used flat, and overprinted with `mix-blend-mode` (multiply on paper, screen at night) for the misregistered chapter numerals and the home title.
- **Type**: Fraunces (display, with `SOFT`/`WONK` axes), Source Serif 4 (body), JetBrains Mono (labels, code, data).
- **Signature pieces**: the masthead splits the page title into position-labelled tokens (t0, t1 …); the KV strip under it fills one cell per section as you read; the left rail is the residual stream with read-state nodes; the token meter estimates the page length in tokens; `/` or `⌘K` opens a command palette that ranks results with a softmax over match scores.
- **Layout**: a named-line grid (`full`, `wide`, `prose`). Prose stays at a reading measure; plates, labs, code and tables break out to `wide`. On ≥1500 px a margin index tracks the current section.
- Motion is small and functional, and disabled under `prefers-reduced-motion`.

## Checks before pushing

1. `node tools/test_labs.js`
2. Build locally with the `github-pages` gem (`bundle exec jekyll build`) and open a few pages in both themes and at phone width (no horizontal scroll).
3. Search for leftover inline styles in chapters: `grep -n 'style="' [0-9]*.md appendix_*.md README.md labs.md` should return nothing.
