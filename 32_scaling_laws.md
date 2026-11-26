---
title: "Chapter 32 — Scaling Laws & Emergent Abilities"
description: "Kaplan vs Chinchilla and why they disagreed, the parametric loss L(N, D) with a verified compute-optimal calculation, why every 2025–26 model is trained far past 20 tokens per parameter, data-constrained and precision scaling, the data wall, test-time and RL scaling laws, and the emergence-vs-mirage debate."
related:
  - { id: training, why: "Compute budgets in real training runs." }
  - { id: math, why: "Power laws and fitting, from first principles." }
---

[← Back to Table of Contents](./README.md)

# Chapter 32 — Scaling Laws & Emergent Abilities

Scaling laws are the closest thing deep learning has to physics: across many orders of magnitude, loss falls as a smooth power law in parameters, data and compute. They decide how big a model should be, how long to train it, and — since 2024 — how much to spend on reinforcement learning and on thinking at inference time.

> **The one-sentence version:** loss is predictable from compute and how you split it between parameters and tokens; capabilities are less predictable, but mostly because of how we measure them.

## Neural Scaling Laws

### Kaplan et al. (2020)

OpenAI found that loss falls as a power law in each resource when the others are not limiting:

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

with N = parameters, D = training tokens, C = training FLOPs. Their compute-optimal recipe scaled parameters much faster than data (N ∝ C^0.73): "bigger model, fewer tokens" — the recipe behind GPT-3 (175B parameters, 300B tokens).

### Chinchilla (Hoffmann et al., 2022)

<div class="img-caption">
  <img src="{{ '/assets/images/scaling_laws.svg' | relative_url }}" alt="Scaling laws chart showing compute-optimal frontier versus under-trained models, plus Chinchilla formula and model comparison table">
  <figcaption>Chinchilla scaling: at a fixed compute budget, parameters and tokens should grow together — about 20 tokens per parameter</figcaption>
</div>

DeepMind trained over 400 models and found that parameters and data should scale **equally** (N ∝ C^0.5, D ∝ C^0.5) — about **20 tokens per parameter**. Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) with the same compute.

<div class="diagram">
<div class="diagram-title">Compute-Optimal Training</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Kaplan (2020)</div>
    <ul>
      <li>N ∝ C^0.73: grow parameters faster</li>
      <li>GPT-3: 175B on 300B tokens (~1.7 tokens/param)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Chinchilla (2022)</div>
    <ul>
      <li>N ∝ C^0.5, D ∝ C^0.5</li>
      <li>Chinchilla: 70B on 1.4T tokens (20 tokens/param)</li>
    </ul>
  </div>
</div>
</div>

Why did two careful studies disagree? Later analyses (Porian et al., 2024) traced most of the gap to methodology: Kaplan counted non-embedding parameters only, ignored the last layer's FLOPs, used a fixed learning-rate schedule length (penalising long runs) and tuned hyperparameters less. With those fixed, the Kaplan setup reproduces Chinchilla.

### The Loss Surface

Chinchilla also fit a parametric loss:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

E is the irreducible entropy of text; the other terms are the penalties for a finite model and finite data. Minimising it subject to C = 6ND gives the compute-optimal split. Using the coefficients as re-fitted by Epoch AI (2024), who found the original paper's third fit inconsistent with its other two:

```python
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658   # Besiroglu et al. (2024) refit
L = lambda N, D: E + A / N**alpha + B / D**beta

def compute_optimal(C):
    """Minimise L(N, D) subject to 6·N·D = C (closed form for this loss)."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for C in (1e21, 1e23, 3.8e25):
    N, D = compute_optimal(C)
    print(f"C={C:.1e}: N*={N/1e9:7.1f}B  D*={D/1e12:6.2f}T  D/N={D/N:5.1f}  loss={L(N, D):.3f}")

C = 3.8e25                                   # Llama 3.1 405B's training compute
for N in (405e9, 70e9, 8e9):
    D = C / (6 * N)
    print(f"N={N/1e9:5.0f}B D={D/1e12:6.1f}T loss={L(N, D):.3f}")
# C=1.0e+21: N*=    2.8B  D*=  0.06T  D/N= 21.6  loss=2.308
# C=1.0e+23: N*=   29.4B  D*=  0.57T  D/N= 19.2  loss=2.035
# C=3.8e+25: N*=  618.7B  D*= 10.24T  D/N= 16.5  loss=1.895
# N=  405B D=  15.6T loss=1.895
# N=   70B D=  90.5T loss=1.918
# N=    8B D= 791.7T loss=2.001
```

The last three lines are the key insight. At Llama 3.1 405B's compute, a 70B model trained on 90T tokens would reach almost the same loss (+0.02 nats on Chinchilla's data) while being **5.8× cheaper to serve** — the loss surface is very flat around the optimum. (Absolute losses are specific to Chinchilla's dataset and tokenizer.)

Move the budget and the model size yourself — and add the inference you expect to serve:

<div class="lab" data-lab="scaling"><noscript>Interactive lab: Chinchilla loss surface, compute-optimal model size, and the trade-off of over-training a smaller model once lifetime inference is counted. Enable JavaScript to use it.</noscript></div>

### Compute-Optimal Reference Points (≈ 20 tokens per parameter)

| Model size | Chinchilla-optimal tokens | Training compute (6ND) |
|-----------|--------------------------|-------------------|
| 1B | 20B | 1.2 × 10²⁰ |
| 7B | 140B | 5.9 × 10²¹ |
| 70B | 1.4T | 5.9 × 10²³ |
| 405B | 8.1T | 2.0 × 10²⁵ |

## Beyond Chinchilla: Over-Training on Purpose

Chinchilla minimises *training* compute. But a model is trained once and served billions of times, so total cost favours smaller models trained much longer (Sardana & Frankle, 2023, *Beyond Chinchilla-Optimal: Accounting for Inference*). Every major 2024–26 model is "over-trained":

| Model | Parameters (active) | Training tokens | Tokens / parameter |
|-------|-----------|----------------|--------------|
| Chinchilla (2022) | 70B | 1.4T | 20 |
| Llama 2 70B (2023) | 70B | 2T | 29 |
| Llama 3.1 8B (2024) | 8B | 15T | 1,875 |
| Llama 3.1 405B (2024) | 405B | 15.6T | 39 |
| Llama 3.2 1B (2024) | 1.2B | 9T | ~7,500 |
| Qwen3-0.6B (2025) | 0.6B | 36T | ~60,000 |
| DeepSeek-V3 (2024) | 37B active / 671B | 14.8T | 400 per active parameter |

Small models keep improving long past the "optimal" point — loss keeps falling, just slowly — and their quality per serving dollar is what matters.

## Other Scaling Axes

| Law | Finding | Source |
|---|---|---|
| **Data-constrained scaling** | Repeating data for up to ~4 epochs is nearly as good as fresh data; returns fade after ~16 epochs | Muennighoff et al., 2023 |
| **Precision scaling** | Training in lower precision reduces "effective parameters"; post-training quantization hurts *more* for models trained on more tokens | Kumar et al., 2024 |
| **The data wall** | The stock of public human text is roughly hundreds of trillions of tokens; frontier runs could exhaust it around 2026–2032, driving synthetic data, multimodal data and RL | Villalobos et al. (Epoch AI), 2024 |
| **MoE scaling** | At fixed active compute, more experts (higher sparsity) keeps lowering loss with diminishing returns | Clark et al., 2022; Krajewski et al., 2024 |

## Test-Time and RL Scaling

The 2024–26 era added two new axes on which performance scales with compute:

- **Test-time compute.** OpenAI's o1 showed accuracy rising roughly log-linearly with the number of thinking tokens; Snell et al. (2024) showed that allocating inference compute adaptively (more for harder prompts) can beat a 14× larger model on some tasks. [Chapter 35](./35_reasoning_models.md) covers how.
- **RL compute.** Reinforcement learning on verifiable tasks keeps improving with compute; a large 2025 study (*The Art of Scaling Reinforcement Learning Compute*) found RL performance follows predictable sigmoid-shaped curves, so small runs can forecast large ones — the same predictability pre-training laws gave earlier.

## Emergent Abilities

Some capabilities appear to switch on at a certain scale instead of improving gradually. Wei et al. (2022) catalogued dozens: few-shot arithmetic in GPT-3 (3-digit addition near zero for small models, ~80% at 175B), chain-of-thought prompting helping only beyond ~60–100B parameters, and multi-step tasks in BIG-Bench.

### Real or Mirage?

<div class="diagram">
<div class="diagram-title">Emergence: Real or Mirage?</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Emergence is real</div>
    <ul>
      <li>Some behaviours change qualitatively (e.g., chain-of-thought starts helping)</li>
      <li>Composite tasks need every sub-step right — a sharp threshold even if each step improves smoothly</li>
      <li>Downstream usefulness genuinely jumps</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Emergence is a mirage</div>
    <ul>
      <li>Schaeffer et al. (2023): with continuous metrics (log-likelihood, edit distance) most "jumps" become smooth curves</li>
      <li>Exact-match on a long answer is the product of many per-token accuracies</li>
      <li>Too few model sizes make curves look like steps</li>
    </ul>
  </div>
</div>
</div>

Both are partly right: underlying per-token competence usually improves smoothly, and task-level success can still rise sharply because it requires many things to go right at once. If per-token accuracy is p and a task needs 20 tokens all correct, success is p²⁰ — 12% at p = 0.9, 82% at p = 0.99. Plan with continuous metrics; expect threshold-like behaviour in the tasks users see.

## Implications for Practice

1. **Given a compute budget**, use scaling laws to pick N and D — then shift towards a smaller model and more tokens if it will be served heavily.
2. **Run small proxies first**: fit loss curves on 10⁶–10⁸× smaller runs and extrapolate data mixes, architectures and hyperparameters (Llama 3 and DeepSeek both report doing this).
3. **Data quality moves the whole curve**: better filtering (FineWeb-Edu, DCLM) is equivalent to more compute.
4. **Watch the other axes**: RL and test-time compute now buy capability that pre-training compute alone would not.
5. **Measure smoothly**: track log-likelihoods of correct answers, not only accuracy, to see progress before it shows up as a "jump".

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Pre-training scaling laws are smooth because next-token loss is an average over trillions of predictions. RL rewards are sparse, per-episode and task-specific. Why should RL also scale predictably — and what would it take for test-time compute to have a clean "law" of its own?</p>
</div>

## Key Takeaways

- Loss falls as a power law in N, D and C; Chinchilla's compute-optimal split is ~20 tokens per parameter (Kaplan's different answer was mostly methodology).
- The loss surface is flat near the optimum, so smaller models trained much longer cost little loss and much less to serve — hence 1,000–60,000 tokens per parameter for small 2025 models.
- Repeated data, low precision, data scarcity and sparsity have their own scaling laws.
- Test-time compute and RL compute are new, predictable scaling axes.
- Emergence is largely a property of discontinuous metrics on composite tasks; the underlying improvement is usually smooth.

## Check Yourself

<details class="check">
<summary>What is the Chinchilla-optimal model size for 10²⁴ FLOPs, at 20 tokens per parameter?</summary>
<p>C = 6ND = 6N(20N) = 120N², so N = √(10²⁴/120) ≈ 91B parameters and D ≈ 1.8T tokens.</p>
</details>

<details class="check">
<summary>Why are 2025 small models trained on tens of thousands of tokens per parameter if that is "compute-suboptimal"?</summary>
<p>Training compute is a one-time cost; serving cost scales with parameters × tokens served for the model's lifetime. Spending extra training compute to make a small model better lowers total cost when it will serve many tokens — and the loss keeps improving, just slowly.</p>
</details>

<details class="check">
<summary>A task requires 30 consecutive correct tokens. Per-token accuracy rises from 0.95 to 0.99 across model sizes. What happens to task accuracy?</summary>
<p>0.95³⁰ ≈ 0.21 → 0.99³⁰ ≈ 0.74. A modest, smooth per-token gain looks like a sharp emergence on the task metric.</p>
</details>

<details class="check">
<summary>According to data-constrained scaling, what should you do when you have only 2T unique tokens and budget for 8T?</summary>
<p>Repeat the data for ~4 epochs — Muennighoff et al. found up to ~4 epochs costs little versus fresh data — and/or spend some budget on synthetic rephrasings and a larger model, since returns from further repetition fall off.</p>
</details>

<details class="check">
<summary>Why did Kumar et al. find that heavily over-trained models suffer more from post-training quantization?</summary>
<p>Training on more tokens packs more information into each weight, so the same rounding error destroys more of it. This matches the observation that Llama 3 (15T tokens) degrades more under low-bit quantization than Llama 2 (2T).</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Scaling Laws for Neural Language Models | Kaplan et al., 2020 | [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361) | Power laws in N, D, C |
| Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann et al., 2022 | [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556) | 20 tokens per parameter, parametric loss |
| Chinchilla Scaling: A Replication Attempt | Besiroglu et al. (Epoch AI), 2024 | [arxiv.org/abs/2404.10102](https://arxiv.org/abs/2404.10102) | Re-fitted loss coefficients |
| Resolving Discrepancies in Compute-Optimal Scaling of Language Models | Porian et al., 2024 | [arxiv.org/abs/2406.19146](https://arxiv.org/abs/2406.19146) | Why Kaplan and Chinchilla differ |
| Beyond Chinchilla-Optimal: Accounting for Inference | Sardana & Frankle, 2023 | [arxiv.org/abs/2401.00448](https://arxiv.org/abs/2401.00448) | Inference-aware sizing |
| Scaling Data-Constrained Language Models | Muennighoff et al., 2023 | [arxiv.org/abs/2305.16264](https://arxiv.org/abs/2305.16264) | Repeated epochs |
| Scaling Laws for Precision | Kumar et al., 2024 | [arxiv.org/abs/2411.04330](https://arxiv.org/abs/2411.04330) | Precision and quantization |
| Will we run out of data? | Villalobos et al. (Epoch AI), 2024 | [arxiv.org/abs/2211.04325](https://arxiv.org/abs/2211.04325) | Data stock projections |
| Emergent Abilities of Large Language Models | Wei et al., 2022 | [arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682) | Emergence catalogue |
| Are Emergent Abilities of LLMs a Mirage? | Schaeffer et al., 2023 | [arxiv.org/abs/2304.15004](https://arxiv.org/abs/2304.15004) | Metric-induced emergence |
| Scaling LLM Test-Time Compute Optimally | Snell et al., 2024 | [arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314) | Test-time compute allocation |
| The Art of Scaling Reinforcement Learning Compute for LLMs | Khatri et al. (Meta), 2025 | [arxiv.org/abs/2510.13786](https://arxiv.org/abs/2510.13786) | Predictable RL scaling |

---

**Next:** [Chapter 33 — Mixture of Experts](./33_mixture_of_experts.md) — how to add parameters without adding compute per token.

[← Back to Table of Contents](./README.md)
