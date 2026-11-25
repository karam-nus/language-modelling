---
title: "Chapter 30 — Evaluation & Benchmarks"
description: "How LLMs are measured in 2026: perplexity done right, pass@k, the saturation of MMLU-era benchmarks, frontier reasoning evals (GPQA, HLE, FrontierMath, AIME, ARC-AGI), agentic evals (SWE-bench, Terminal-Bench, τ²-bench, OSWorld, GDPval, METR time horizons), long-context and multimodal evals, LMArena and its critiques, LLM-as-judge pitfalls, contamination, and a practical evaluation strategy."
---

[← Back to Table of Contents](./README.md)

# Chapter 30 — Evaluation & Benchmarks

Evaluation is how you know whether a training run, a quantization or a prompt change helped. It is also where the field fools itself most often: benchmarks saturate, leak into training data, reward the wrong thing, and depend on harness details nobody reports.

> **The one-sentence version:** a benchmark is only useful while it is unsaturated, uncontaminated and close to the behaviour you care about — so the frontier keeps moving to harder, fresher, more agentic tests, and your own task-specific evals matter more than any leaderboard.

## Perplexity

The most fundamental language-model metric — how surprised is the model by held-out text?

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i \mid x_{<i})\right)$$

- Lower is better; a model that is always certain and right has PPL = 1.
- Only comparable between models with the **same tokenizer** on the same text and context length; across tokenizers use **bits per byte** ([Chapter 11](./11_optimizers_and_loss_functions.md)).
- It tracks pre-training progress well and post-training quality poorly: RLHF often *raises* perplexity on web text while making the model far more useful.

Sliding-window perplexity, scoring each token with up to `max_length` tokens of context:

```python
import torch

@torch.no_grad()
def sliding_ppl(model, tok, text, max_length=4096, stride=1024):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    nll_sum, n_scored, prev_end = 0.0, 0, 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + max_length, ids.size(1))
        trg_len = end - prev_end                    # tokens not yet scored
        chunk = ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-trg_len] = -100                 # context-only positions are not scored again
        loss = model(chunk, labels=labels).loss     # mean over scored positions (after the internal shift)
        n = (labels[:, 1:] != -100).sum().item()
        nll_sum += loss.item() * n
        n_scored += n
        prev_end = end
        if end == ids.size(1):
            break
    return float(torch.exp(torch.tensor(nll_sum / n_scored)))
```

## Pass@k and Sampling-Based Metrics

For generative tasks with a checker (code tests, maths answers), **pass@k** is the probability that at least one of k samples is correct. Estimate it without bias from n ≥ k samples of which c are correct:

$$\text{pass@}k = 1 - \binom{n-c}{k} \Big/ \binom{n}{k}$$

For reasoning models, report **pass@1 averaged over several samples** (e.g., 8–64 at the recommended temperature) rather than a single greedy run — scores on 30-question sets like AIME swing by several points between seeds. Also report the **thinking budget** and **tokens used**: two models with equal accuracy can differ 5× in cost.

## From Saturated Benchmarks to Frontier Evals

The 2020–23 workhorses are now saturated for frontier models — they still discriminate between small models and catch broken checkpoints, but not between leaders:

| Benchmark (era) | What it tests | Status in 2026 |
|-----------|-------|--------|
| MMLU (2020), 57 subjects | Broad academic multiple choice | Saturated (~90%+); MMLU-Pro (10 options, harder) replaced it |
| HellaSwag, WinoGrande, ARC | Commonsense, science QA | Saturated; still useful for small/base models |
| GSM8K (2021) | Grade-school maths | Saturated; contamination widespread |
| HumanEval / MBPP (2021) | Short Python functions | Saturated; replaced by LiveCodeBench, SWE-bench |
| MT-Bench, AlpacaEval (2023) | LLM-judged chat quality | Superseded by Arena-Hard and live human arenas |

### Reasoning and Knowledge

| Benchmark | What it tests | Why it matters |
|---|---|---|
| **GPQA Diamond** (2023) | 198 graduate-level, "Google-proof" science questions | Long-lived expert-knowledge probe |
| **Humanity's Last Exam** (2025) | ~2,500 expert-written questions across fields | Designed to be hard for frontier models at release |
| **AIME / HMMT** (yearly) | Olympiad-style maths with integer answers | Fresh every year — natural contamination control |
| **FrontierMath** (2024) | Unpublished research-level maths problems | Private; probes genuine mathematical reasoning |
| **ARC-AGI-2** (2025) | Novel visual abstraction puzzles | Skill acquisition on unfamiliar tasks rather than knowledge |
| **SimpleQA** (2024) | Short factual questions | Measures hallucination and calibration |

### Coding and Agents

| Benchmark | What it tests |
|---|---|
| **LiveCodeBench** | Contest problems released after a model's cutoff |
| **SWE-bench Verified** (2024) | 500 human-validated GitHub issues: edit a real repository so hidden tests pass |
| **SWE-bench Pro / multilingual variants** (2025) | Harder, less contaminated, more languages |
| **Terminal-Bench** (2025) | Multi-step tasks in a real shell (build, debug, configure) |
| **τ²-bench** (2025) | Tool-using customer-service agents interacting with a simulated user, under policies |
| **BrowseComp** (2025) | Hard-to-find facts requiring persistent web browsing |
| **OSWorld** (2024) | Computer use across real desktop applications |
| **GDPval** (2025) | Real work products in 44 occupations, graded by industry experts |

**METR's time-horizon** measure summarises agentic progress in one number: the length of task (measured in how long it takes skilled humans) that a model completes with 50% reliability. In METR's 2025 analysis it had been doubling roughly every seven months since 2019 — the trend behind the shift from "answer a question" to "do a day's work" evaluations.

### Long Context, Multimodal and Safety

- **Long context**: RULER, HELMET, NoLiMa, OpenAI-MRCR — multi-needle and multi-hop retrieval at 128K–1M tokens; effective context is usually far below advertised ([Chapter 12](./12_mid_training.md)).
- **Multimodal**: MMMU (college-level multimodal questions), MathVista, ChartQA, DocVQA, Video-MME.
- **Instruction following**: IFEval and IFBench (verifiable format constraints).
- **Safety and dangerous capabilities**: refusal and over-refusal sets, jailbreak robustness, and frontier-lab evaluations of biological, cyber and autonomy risks under published safety frameworks.

## Human Preference: LMArena

**LMArena** (formerly Chatbot Arena, LMSYS) collects blind pairwise votes from users and fits a Bradley–Terry model to produce Elo-style ratings. It measures what people prefer in open-ended chat, which no static benchmark captures. Its known weaknesses, documented in *The Leaderboard Illusion* (2025): labs can test many private variants and publish only the best; votes reward style (length, formatting, confidence); and a model tuned for the arena may differ from the version shipped (the Llama 4 "experimental" variant episode in April 2025). Use arena ratings as one signal, with style-controlled rankings where available.

## LLM-as-Judge

Model judges scale evaluation of open-ended outputs (Arena-Hard, AlpacaEval, rubric grading). Their biases are well documented: **position** (prefer the first answer), **verbosity** (prefer longer answers), **self-preference** (prefer their own family's outputs) and sensitivity to formatting. Mitigations: swap answer order and average, length-control, use a different model family as judge, write explicit rubrics, and calibrate against a few hundred human labels.

## Running Evaluations

```bash
pip install "lm-eval[vllm]"

# Fast evaluation through vLLM, with the model's chat template
lm_eval --model vllm \
  --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.85 \
  --tasks mmlu_pro,gpqa_diamond_cot_zeroshot,ifeval \
  --apply_chat_template --batch_size auto --output_path results/

# Compare an original and a quantized checkpoint on identical settings
for m in meta-llama/Llama-3.1-8B-Instruct ./Llama-3.1-8B-Instruct-W4A16; do
  lm_eval --model vllm --model_args pretrained=$m --tasks gsm8k,ifeval \
          --apply_chat_template --output_path results/$(basename $m)
done
```

`lighteval`, OpenAI's `simple-evals`, Inspect (UK AISI) for agentic and safety evals, and SWE-bench's own harness cover what lm-eval does not. Whatever you use, **report the harness, prompt format, few-shot count, sampling settings and number of runs** — these routinely move scores by several points.

## Benchmark Limitations

<div class="diagram">
<div class="diagram-title">Why Benchmark Scores Mislead</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Contamination</div>
    <div class="card-desc">Test items leak into web-scale training data; models recall rather than reason. Fresh (AIME, LiveCodeBench) and private (FrontierMath) sets resist it.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Goodhart's law</div>
    <div class="card-desc">Once a benchmark is a target, labs optimise for it; high scores stop implying the general skill it was meant to sample.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Harness sensitivity</div>
    <div class="card-desc">Prompt format, chat template, answer extraction, thinking budget and tool access can each move results more than a model upgrade.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Evaluation awareness</div>
    <div class="card-desc">2025 studies found frontier models can often tell when they are being tested — a problem for safety evaluations that assume natural behaviour.</div>
  </div>
</div>
</div>

## Practical Evaluation Strategy

For your own models and applications:

1. **Build a task-specific eval first** — 100–500 real examples with automated checks or rubrics. It is worth more than any public leaderboard.
2. **Track perplexity / loss** on held-out data during training — cheap and catches regressions early.
3. **Run standard suites** (lm-eval, lighteval) to compare with baselines and catch capability loss after fine-tuning or quantization.
4. **Use LLM judges with care** for open-ended outputs; validate the judge against human labels.
5. **Read outputs.** Fifty transcripts reveal formatting bugs, reward hacking and template errors that aggregate scores hide.
6. **Measure cost and latency alongside quality** — tokens per solved task, not just accuracy.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Every hard benchmark of 2023 was saturated by 2025, and each replacement is more expensive to build and run — expert-written questions, live environments, human graders. If benchmark half-lives keep shrinking, what does a durable measurement of capability look like?</p>
</div>

## Key Takeaways

- Perplexity tracks training, not usefulness; compare only like-for-like, and use bits per byte across tokenizers.
- Report pass@1 over multiple samples, with thinking budget and tokens, for reasoning models.
- MMLU/GSM8K/HumanEval are saturated; 2026 frontier evaluation is GPQA, HLE, FrontierMath, AIME, ARC-AGI, SWE-bench, Terminal-Bench, τ²-bench, OSWorld, GDPval and METR time horizons.
- Arena ratings and LLM judges measure preference — useful, biased, gameable.
- Your own task evals, read outputs and cost-aware metrics beat leaderboard positions.

## Check Yourself

<details class="check">
<summary>A coding model produces 10 samples per problem; on one problem 3 pass. What is the unbiased pass@1 and pass@5 for that problem?</summary>
<p>pass@1 = 1 − C(7,1)/C(10,1) = 1 − 7/10 = 0.3. pass@5 = 1 − C(7,5)/C(10,5) = 1 − 21/252 ≈ 0.917.</p>
</details>

<details class="check">
<summary>Why are yearly competitions such as AIME useful benchmarks despite having only 30 problems?</summary>
<p>New problems appear every year, after most models' training cutoffs, so contamination is unlikely; and they are hard with exact answers. The small size means high variance, so average over many samples and report confidence.</p>
</details>

<details class="check">
<summary>Name three LLM-as-judge biases and one mitigation for each.</summary>
<p>Position bias → evaluate both orders and average. Verbosity bias → length-controlled scoring or rubrics that ignore length. Self-preference → use a judge from a different model family (or several).</p>
</details>

<details class="check">
<summary>What does METR's "50% time horizon" measure, and why is it useful?</summary>
<p>The human-expert duration of tasks a model completes with 50% success. It converts many agentic benchmarks into one interpretable, comparable number and shows a consistent exponential trend over time.</p>
</details>

<details class="check">
<summary>After RLHF, a model's WikiText perplexity rose from 6.0 to 7.5, but users prefer it strongly. Is something wrong?</summary>
<p>Not necessarily. Post-training shifts the model's distribution toward assistant-style responses and away from imitating web text, which raises perplexity on web text while improving the behaviour users care about. Perplexity is the wrong metric for post-trained chat quality.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Evaluating Large Language Models Trained on Code (Codex, pass@k) | Chen et al., 2021 | [arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374) | Unbiased pass@k |
| Measuring Massive Multitask Language Understanding (MMLU) | Hendrycks et al., 2020 | [arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300) | MMLU |
| GPQA | Rein et al., 2023 | [arxiv.org/abs/2311.12022](https://arxiv.org/abs/2311.12022) | Graduate-level QA |
| Humanity's Last Exam | Phan et al., 2025 | [arxiv.org/abs/2501.14249](https://arxiv.org/abs/2501.14249) | Frontier knowledge benchmark |
| SWE-bench | Jimenez et al., 2023 | [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770) | Repository-level coding |
| Measuring AI Ability to Complete Long Tasks | METR, 2025 | [arxiv.org/abs/2503.14499](https://arxiv.org/abs/2503.14499) | Time horizons |
| Chatbot Arena | Chiang et al., 2024 | [arxiv.org/abs/2403.04132](https://arxiv.org/abs/2403.04132) | Pairwise human preference |
| The Leaderboard Illusion | Singh et al., 2025 | [arxiv.org/abs/2504.20879](https://arxiv.org/abs/2504.20879) | Arena critiques |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Zheng et al., 2023 | [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685) | Judge biases |
| GDPval | OpenAI, 2025 | [openai.com/index/gdpval](https://openai.com/index/gdpval/) | Occupational task evaluation |
| ARC-AGI-2 | ARC Prize, 2025 | [arcprize.org](https://arcprize.org/arc-agi/2/) | Abstraction benchmark |
| lm-evaluation-harness | EleutherAI | [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Running evals |

---

**Next:** [Chapter 31 — Serving & Deployment](./31_serving_and_deployment.md) — getting a model in front of users, fast and affordably.

[← Back to Table of Contents](./README.md)
