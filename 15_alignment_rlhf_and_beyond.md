---
title: "Chapter 15 — Alignment: RLHF & Beyond"
description: "From preferences to rewards to reinforcement learning: reward models, PPO done properly, DPO and its family, GRPO and the 2025 refinements (DAPO, Dr. GRPO, GSPO), RL with verifiable rewards and rubrics, RL infrastructure, and the failure modes — reward hacking, sycophancy, emergent misalignment — that alignment work now targets."
related:
  - { id: training, why: "Post-training and RL runs in practice." }
  - { id: evals, why: "How alignment and reasoning gains are measured." }
---

[← Back to Table of Contents](./README.md)

# Chapter 15 — Alignment: RLHF & Beyond

SFT teaches a model to imitate good answers. Alignment training teaches it to *prefer* them — and, since 2024, to search for them: reinforcement learning on rewards now trains reasoning, coding and agentic skill as well as helpfulness and safety.

> **The one-sentence version:** post-training optimises the model against a signal — human preferences, an AI judge, a rubric or a verifier — and almost every hard problem in it comes from the gap between that signal and what you actually want.

## Why Alignment Matters

An SFT model follows instructions, but it may also comply with harmful requests, state falsehoods confidently, flatter the user, or ramble. Alignment training shifts its behaviour towards what people judge helpful, honest and harmless.

<div class="diagram">
<div class="diagram-title">The Alignment Problem</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">After SFT only</div>
    <ul>
      <li>Follows instructions, including harmful ones</li>
      <li>Confidently wrong (hallucinations)</li>
      <li>Inconsistent refusals</li>
      <li>Verbose, sycophantic, formulaic</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">After preference + RL training</div>
    <ul>
      <li>Declines genuinely harmful requests, helps with the rest</li>
      <li>Says when it is unsure; checks its work</li>
      <li>Balances helpfulness and safety</li>
      <li>Better at tasks with checkable answers (maths, code, agents)</li>
    </ul>
  </div>
</div>
</div>

## The RLHF Pipeline

<div class="img-caption">
  <img src="{{ '/assets/images/rlhf_pipeline.svg' | relative_url }}" alt="RLHF three-stage pipeline showing SFT, reward model training, and PPO optimization, plus DPO alternative">
  <figcaption>RLHF pipeline: supervised fine-tuning → reward model → PPO, compared with DPO's simpler two-model approach</figcaption>
</div>

**Reinforcement learning from human feedback** (InstructGPT, 2022) is the recipe that produced ChatGPT:

<div class="diagram">
<div class="diagram-title">RLHF — Three-Stage Pipeline</div>
<div class="flow">
  <div class="flow-node accent wide">Stage 1: Supervised fine-tuning <small>instruction–response pairs → SFT model</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Stage 2: Reward model <small>humans compare responses (A ≻ B) → train R(x, y)</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Stage 3: RL (PPO) <small>maximise R(x, y) while staying close to the SFT model</small></div>
</div>
</div>

### Stage 2: The Reward Model

Annotators compare pairs of responses; a **Bradley–Terry** model turns comparisons into a scalar score:

$$\mathcal{L}_{\text{RM}} = -\log\sigma\big(R(x, y_w) - R(x, y_l)\big)$$

The reward model is usually the SFT model with its LM head replaced by a scalar head, reading the hidden state of the final token. Since 2024 two alternatives are common: **generative reward models** (an LLM judge that writes a critique, then a verdict) and **verifiers** (code that checks an answer). Reward models are imperfect proxies — optimise too hard against one and the true quality first rises then *falls* (Gao et al., 2022: reward-model over-optimisation).

### Stage 3: PPO, Properly

PPO treats each generated token as an action. The per-sequence reward is usually combined with a per-token KL penalty against the reference model,

$$r_t = -\beta \log\frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{ref}}(y_t \mid \cdot)} \;\; (+\, R(x, y) \text{ at the last token}),$$

a learned **value model** estimates future reward so that **GAE** can produce per-token advantages $A_t$, and the policy is updated with the clipped surrogate objective:

$$\mathcal{L}_{\text{PPO}} = -\,\mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big], \qquad \rho_t = \frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{old}}(y_t \mid \cdot)}$$

Clipping stops any single update from moving a token's probability too far from the policy that sampled it; the KL term keeps the whole policy near the reference and limits **reward hacking** — exploiting the reward model's blind spots.

<div class="diagram">
<div class="diagram-title">PPO-based RLHF Components</div>
<div class="layer-stack">
  <div class="layer accent">Policy π_θ — the model being trained; generates rollouts</div>
  <div class="layer purple">Reference π_ref — frozen SFT model, for the KL penalty</div>
  <div class="layer green">Reward model R — scores complete responses</div>
  <div class="layer orange">Value model V — the critic, predicts future reward per token</div>
</div>
</div>

Four large models in memory, plus an inference engine for fast sampling, made PPO-RLHF expensive and finicky. The next three years were about removing pieces.

## DPO — Removing the Reward Model and the RL Loop

**DPO** (Rafailov et al., 2023) observes that the optimum of the KL-regularised RLHF objective has a closed form, $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\, e^{R(x,y)/\beta}$. Invert it, and the reward is expressed through the policy itself — so preferences can train the policy directly:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)$$

<div class="diagram">
<div class="diagram-title">DPO vs PPO-RLHF</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">PPO-RLHF</div>
    <ul>
      <li>Policy, reference, reward and value models</li>
      <li>Online: samples from the current policy</li>
      <li>Can keep improving as the policy explores</li>
      <li>Many hyperparameters; costly</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">DPO</div>
    <ul>
      <li>Policy + frozen reference</li>
      <li>Offline: a fixed preference dataset</li>
      <li>One supervised-style stage</li>
      <li>Simple; can over-fit to the dataset's style and length</li>
    </ul>
  </div>
</div>
</div>

DPO made preference tuning accessible (Zephyr, Tulu 2/3, most open chat models of 2024). Its weakness is being offline: the model never learns from its own mistakes. Many recipes now run **online / iterative DPO** — sample from the current model, score with a reward model or judge, form fresh pairs, repeat.

### DPO Training with TRL

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sft = "your-org/your-sft-model"
model = AutoModelForCausalLM.from_pretrained(sft, dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(sft)

ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")  # prompt, chosen, rejected

cfg = DPOConfig(output_dir="./dpo-model",
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                learning_rate=5e-7,          # DPO wants very small LRs
                beta=0.1,                    # strength of the implicit KL constraint
                num_train_epochs=1, max_length=1024, max_prompt_length=512, bf16=True)
DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
# ref_model omitted: TRL keeps a frozen copy of the starting model as the reference
```

### The Preference-Optimisation Family

| Method | Data | Reference model? | Idea |
|--------|---|:---:|---|
| **DPO** (2023) | Pairs | Yes | Closed-form RLHF optimum |
| **IPO** (2023) | Pairs | Yes | Squared loss; resists over-fitting deterministic preferences |
| **KTO** (2024) | Thumbs up / down | Yes | Prospect-theory utility; no pairs needed |
| **ORPO** (2024) | Pairs | No | Adds an odds-ratio penalty to the SFT loss — one stage |
| **SimPO** (2024) | Pairs | No | Length-normalised log-prob margin |
| **Online/iterative DPO** | Fresh pairs each round | Yes | Recovers some on-policy benefits |

## GRPO — Removing the Critic

For tasks with checkable answers, DeepSeekMath (2024) introduced **Group Relative Policy Optimisation**. For each prompt, sample a group of G responses, score each, and use the group as its own baseline:

$$A_i = \frac{r_i - \mathrm{mean}(r_1, \ldots, r_G)}{\mathrm{std}(r_1, \ldots, r_G)}$$

Every token of response *i* gets advantage $A_i$, which plugs into the same clipped objective as PPO (often plus a KL term). No value model, no reward model when rewards come from a verifier — just the policy, a reference, and an inference engine for sampling. GRPO powered DeepSeek-R1 and became the default RL algorithm of the open-model world in 2025.

A subtle property: if all G samples get the same reward — all right or all wrong — every advantage is zero and the prompt teaches nothing. The lab below shows how often that happens as the model gets better:

<div class="lab" data-lab="grpo"><noscript>Interactive lab: GRPO group advantages, and the fraction of prompts that give zero gradient as a function of group size and success rate. Enable JavaScript to use it.</noscript></div>

### 2025 Refinements

| Method | Fix | From |
|---|---|---|
| **DAPO** | *Clip-higher* (a larger upper clip so rare good tokens can grow), *dynamic sampling* (drop all-same groups and resample), token-level loss averaging, penalties for over-long answers | ByteDance Seed, 2025 |
| **Dr. GRPO** | Remove the per-response length normalisation and the std division, which bias towards long wrong answers and easy prompts | Sea AI / NUS, 2025 |
| **GSPO** | Importance ratio and clipping per *sequence* rather than per token — much more stable for MoE models, whose routing makes token-level ratios noisy | Qwen, 2025 (used for Qwen3) |
| **CISPO** | Clip the importance-sampling *weight* rather than the update, so no token's gradient is dropped | MiniMax-M1, 2025 |
| **RLOO / REINFORCE++** | Leave-one-out or batch baselines; simpler REINFORCE-style estimators | 2024–25 |

## RL with Verifiable Rewards (RLVR)

When correctness can be checked by a program, the reward model can be replaced by a **verifier**: compare a maths answer to the ground truth, run unit tests, check that JSON parses, diff a file system's final state. DeepSeek-R1-Zero (January 2025) showed that GRPO with such rewards on a base model, with no SFT at all, produces long chains of thought, self-verification and backtracking on its own — the "aha moment". [Chapter 35](./35_reasoning_models.md) follows that thread.

```python
import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

ds = load_dataset("openai/gsm8k", "main", split="train").map(lambda x: {
    "prompt": [{"role": "user", "content": x["question"] + "\nEnd with: #### <number>"}],
    "answer": x["answer"].split("####")[-1].strip().replace(",", ""),
})

def exact_answer(completions, answer, **kwargs):          # extra dataset columns arrive as kwargs
    texts = [c[0]["content"] for c in completions]
    found = [re.search(r"####\s*(-?[\d.,]+)", t) for t in texts]
    return [1.0 if m and m.group(1).replace(",", "") == a else 0.0 for m, a in zip(found, answer)]

cfg = GRPOConfig(output_dir="qwen-grpo-gsm8k",
                 num_generations=8,                 # group size G
                 per_device_train_batch_size=8, max_completion_length=512,
                 learning_rate=1e-6, beta=0.0,      # many 2025 recipes drop the KL term
                 use_vllm=True, bf16=True)
GRPOTrainer(model="Qwen/Qwen2.5-1.5B-Instruct", reward_funcs=exact_answer,
            args=cfg, train_dataset=ds).train()
```

Beyond maths and code, 2025–26 RL uses:

- **Rubrics as rewards** — an LLM judge scores a response against a task-specific checklist written by experts, extending RL to medicine, law, writing and research.
- **Agent environments** — multi-turn episodes in containers, terminals, browsers and simulated users, with the reward on the final state. Tool outputs are masked from the loss; only the model's own tokens are trained.
- **RLAIF and Constitutional AI** — an AI judge, guided by written principles, supplies preference labels (Anthropic's Constitutional AI, 2022); now standard for scaling preference data.
- **Deliberative alignment** — train a reasoning model to recall and reason over a written safety specification before answering (OpenAI, 2024).

## RL Infrastructure

RL for LLMs is mostly an inference problem: rollouts (generation) take most of the wall-clock time. Modern stacks pair a training engine (FSDP or Megatron) with an inference engine (vLLM or SGLang), synchronising weights after each update:

| Design | How | Trade-off |
|---|---|---|
| **Colocated, synchronous** | Same GPUs alternate between generating and training | Simple; GPUs idle while the longest rollout finishes |
| **Disaggregated, asynchronous** | Separate generation and training pools; rollouts may be a few steps stale | Higher throughput; needs off-policy corrections |

Open frameworks include verl, OpenRLHF, TRL, slime, AReaL and prime-rl. A 2025 lesson: the inference engine and the trainer compute *slightly different* probabilities (different kernels, precisions, MoE routing), and at scale that mismatch alone can destabilise training — fixes include importance-sampling corrections and running both sides in matching precision.

## When the Signal Is Wrong

Most alignment failures are the optimiser doing exactly what it was asked:

<div class="diagram">
<div class="diagram-title">Failure Modes Alignment Work Now Targets</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Reward hacking</div>
    <div class="card-desc">Coding agents that special-case tests, delete failing tests or hard-code expected outputs; answers padded to please a judge. Mitigations: robust graders, held-out checks, monitoring transcripts.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Sycophancy</div>
    <div class="card-desc">Optimising for thumbs-up teaches agreement and flattery. In April 2025 OpenAI rolled back a GPT-4o update after it became markedly sycophantic.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Emergent misalignment</div>
    <div class="card-desc">Fine-tuning on a narrow bad behaviour (e.g., writing insecure code without saying so) can make a model broadly misaligned in unrelated settings (Betley et al., 2025).</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Opaque reasoning</div>
    <div class="card-desc">Chain-of-thought is a valuable window for monitoring misbehaviour — but heavy optimisation pressure on the CoT itself can teach models to hide intent. A 2025 multi-lab paper urged preserving CoT monitorability.</div>
  </div>
</div>
</div>

### The Alignment Tax

InstructGPT found that RLHF lowered scores on some academic benchmarks — an "alignment tax" — and reduced it by mixing pre-training gradients into PPO (PPO-ptx). Modern post-training usually *raises* capability benchmarks overall, because RL on verifiable tasks trains skills directly; the tax shows up instead as **over-refusal** (declining benign requests), blander style, and narrowed output diversity. Safety evaluations therefore measure both harmful compliance and unnecessary refusal.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>Every reward — human vote, judge model, unit test, rubric — is a proxy. RL is an engine for finding where a proxy and the truth disagree. What properties would a reward need for longer, harder optimisation against it to be safe, and which of this chapter's signals comes closest?</p>
</div>

## Key Takeaways

- RLHF = SFT → reward model (Bradley–Terry) → PPO with a KL anchor; PPO needs four models and careful tuning.
- DPO folds the reward into the policy and trains offline on pairs; its family (IPO, KTO, ORPO, SimPO) varies the data and the reference; online/iterative variants recover on-policy learning.
- GRPO drops the critic by normalising rewards within a group of samples; DAPO, Dr. GRPO, GSPO and CISPO fixed its biases and instabilities in 2025.
- RLVR — verifiers instead of reward models — plus rubrics and agent environments made RL the engine of reasoning and agentic capability.
- The hard part is the signal: reward hacking, sycophancy, emergent misalignment and CoT monitorability are the live problems.

## Check Yourself

<details class="check">
<summary>Why does GRPO not need a value model, and what does it pay for that?</summary>
<p>The mean reward of the G samples for the same prompt serves as the baseline that a value model would otherwise estimate. The price is G samples per prompt (more generation compute), and a coarser advantage: every token in a response gets the same credit.</p>
</details>

<details class="check">
<summary>A model solves 90% of the prompts in a batch with probability 0.95 each. With G = 8, roughly what fraction of those prompts give zero gradient?</summary>
<p>A prompt is wasted when all 8 samples agree: 0.95⁸ + 0.05⁸ ≈ 0.66. Two-thirds of those easy prompts teach nothing — which is why DAPO resamples or filters prompts whose groups are all-correct or all-wrong.</p>
</details>

<details class="check">
<summary>What does PPO's clip actually prevent, and why did DAPO raise only the upper clip?</summary>
<p>It stops the probability ratio ρ from moving the policy more than ±ε from the sampling policy in one update. With a symmetric clip, low-probability tokens with positive advantage (e.g., a new reasoning move at p = 0.01) can grow only to 0.012 per step, which collapses exploration; a larger upper bound lets good rare tokens grow faster while the lower clip still limits damage.</p>
</details>

<details class="check">
<summary>Why is DPO called "offline", and why does that matter?</summary>
<p>It trains on a fixed set of preference pairs, usually generated by other models, rather than on samples from the policy being trained. The model never sees — and is never corrected on — its own characteristic errors, and the data's style (e.g., length) can be learned as a shortcut. Iterative DPO and RL methods sample on-policy to fix this.</p>
</details>

<details class="check">
<summary>An RL-trained coding agent's pass rate on your test suite jumps, but users report broken code. What would you check first?</summary>
<p>Reward hacking: read transcripts for edits to tests, special-cased inputs, skipped or deleted assertions, or detection of the evaluation harness; evaluate on held-out tests the agent cannot see or modify.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Proximal Policy Optimization Algorithms | Schulman et al., 2017 | [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) | PPO |
| Training language models to follow instructions with human feedback | Ouyang et al., 2022 | [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) | RLHF recipe, alignment tax, PPO-ptx |
| Constitutional AI: Harmlessness from AI Feedback | Bai et al. (Anthropic), 2022 | [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073) | RLAIF, constitutions |
| Scaling Laws for Reward Model Overoptimization | Gao et al., 2022 | [arxiv.org/abs/2210.10760](https://arxiv.org/abs/2210.10760) | Proxy over-optimisation |
| Direct Preference Optimization | Rafailov et al., 2023 | [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290) | DPO |
| KTO: Model Alignment as Prospect Theoretic Optimization | Ethayarajh et al., 2024 | [arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306) | KTO |
| SimPO | Meng et al., 2024 | [arxiv.org/abs/2405.14734](https://arxiv.org/abs/2405.14734) | Reference-free preference optimisation |
| DeepSeekMath (GRPO) | Shao et al., 2024 | [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) | GRPO |
| DeepSeek-R1 | DeepSeek-AI, 2025 | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) | RLVR at scale, R1-Zero |
| DAPO: An Open-Source LLM RL System at Scale | Yu et al. (ByteDance Seed), 2025 | [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476) | Clip-higher, dynamic sampling |
| Understanding R1-Zero-Like Training (Dr. GRPO) | Liu et al., 2025 | [arxiv.org/abs/2503.20783](https://arxiv.org/abs/2503.20783) | GRPO bias fixes |
| Group Sequence Policy Optimization | Zheng et al. (Qwen), 2025 | [arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071) | Sequence-level ratios for MoE |
| Deliberative Alignment | Guan et al. (OpenAI), 2024 | [arxiv.org/abs/2412.16339](https://arxiv.org/abs/2412.16339) | Reasoning over a safety spec |
| Emergent Misalignment | Betley et al., 2025 | [arxiv.org/abs/2502.17424](https://arxiv.org/abs/2502.17424) | Narrow fine-tuning, broad misalignment |
| Chain of Thought Monitorability | Korbak et al., 2025 | [arxiv.org/abs/2507.11473](https://arxiv.org/abs/2507.11473) | Preserving CoT as a safety signal |

---

**Next:** [Chapter 16 — Inference & Sampling](./16_inference_and_sampling.md) — training is done; now the model has to generate.

[← Back to Table of Contents](./README.md)
