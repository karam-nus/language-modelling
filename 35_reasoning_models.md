---
title: "Chapter 35 — Reasoning Models"
description: "From chain-of-thought prompting to RL-trained thinking: zero- and few-shot CoT, self-consistency, how o1 and DeepSeek-R1 changed training (RLVR, R1-Zero, cold starts), test-time compute strategies (parallel, sequential, search, Deep Think), thinking budgets and overthinking, interleaved thinking with tools, CoT faithfulness, and whether RL creates new abilities."
---

[← Back to Table of Contents](./README.md)

# Chapter 35 — Reasoning Models

In September 2024 OpenAI's o1 showed that a model trained with reinforcement learning to "think" — write a long private chain of reasoning before answering — gets steadily better the longer it thinks. Four months later DeepSeek-R1 showed how to do it in the open. By 2026 every frontier model is a reasoning model with an adjustable thinking budget, and the techniques of this chapter are standard.

> **The one-sentence version:** reasoning models are trained with RL on checkable problems to produce long chains of thought that explore, verify and backtrack — turning extra inference tokens into accuracy.

## The Reasoning Gap

A model emits one token per forward pass, with a fixed amount of computation per token. Problems that need many sequential steps — multi-digit arithmetic, proofs, debugging, planning — cannot be solved "in one go" by a fixed-depth network. Writing the intermediate steps out lets the model spend more forward passes on a problem and condition each step on the last.

## Chain-of-Thought Prompting

<div class="img-caption">
  <img src="{{ '/assets/images/chain_of_thought.svg' | relative_url }}" alt="Chain-of-thought prompting comparison versus standard prompting, plus test-time compute scaling with Best-of-N, PRM, and MCTS">
  <figcaption>CoT makes the model reason step by step; test-time compute scaling extends this with sampling and search over reasoning paths</figcaption>
</div>

Wei et al. (2022) showed that few-shot exemplars *with worked reasoning* dramatically improve large models on maths and logic; Kojima et al. (2022) found that simply appending "Let's think step by step" works zero-shot. The classic example from Wei et al.:

<div class="diagram">
<div class="diagram-title">Standard vs Chain-of-Thought (Wei et al., 2022)</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Standard prompting</div>
    <ul>
      <li>Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?</li>
      <li>A: The answer is 27. ✗</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Chain-of-thought prompting</div>
    <ul>
      <li>Q: (same question)</li>
      <li>A: The cafeteria had 23 apples originally. They used 20, so 23 − 20 = 3. They bought 6 more, so 3 + 6 = 9. The answer is 9. ✓</li>
    </ul>
  </div>
</div>
</div>

CoT works because it decomposes problems, spends more compute (more tokens = more forward passes) and creates intermediate results the model can condition on. With prompting alone it helped mainly very large models (~60–100B+); smaller models produced fluent but wrong chains.

## Self-Consistency

Sample several reasoning paths at temperature > 0 and take a **majority vote** over final answers (Wang et al., 2022). Errors in different paths tend to disagree; correct paths agree. The paper reported up to +18 points on GSM8K for PaLM 540B over single-sample CoT.

```python
from collections import Counter

def self_consistency(generate, extract_answer, prompt, n=16, temperature=0.7):
    answers = [extract_answer(generate(prompt, temperature=temperature)) for _ in range(n)]
    answers = [a for a in answers if a is not None]
    return Counter(answers).most_common(1)[0][0] if answers else None
```

How much does voting buy? If each sample is right with probability p and wrong answers scatter, majority voting climbs towards 100% as n grows when p beats the most common wrong answer — and falls towards 0% when it doesn't. Try it:

<div class="lab" data-lab="tts"><noscript>Interactive lab: accuracy of majority voting and best-of-N with a verifier as the number of samples grows. Enable JavaScript to use it.</noscript></div>

## RL-Trained Reasoning: o1 and DeepSeek-R1

Prompting elicits reasoning; **reinforcement learning** trains it. The recipe, made public by DeepSeek-R1 (January 2025):

<div class="diagram">
<div class="diagram-title">How DeepSeek-R1 Was Trained</div>
<div class="flow">
  <div class="flow-node accent wide">R1-Zero: GRPO directly on the DeepSeek-V3 base model <small>rewards: correct final answer (maths checker, code tests) + output format — no supervised reasoning data</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Observed: responses grow from hundreds to thousands of tokens; self-verification and backtracking ("wait…") appear on their own <small>but language mixing and poor readability</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">R1: small "cold-start" SFT on readable long-CoT examples → reasoning RL → rejection-sample ~800K SFT examples → RL again for helpfulness and harmlessness</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">Distil: fine-tune Qwen2.5 and Llama 3 bases (1.5B–70B) on R1 outputs</div>
</div>
</div>

Behaviours that RL reinforces, because they raise the probability of a correct final answer:
- **Exploration and backtracking**: "Wait, that's wrong. Let me try a different approach."
- **Verification**: "Check: 347 × 892 = 347 × 900 − 347 × 8 = 312,300 − 2,776 = 309,524. ✓"
- **Decomposition and planning**: listing cases, setting up equations before solving.

OpenAI has not published o1's training details; whether it used process rewards, outcome rewards or search during training is not public.

### Outcome vs Process Rewards

<div class="diagram">
<div class="diagram-title">Reward Model Types</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Outcome rewards (ORM / verifier)</div>
    <ul>
      <li>Score only the final answer</li>
      <li>Cheap and hard to game when a checker exists</li>
      <li>Sparse: no credit for good partial reasoning</li>
      <li>What R1 used — and it was enough</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Process rewards (PRM)</div>
    <ul>
      <li>Score each reasoning step</li>
      <li>Dense signal; useful for search and reranking</li>
      <li>Expensive labels (PRM800K) or noisy automatic labels</li>
      <li>Prone to reward hacking when optimised against directly</li>
    </ul>
  </div>
</div>
</div>

## Reasoning Models in 2025–26

| Model / family | Year | Notes |
|-------|---------|---------------|
| OpenAI o1 → o3 → GPT-5.x | 2024–26 | Hidden reasoning; adjustable reasoning effort; GPT-5 routes between fast and thinking modes |
| DeepSeek-R1, R1-0528 → V3.1/V3.2 → V4 | 2025–26 | Open; later versions merge thinking and non-thinking modes in one model |
| Qwen QwQ-32B → Qwen3 → Qwen3.5/3.6 | 2025–26 | Hybrid thinking switched per request (`enable_thinking`); separate "Thinking" checkpoints |
| Gemini 2.5 → 3 Deep Think | 2025 | Thinking budgets; Deep Think explores parallel lines of reasoning |
| Claude extended thinking (4 → 5 family) | 2025–26 | Effort control; interleaved thinking between tool calls |
| gpt-oss | 2025 | Open; reasoning effort low / medium / high set in the system prompt |
| Kimi K2 Thinking, MiniMax-M2, GLM-4.5/5 | 2025–26 | Open "agentic thinkers" that interleave reasoning with hundreds of tool calls |
| s1, OpenThinker, Phi-4-reasoning | 2025 | Small-data or small-model reasoning via distillation and SFT |

A milestone: in July 2025 experimental models from Google DeepMind and OpenAI reached gold-medal level on the International Mathematical Olympiad, writing natural-language proofs under contest time limits.

## Test-Time Compute Strategies

| Strategy | How | Examples |
|---|---|---|
| **Longer thinking (sequential)** | One chain, more tokens: explore, check, revise | o1/R1-style thinking; s1 "budget forcing" |
| **Parallel sampling** | N independent chains, then vote or pick with a verifier | Self-consistency, best-of-N, GPT-5 Pro / Deep Think-style parallel thinking |
| **Search** | Branch at steps, score partial chains (PRM or value), expand the best | Tree of Thoughts, MCTS variants, beam search over steps |
| **Tool use inside reasoning** | Run code, search, call APIs mid-thought | Interleaved thinking in agentic models |

Snell et al. (2024) found that the best strategy depends on difficulty: for easy problems, sequential revision is most compute-efficient; for hard ones, parallel sampling with a verifier helps more — and adapting the budget per prompt beat a 14× larger model at equal compute in their setting.

## Controlling How Much a Model Thinks

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
msgs = [{"role": "user", "content": "Is 9,991 prime?"}]

fast  = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
think = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
# gpt-oss: put "Reasoning: high" (or low / medium) in the system message
# Hosted APIs: a reasoning-effort or thinking-budget parameter
```

**Budget forcing** (s1, 2025) controls length at decode time: to stop early, force the end-of-thinking token; to think longer, suppress it and append "Wait" — which often makes the model re-check and fix its answer.

### Overthinking and Underthinking

More thinking is not always better. Reasoning models often spend thousands of tokens on trivial questions ("overthinking"), and sometimes abandon promising lines too early ("underthinking"). Remedies in 2025–26 training: length-aware rewards, adaptive thinking (the model decides whether to think), and per-request effort settings. For serving, thinking tokens dominate cost and latency ([Chapter 31](./31_serving_and_deployment.md)).

## Can We Trust the Chain of Thought?

- **Faithfulness**: Anthropic (2025) found reasoning models often fail to mention hints that actually changed their answers — the visible chain is not a complete account of the computation.
- **Monitorability**: chains of thought still reveal a great deal (including reward-hacking intentions), so labs argue for preserving them and not training directly against their content ([Chapter 15](./15_alignment_rlhf_and_beyond.md)).
- **Hidden vs visible**: some providers return only summaries of the reasoning; open models expose it fully.

## Does RL Create New Abilities?

A live 2025 debate. Yue et al. found that at large k, the **base** model's pass@k on maths and code matched or exceeded the RL-trained model's — suggesting RL mostly *sharpens* the distribution towards solutions the base model could already find, rather than adding new ones. Counter-evidence (e.g., prolonged-RL studies) showed longer, more diverse RL runs can reach problems the base model never solves. The practical consensus: pre-training and mid-training set what is reachable ([Chapter 12](./12_mid_training.md)); RL makes it reliable and fast to reach.

<div class="callout ponder">
<span class="callout-title">Ponder</span>
<p>If a reasoning model gets better with every extra thousand thinking tokens, the cost of an answer is no longer fixed by the model — it is chosen per query. Who should choose it: the user, the model, or the provider? And what would a good "is this worth thinking about?" signal look like?</p>
</div>

## Key Takeaways

- Chain-of-thought (few-shot or "think step by step") lets a fixed-depth model spend more computation per problem; self-consistency votes over samples.
- o1 and DeepSeek-R1 trained reasoning with RL on verifiable rewards; R1-Zero showed long, self-correcting chains emerge without supervised reasoning data.
- Test-time compute is spent sequentially (longer thinking), in parallel (sampling + voting/verifiers) or by search; the best mix depends on difficulty.
- Thinking budgets, hybrid thinking modes and budget forcing control cost; overthinking is a real waste.
- Chains of thought are useful but not fully faithful; RL largely sharpens abilities the base model already has.

## Check Yourself

<details class="check">
<summary>Each sample is correct with probability 0.6 and every wrong sample gives a different wrong answer. What happens to majority-vote accuracy as N grows?</summary>
<p>It rises towards 100%: the correct answer gets about 0.6N votes while each wrong answer gets about one, so the correct answer wins with probability approaching 1. If wrong answers were concentrated on one value with probability above 0.6, voting would converge on that wrong answer instead.</p>
</details>

<details class="check">
<summary>What was the difference between DeepSeek-R1-Zero and DeepSeek-R1?</summary>
<p>R1-Zero applied RL with rule-based rewards directly to the base model, with no SFT — proving reasoning can emerge from RL alone but producing hard-to-read, language-mixed chains. R1 added a small cold-start SFT stage, further RL, rejection-sampled SFT data and a final general RL stage for readability and helpfulness.</p>
</details>

<details class="check">
<summary>Why are outcome rewards less vulnerable to reward hacking than process reward models in RLVR?</summary>
<p>A programmatic checker on the final answer (tests pass, number matches) is hard to satisfy without solving the problem. A learned PRM scores the <em>look</em> of steps and can be exploited by text that resembles good reasoning without being correct.</p>
</details>

<details class="check">
<summary>If RL mainly sharpens existing abilities, why does it still matter so much in practice?</summary>
<p>Users and agents mostly get one attempt (pass@1). Turning a solution the base model finds in 1 of 100 samples into one it finds on the first try is a huge practical gain — and RL also teaches efficient habits (verification, decomposition) that make long agentic tasks feasible.</p>
</details>

<details class="check">
<summary>How does s1's budget forcing make a model think longer?</summary>
<p>When the model tries to emit its end-of-thinking delimiter before the budget is used, the decoder suppresses it and appends "Wait", prompting the model to continue reasoning — often re-checking and correcting its answer.</p>
</details>

## References

| Source | Author / Org, Year | Link | What this chapter takes from it |
|---|---|---|---|
| Chain-of-Thought Prompting Elicits Reasoning in LLMs | Wei et al., 2022 | [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903) | Few-shot CoT |
| Large Language Models are Zero-Shot Reasoners | Kojima et al., 2022 | [arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916) | "Let's think step by step" |
| Self-Consistency Improves Chain of Thought Reasoning | Wang et al., 2022 | [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171) | Majority voting |
| Tree of Thoughts | Yao et al., 2023 | [arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601) | Search over reasoning |
| Let's Verify Step by Step | Lightman et al., 2023 | [arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050) | Process supervision |
| Learning to Reason with LLMs (o1) | OpenAI, 2024 | [openai.com/index/learning-to-reason-with-llms](https://openai.com/index/learning-to-reason-with-llms/) | Test-time compute scaling |
| Scaling LLM Test-Time Compute Optimally | Snell et al., 2024 | [arxiv.org/abs/2408.03314](https://arxiv.org/abs/2408.03314) | Compute-optimal inference |
| DeepSeek-R1 | DeepSeek-AI, 2025 | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948) | R1-Zero, R1 pipeline, distillation |
| s1: Simple test-time scaling | Muennighoff et al., 2025 | [arxiv.org/abs/2501.19393](https://arxiv.org/abs/2501.19393) | Budget forcing |
| Does RL Really Incentivize Reasoning Capacity Beyond the Base Model? | Yue et al., 2025 | [arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837) | pass@k analysis |
| Reasoning Models Don't Always Say What They Think | Chen et al. (Anthropic), 2025 | [arxiv.org/abs/2505.05410](https://arxiv.org/abs/2505.05410) | CoT faithfulness |
| Qwen3 Technical Report | Qwen Team, 2025 | [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388) | Hybrid thinking modes |

---

**Next:** [Chapter 36 — Retrieval-Augmented Generation](./36_rag.md) — reasoning helps a model use what it knows; retrieval gives it what it does not.

[← Back to Table of Contents](./README.md)
