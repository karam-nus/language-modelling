[← Back to Table of Contents](./README.md)

# Chapter 31 — Reasoning Models

> *"Chain-of-thought doesn't teach the model new facts — it teaches the model to use the facts it already knows, step by step."*

## The Reasoning Gap

Standard LLMs generate token by token, choosing the most likely next token. This works well for fluent text but struggles with tasks that require **multi-step reasoning**:

- "What is 347 × 892?" — requires carrying digits across steps
- "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?" — requires logical chaining
- "Write a function that finds the longest increasing subsequence" — requires algorithm design

**Reasoning models** address this by generating intermediate reasoning steps before the final answer.

## Chain-of-Thought (CoT) Prompting

Wei et al. (2022) showed that adding "Let's think step by step" dramatically improves math and logic:

<div class="diagram">
<div class="diagram-title">Standard vs Chain-of-Thought</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Standard Prompting</div>
    <ul>
      <li>Q: Roger has 5 balls. He buys 2 cans of 3 balls each. How many does he have?</li>
      <li>A: 11 ✗</li>
      <li>(No reasoning shown, just guesses)</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">CoT Prompting</div>
    <ul>
      <li>Q: ...same question...</li>
      <li>A: Roger started with 5 balls. He bought 2 cans × 3 balls = 6 balls. Total: 5 + 6 = 11. ✓</li>
      <li>(Step-by-step reasoning leads to correct answer)</li>
    </ul>
  </div>
</div>
</div>

```python
# Zero-shot CoT — just append the magic phrase
prompt = """Q: A store had 45 apples. They sold 12 on Monday and received 
a shipment of 30 on Tuesday. Then sold 25 on Wednesday. How many remain?

Let's think step by step:"""

# Few-shot CoT — provide worked examples in the prompt
few_shot_prompt = """Q: If there are 3 cars in the parking lot and 2 more arrive, 
how many cars are there?
A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: {actual_question}
A:"""
```

CoT works because it:
1. **Decomposes** complex problems into simpler sub-problems
2. **Allocates more compute**: more tokens = more FLOPs per problem
3. **Creates intermediate representations** the model can condition on
4. Only effective at **≥ ~60B parameters** (smaller models generate incoherent chains)

## Self-Consistency

Sample multiple chain-of-thought reasoning paths and take the **majority vote**:

<div class="diagram">
<div class="diagram-title">Self-Consistency (Wang et al., 2022)</div>
<div class="flow">
  <div class="flow-step accent">Same question asked N times with temperature > 0</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">Path 1: ... 5 + 6 = 11 → Answer: 11</div>
  <div class="flow-step green">Path 2: ... 5 + 6 = 11 → Answer: 11</div>
  <div class="flow-step red">Path 3: ... 5 + 3 = 8 → Answer: 8 (wrong reasoning)</div>
  <div class="flow-step green">Path 4: ... 5 + 6 = 11 → Answer: 11</div>
  <div class="flow-step red">Path 5: ... 5 + 6 = 12 → Answer: 12 (arithmetic error)</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">Majority vote: 11 wins (3/5) → Final answer: 11 ✓</div>
</div>
</div>

```python
import collections

def self_consistency(model, prompt, n_samples=10, temperature=0.7):
    """Generate multiple reasoning paths and majority vote."""
    answers = []
    for _ in range(n_samples):
        response = model.generate(prompt, temperature=temperature, max_tokens=512)
        answer = extract_final_answer(response)  # parse the numeric/categorical answer
        answers.append(answer)
    
    # Majority vote
    counter = collections.Counter(answers)
    return counter.most_common(1)[0][0]
```

Self-consistency improves GSM8K accuracy by 5–15% over single-sample CoT.

## Test-Time Compute Scaling (o1-style)

The breakthrough insight: **scaling compute at inference time** (more thinking tokens) can be as effective as scaling model size:

<div class="diagram">
<div class="diagram-title">Two Dimensions of Scaling</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Pre-Training Scaling</div>
    <ul>
      <li>More parameters, more data</li>
      <li>Fixed compute per token at inference</li>
      <li>Expensive to scale (months of GPU time)</li>
      <li>Improves general knowledge + capabilities</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Test-Time Scaling</div>
    <ul>
      <li>Same model, more inference tokens</li>
      <li>Variable compute per problem (think harder on hard problems)</li>
      <li>Cheap to scale (just generate more tokens)</li>
      <li>Improves reasoning on specific problems</li>
    </ul>
  </div>
</div>
</div>

### How o1 Works (Conceptual)

OpenAI's o1 family generates long internal reasoning chains before answering:

<div class="diagram">
<div class="diagram-title">o1 Reasoning Process</div>
<div class="flow">
  <div class="flow-step accent">User question received</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">[Internal thinking — not shown to user]</div>
  <div class="flow-step green">Break problem into sub-problems</div>
  <div class="flow-step green">Try approach A → hit dead end → backtrack</div>
  <div class="flow-step green">Try approach B → partial progress</div>
  <div class="flow-step green">Verify intermediate results</div>
  <div class="flow-step green">Complete the solution</div>
  <div class="flow-step green">Self-check: does the answer make sense?</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">[Summary answer shown to user]</div>
</div>
</div>

Key capabilities trained into reasoning models:
- **Backtracking**: "Wait, that's wrong. Let me try a different approach."
- **Verification**: "Let me check: 347 × 892 = 309,524. Verify: 892 × 300 = 267,600, 892 × 47 = 41,924. Sum: 309,524. ✓"
- **Self-reflection**: "This approach is getting too complicated. There might be a simpler way."

## Process Reward Models vs Outcome Reward Models

<div class="diagram">
<div class="diagram-title">Reward Model Types</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Outcome Reward Model (ORM)</div>
    <ul>
      <li>Scores only the final answer</li>
      <li>Correct answer → reward = 1, wrong → 0</li>
      <li>Sparse signal — hard to learn from</li>
      <li>Can't distinguish lucky guesses from good reasoning</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Process Reward Model (PRM)</div>
    <ul>
      <li>Scores each reasoning step independently</li>
      <li>Step 1: correct ✓, Step 2: correct ✓, Step 3: error ✗</li>
      <li>Dense signal — model learns where it went wrong</li>
      <li>Used in o1 and similar systems</li>
    </ul>
  </div>
</div>
</div>

Training a PRM requires **step-level annotations** (expensive) or automated verification (for math/code where answers can be checked programmatically).

## Open-Source Reasoning Models

| Model | Approach | Key Innovation |
|-------|---------|---------------|
| **DeepSeek-R1** | RL-trained reasoning | Long CoT via RL, open weights, distillation to smaller models |
| **QwQ (Qwen)** | Long reasoning chains | Extended thinking with self-reflection |
| **Marco-o1** | Open replication of o1-style | Process reward + MCTS search |
| **s1 (simple test-time scaling)** | Budget forcing | Control reasoning length via "Wait" token injection |

### DeepSeek-R1

DeepSeek-R1 demonstrated that **reasoning can emerge from pure RL** without supervised CoT data:

1. Start with DeepSeek-V3 base model
2. Apply RL (GRPO) with only outcome verification (correct/incorrect)
3. The model spontaneously learns to reason, self-verify, and backtrack
4. Distill the reasoning ability into smaller models (1.5B, 7B, 14B, 32B, 70B)

## Budget Forcing: Controlling Reasoning Depth

Not every question needs deep reasoning. Budget forcing lets you control how much the model "thinks":

```python
# Conceptual: control reasoning depth
def generate_with_budget(model, prompt, thinking_budget="medium"):
    """
    thinking_budget: "low" (fast), "medium", "high" (thorough)
    """
    budgets = {"low": 256, "medium": 1024, "high": 4096}
    max_thinking_tokens = budgets[thinking_budget]
    
    response = model.generate(
        prompt,
        max_tokens=max_thinking_tokens + 512,  # thinking + answer
        stop=["</answer>"],
    )
    return extract_answer(response)
```

## The Reasoning Landscape

<div class="diagram">
<div class="diagram-title">Evolution of LLM Reasoning</div>
<div class="timeline">
  <div class="timeline-item green">
    <div class="timeline-date">2022</div>
    <div class="timeline-desc">Chain-of-Thought prompting (Wei et al.) — "Let's think step by step"</div>
  </div>
  <div class="timeline-item accent">
    <div class="timeline-date">2022</div>
    <div class="timeline-desc">Self-Consistency (Wang et al.) — majority vote over multiple CoT paths</div>
  </div>
  <div class="timeline-item purple">
    <div class="timeline-date">2023</div>
    <div class="timeline-desc">Tree of Thoughts — search over reasoning trees with LLM-guided evaluation</div>
  </div>
  <div class="timeline-item orange">
    <div class="timeline-date">2024</div>
    <div class="timeline-desc">o1 (OpenAI) — RL-trained internal reasoning with backtracking + verification</div>
  </div>
  <div class="timeline-item red">
    <div class="timeline-date">2025</div>
    <div class="timeline-desc">DeepSeek-R1, QwQ — open-source reasoning models, RL-based emergence of CoT</div>
  </div>
</div>
</div>

## What's Next

Reasoning helps models think better with their internal knowledge. But what if the model's knowledge is outdated or insufficient? The next chapter covers **Retrieval-Augmented Generation (RAG)** — grounding model outputs in external, up-to-date information.

[← Previous: Chapter 30 — Beyond Transformers — SSMs & Alternatives](./30_ssms_and_alternatives.md) · **Next: [Chapter 32 — Retrieval-Augmented Generation (RAG) →](./32_rag.md)**

---

*Last updated: April 2026*
