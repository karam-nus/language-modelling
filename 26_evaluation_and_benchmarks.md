[← Back to Table of Contents](./README.md)

# Chapter 26 — Evaluation & Benchmarks

> *"If you can't measure it, you can't improve it. But if you measure the wrong thing, you optimize for the wrong outcome."*

## Perplexity

The most fundamental language model metric — how surprised is the model by the test data?

$$
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i | x_{<i})\right)
$$

- **Lower is better** — a perfect model has PPL = 1
- Typical values: GPT-2 ~29 on WikiText-103, LLaMA-3-8B ~6
- Only meaningful for comparing models on the **same tokenizer + dataset**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, tokenizer, text, max_length=2048, stride=512):
    """Compute perplexity with sliding window."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)
    
    nlls = []
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - (begin if begin == 0 else begin + max_length - stride)
        
        input_chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
        nlls.append(outputs.loss * target_len)
        
        if end == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()
```

## Major Benchmarks

<div class="diagram">
<div class="diagram-title">LLM Benchmark Categories</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card green">
    <div class="card-title">Knowledge & Reasoning</div>
    <div class="card-desc">MMLU, ARC, HellaSwag, WinoGrande, TruthfulQA, GPQA</div>
  </div>
  <div class="diagram-card accent">
    <div class="card-title">Math & Code</div>
    <div class="card-desc">GSM8K, MATH, HumanEval, MBPP, LiveCodeBench</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Chat & Instruction</div>
    <div class="card-desc">MT-Bench, AlpacaEval, Chatbot Arena (ELO), WildBench</div>
  </div>
</div>
</div>

### Knowledge & Reasoning Benchmarks

| Benchmark | Tasks | Metric | What It Tests |
|-----------|-------|--------|--------------|
| **MMLU** | 57 subjects, 14K questions | Accuracy | Broad academic knowledge (STEM, humanities, social sciences) |
| **MMLU-Pro** | Harder MMLU with 10 choices | Accuracy | Deeper reasoning with reduced guessing |
| **ARC** (Challenge) | Grade-school science, 2.5K | Accuracy | Scientific reasoning |
| **HellaSwag** | 10K sentence completions | Accuracy | Common-sense reasoning, adversarial |
| **WinoGrande** | 44K pronoun resolution | Accuracy | Coreference / common sense |
| **TruthfulQA** | 817 questions | MC accuracy | Resistance to common misconceptions |
| **GPQA** | Graduate-level science | Accuracy | Expert knowledge (PhD difficulty) |

### Math & Code Benchmarks

| Benchmark | Format | Metric | What It Tests |
|-----------|--------|--------|--------------|
| **GSM8K** | 8.5K grade-school math word problems | Solve rate | Multi-step arithmetic reasoning |
| **MATH** | 12.5K competition math problems (5 levels) | Solve rate | Advanced mathematical reasoning |
| **HumanEval** | 164 Python function completions | pass@k | Code generation correctness |
| **MBPP** | 974 Python problems | pass@k | Basic Python programming |
| **LiveCodeBench** | Continuously updated contest problems | pass@k | Non-contaminated coding ability |

### Chat & Human Evaluation

| Benchmark | Method | Metric | What It Tests |
|-----------|--------|--------|--------------|
| **MT-Bench** | GPT-4 judges 80 multi-turn conversations | 1–10 score | Instruction following, multi-turn coherence |
| **AlpacaEval 2** | GPT-4 compares outputs to reference | Win rate (LC) | Instruction following quality |
| **Chatbot Arena** | Humans vote on blind pairwise comparisons | ELO rating | Overall chat preference (gold standard) |
| **WildBench** | Real user queries, LLM-judged | Win rate | Challenging real-world tasks |

## The Open LLM Leaderboard

HuggingFace's [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) evaluates models using lm-eval-harness:

**Leaderboard v2 suite** (as of 2024):
- MMLU-Pro (knowledge)
- GPQA (expert knowledge)
- MuSR (multi-step reasoning)
- MATH (mathematical reasoning)
- BBH (Big-Bench Hard — diverse reasoning)
- IFEval (instruction following)

## Running Evaluations with lm-eval-harness

```bash
# Install
pip install lm-eval

# Evaluate a model on MMLU
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path results/

# Multiple benchmarks
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
    --tasks mmlu,hellaswag,arc_challenge,winogrande,gsm8k \
    --batch_size auto \
    --output_path results/

# Evaluate a GPTQ quantized model
lm_eval --model hf \
    --model_args pretrained=TheBloke/Llama-3-8B-GPTQ,autogptq=True \
    --tasks mmlu \
    --batch_size auto
```

## Benchmark Limitations

<div class="diagram">
<div class="diagram-title">Known Issues with Benchmarks</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Contamination</div>
    <ul>
      <li>Training data may contain benchmark questions</li>
      <li>Models memorize answers rather than reason</li>
      <li>Especially problematic for MMLU, HellaSwag</li>
      <li>LiveCodeBench addresses this with fresh problems</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Goodhart's Law</div>
    <ul>
      <li>"When a measure becomes a target, it ceases to be a good measure"</li>
      <li>Models are optimized for benchmarks, not real tasks</li>
      <li>High MMLU ≠ good chatbot</li>
      <li>Chatbot Arena (human pref) remains the most trusted signal</li>
    </ul>
  </div>
</div>
</div>

**Other limitations:**
- **Sensitivity to prompting**: Few-shot count, prompt format, and chat template all affect scores significantly
- **Saturation**: Some benchmarks become too easy — HellaSwag is >95% for most modern models
- **Narrow scope**: Benchmarks don't test creativity, nuance, or long-form coherence
- **Static datasets**: Real-world capability evolves, benchmarks don't

## Practical Evaluation Strategy

For your own fine-tuned models:

1. **Perplexity** on a held-out set — did training reduce loss?
2. **Task-specific metrics** — accuracy, F1, exact match on **your** use case
3. **lm-eval-harness** on standard benchmarks — to compare with baselines
4. **Qualitative inspection** — read 50 outputs manually. There is no substitute
5. **Chatbot Arena** (if applicable) — compare against known models

## What's Next

Evaluating models is one thing — serving them at scale is another. The next chapter covers **serving and deployment** with vLLM, TGI, and other inference frameworks.

[← Previous: Chapter 25 — The transformers Library Internals](./25_transformers_library.md) · **Next: [Chapter 27 — Serving & Deployment →](./27_serving_and_deployment.md)**

---

*Last updated: April 2026*
