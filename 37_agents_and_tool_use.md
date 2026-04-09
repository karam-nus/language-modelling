---
title: "Chapter 37 — Agents & Tool Use"
---

[← Back to Table of Contents](./README.md)

# Chapter 37 — Agents & Tool Use

> *"A language model that can only generate text is a brain in a jar. An agent is a brain with hands."*

> For a comprehensive deep dive into AI agents, multi-agent systems, and frameworks, see the companion guide: **[Agents Guide](https://karam-nus.github.io/agents/)**.

## The Agent Loop

An agent is an LLM that can **observe, reason, act, and observe again** in a loop:

<div class="diagram">
<div class="diagram-title">Agent Execution Loop</div>
<div class="cycle">
  <div class="cycle-step accent">🤔 Reason — analyze the task and decide what to do next</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step green">🛠️ Act — call a tool (search, code execution, API call)</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step purple">👀 Observe — process the tool's output / result</div>
  <div class="cycle-arrow">→</div>
  <div class="cycle-step orange">🔄 Repeat — until the task is complete or max steps reached</div>
</div>
</div>

## Function Calling

The foundation of tool use — LLMs generate structured function calls instead of (or alongside) natural text:

```python
# OpenAI function calling
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
)

# Model responds with:
# tool_calls: [{"function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}]
```

The model doesn't execute the function — it generates a structured request. Your code executes it and feeds the result back:

<div class="diagram">
<div class="diagram-title">Function Calling Flow</div>
<div class="flow-h">
  <div class="flow-step accent">User: "Weather in Tokyo?"</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step green">LLM: call get_weather("Tokyo")</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step purple">System: executes function → "22°C, sunny"</div>
  <div class="flow-arrow">→</div>
  <div class="flow-step orange">LLM: "It's 22°C and sunny in Tokyo."</div>
</div>
</div>

## Model Context Protocol (MCP)

**MCP** (Anthropic, 2024) standardizes how LLMs connect to external tools and data sources:

<div class="diagram">
<div class="diagram-title">MCP Architecture</div>
<div class="layer-stack">
  <div class="layer accent">LLM Application (Claude, VS Code, custom app) — MCP Client</div>
  <div class="layer green">MCP Protocol — standardized JSON-RPC over stdio/SSE</div>
  <div class="layer purple">MCP Server: GitHub — repos, issues, PRs</div>
  <div class="layer purple">MCP Server: Database — query, schema inspection</div>
  <div class="layer purple">MCP Server: File System — read, write, search</div>
  <div class="layer purple">MCP Server: Web Search — Brave, Google, etc.</div>
</div>
</div>

MCP provides a universal interface so that:
- Any LLM application can connect to any MCP server
- Tools are described once and discovered automatically
- Context (resources) can be exposed alongside tools

## Coding Agents

Agents that write, debug, and modify code:

| Agent | Approach | Key Feature |
|-------|---------|-------------|
| **GitHub Copilot** | IDE-integrated, multi-model | Code completion, chat, workspace agents, MCP |
| **Cursor** | IDE-integrated | Codebase-aware, multi-file edits |
| **Codex (OpenAI)** | Cloud agent | Sandboxed environment, async task execution |
| **SWE-Agent** | Research agent | Interacts with repos via terminal commands |
| **Devin** | Full IDE agent | Plans, implements, tests autonomously |
| **Aider** | Terminal-based | Git-integrated, multi-file editing |

## Computer Use Agents

Agents that interact with graphical interfaces:

<div class="diagram">
<div class="diagram-title">Computer Use Agent Loop</div>
<div class="flow">
  <div class="flow-step accent">Take screenshot of the screen</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step green">VLM analyzes the screenshot — identifies UI elements</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step purple">Decide action: click(x, y), type("text"), scroll, key("Enter")</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step orange">Execute action on the computer</div>
  <div class="flow-arrow">↓</div>
  <div class="flow-step accent">Take new screenshot → repeat</div>
</div>
</div>

Examples: Claude Computer Use, OpenAI Operator, browser-use agents.

## ReAct Pattern

**ReAct** (Reasoning + Acting) interleaves thinking and tool use:

```
Question: What is the population of the capital of France?

Thought: I need to find the capital of France, then its population.
Action: search("capital of France")
Observation: The capital of France is Paris.

Thought: Now I need the population of Paris.
Action: search("population of Paris 2024")
Observation: The population of Paris is approximately 2.1 million (city proper).

Thought: I have the answer.
Answer: The population of Paris, the capital of France, is approximately 2.1 million.
```

## Multi-Agent Systems

Complex tasks can be decomposed across specialized agents:

<div class="diagram">
<div class="diagram-title">Multi-Agent Architecture</div>
<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-title">Orchestrator</div>
    <div class="card-desc">Plans the task, delegates to specialists, synthesizes results.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-title">Researcher</div>
    <div class="card-desc">Searches the web, reads documents, gathers information.</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-title">Coder</div>
    <div class="card-desc">Writes code, runs tests, fixes bugs.</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-title">Reviewer</div>
    <div class="card-desc">Checks outputs for correctness, suggests improvements.</div>
  </div>
</div>
</div>

Frameworks: **LangGraph**, **CrewAI**, **AutoGen**, **Semantic Kernel**, **OpenAI Swarm**.

## Agent Challenges

| Challenge | Description |
|-----------|-----------|
| **Reliability** | Agents make mistakes that compound across steps. Error recovery is hard. |
| **Cost** | Long agent traces consume many tokens. A 50-step agent run costs 50× a single query. |
| **Safety** | Agents can take irreversible actions (delete files, send emails). Sandboxing is critical. |
| **Evaluation** | Hard to benchmark — success depends on environment, not just model quality. |
| **Context window** | Long traces may exceed context limits. Summarization/compression needed. |

## What's Next

We've now covered the full stack: from tokenization to training, inference, hardware, and applications. The final chapter looks at the **frontier** — where the field is heading and what remains unsolved.

[← Previous: Chapter 36 — Retrieval-Augmented Generation](./36_rag.md) · **Next: [Chapter 38 — The Frontier →](./38_frontier.md)**

---

*Last updated: April 2026*
