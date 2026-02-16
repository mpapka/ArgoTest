# 03 — Multi-Turn Conversations

## Overview

LLMs are stateless — each API call is independent. To create a conversation
with context, the caller must send the **full message history** with every
request. This demo proves that the Argo Gateway correctly passes conversation
history to the model, enabling multi-turn context retention.

## How It Works

The demo builds a 3-turn conversation where each turn depends on the
previous:

```
Turn 1: "I'm working on a research project called Meridian."
           → Model acknowledges the project

Turn 2: "Meridian will study climate patterns in the Arctic
          using satellite data. What challenges might we face?"
           → Model discusses challenges, references the project

Turn 3: "Based on our conversation, can you summarize what my
          project is called and what it will study?"
           → Model must recall "Meridian" and "Arctic" from context
```

After each turn, the model's response is appended to the message history
and sent along with the next turn. The growing messages array is the only
mechanism for context — the gateway itself is stateless.

## Message History Growth

```
Turn 1 payload:  [system, user₁]                              → 2 messages
Turn 2 payload:  [system, user₁, assistant₁, user₂]           → 4 messages
Turn 3 payload:  [system, user₁, assistant₁, user₂, assistant₂, user₃]  → 6 messages
```

Each payload includes the full conversation so far. The alternating
`user`/`assistant` pattern is required by all model providers.

## Context Retention Validation

The final turn's response is checked for keywords from earlier turns:

| Keyword | Introduced In | Checked In |
|---------|--------------|------------|
| `meridian` | Turn 1 | Turns 1, 2, 3 |
| `arctic` | Turn 2 | Turns 2, 3 |

If the model answers Turn 3 correctly (mentions both "Meridian" and
"Arctic"), it proves the full conversation history was delivered and
processed.

## Quick Start

```bash
# Run both models (GPT-4o + Claude Sonnet 4.5)
python multiTurnDemo.py -u <your_anl_username>

# Specific environment
python multiTurnDemo.py -u <your_anl_username> --env dev

# Single model
python multiTurnDemo.py -u <your_anl_username> --model gpt4o
python multiTurnDemo.py -u <your_anl_username> --model claudesonnet45
```

### Requirements

- Python 3.10+
- `requests` library (already in the project's `requirements.txt`)
- An ANL domain username with Argo Gateway access

## Validation Checks

Per turn:

| Check | Pass Condition |
|-------|---------------|
| HTTP Status | `200` |
| Valid JSON | Response parses as a dict |
| No Error | No `error` field |
| Non-empty Response | Response text is non-empty |
| Contains Keywords | Expected keywords found in response |
| Latency | < 120 seconds |

## See Also

- [02 — Basic Chat](../02-chat-basic/) — single-turn messages and prompt formats
- [Main test suite](../../argoTestSuite.py) — comprehensive tests across all
  models, endpoints, and features
- [Project README](../../README.md)
