# 01 — Tool / Function Calling

## What is Tool Calling?

Tool calling (also known as function calling) lets a language model declare
that it wants to invoke an external function rather than producing a plain-text
answer. The model receives a set of **tool definitions** — JSON Schema
descriptions of available functions — and, when appropriate, responds with a
structured request to call one of them.

This enables powerful patterns:

- Retrieving live data (weather, database lookups, API calls)
- Performing computations the model cannot do natively
- Triggering side-effects (sending email, creating records)

## Why Numerical Tools?

An LLM can do symbolic math, but it **cannot** execute numerical algorithms
like Simpson's rule integration or central-difference derivatives to arbitrary
precision. That makes these ideal tool-calling candidates — the model
genuinely needs the tool to answer the question.

This demo provides two tools:

| Tool | What it computes |
|------|-----------------|
| `numericalIntegrate` | Definite integral via Simpson's rule |
| `numericalDerivative` | Derivative at a point via central differences |

## Complete Round-Trip Flow

The demo runs a **full tool-calling loop**, not just a one-shot check:

```
1. Send prompt + tool definitions to Argo
       ↓
2. Model responds with tool-call request(s)
       ↓
3. Demo executes the tools locally (real Python math)
       ↓
4. Send tool results back to the model
       ↓
5. Model produces final natural-language answer with numbers
```

## How the Argo Gateway Supports Tool Calling

The Argo Gateway exposes a unified chat endpoint:

```
POST /api/v1/resource/chat/
```

Pass tool definitions in the `tools` field of the JSON body. The gateway
forwards them to the underlying model (OpenAI, Anthropic, Google) and returns
the model's response — which may include a structured tool-call request.

## Tool Definition Format

Each tool is a JSON object following the OpenAI-style schema:

```json
{
  "type": "function",
  "function": {
    "name": "numericalIntegrate",
    "description": "Compute the definite integral of a math expression ...",
    "parameters": {
      "type": "object",
      "properties": {
        "expression":  { "type": "string",  "description": "e.g. 'sin(x) * exp(-x)'" },
        "lower_bound": { "type": "number",  "description": "Lower limit" },
        "upper_bound": { "type": "number",  "description": "Upper limit" }
      },
      "required": ["expression", "lower_bound", "upper_bound"]
    }
  }
}
```

The gateway normalises this format across providers so you can use the same
tool definitions regardless of the backend model.

## Model-Family Differences

| Aspect | OpenAI (GPT) | Anthropic (Claude) |
|--------|-------------|-------------------|
| `max_tokens` | Optional | **Required** |
| `temperature` + `top_p` | Both allowed | Some models accept only one (single-param constraint) |
| Parameter style | `standard` | `anthropic_single` or `anthropic_standard` |

The demo script handles these differences automatically.

## Quick Start

```bash
# Run both examples (GPT-4o + Claude Sonnet 4.5)
python toolCallingDemo.py -u <your_anl_username>

# Specific environment
python toolCallingDemo.py -u <your_anl_username> --env dev

# Single model only
python toolCallingDemo.py -u <your_anl_username> --model gpt4o
python toolCallingDemo.py -u <your_anl_username> --model claudesonnet45
```

### Requirements

- Python 3.10+
- `requests` library (already in the project's `requirements.txt`)
- An ANL domain username with Argo Gateway access

## Validation Checks

Each step of the round-trip is validated:

| Check | Pass Condition |
|-------|---------------|
| HTTP Status | `200` |
| Valid JSON | Response parses as a dict (no `raw` fallback) |
| No Error | No `error` field in the response |
| Non-empty Response | Response text is non-empty |
| Latency | < 120 seconds |
| Tool Calls Parsed | At least one tool call extracted from Step 1 |
| Tools Executed | All requested tools ran successfully |
| Contains Numerical Result | Final answer includes computed values |

## See Also

- [Main test suite](../../argoTestSuite.py) — comprehensive tests across all
  models, endpoints, and features
- [Project README](../../README.md)
