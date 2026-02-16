# 02 — Basic Chat

## Overview

The Argo Gateway's `/api/v1/resource/chat/` endpoint accepts two distinct
input formats for sending prompts to a model:

1. **Messages format** — an array of `{role, content}` objects (OpenAI-style)
2. **Prompt format** — separate `system` and `prompt` fields

This demo exercises both formats against two model families (GPT-4o and
Claude Sonnet 4.5) and validates the full response structure.

## Messages Format

The messages array mirrors the OpenAI Chat Completions API:

```json
{
  "user":     "<anl_username>",
  "model":    "gpt4o",
  "messages": [
    {"role": "system", "content": "You are Argo, a helpful assistant."},
    {"role": "user",   "content": "What is Argonne National Laboratory?"}
  ],
  "temperature": 0.1,
  "max_tokens":  300
}
```

Roles include `system` (optional instructions), `user` (the question), and
`assistant` (for multi-turn context — see example 03).

## Prompt Format

The prompt format is a simpler alternative:

```json
{
  "user":   "<anl_username>",
  "model":  "gpt4o",
  "system": "You are Argo, a helpful assistant.",
  "prompt": ["What is Argonne National Laboratory?"],
  "temperature": 0.1,
  "max_tokens":  300
}
```

The `prompt` field is a list of strings. The `system` field is a single
string (optional).

## Model-Family Differences

| Aspect | OpenAI (GPT) | Anthropic (Claude) |
|--------|-------------|-------------------|
| `max_tokens` | Optional | **Required** |
| `temperature` + `top_p` | Both allowed | Some models accept only one |
| Parameter style | `standard` | `anthropic_single` |

Both formats work identically across providers — the gateway handles
translation.

## Quick Start

```bash
# Run all combinations (2 models x 2 formats = 4 examples)
python chatDemo.py -u <your_anl_username>

# Specific environment
python chatDemo.py -u <your_anl_username> --env dev

# Single model
python chatDemo.py -u <your_anl_username> --model gpt4o

# Single format
python chatDemo.py -u <your_anl_username> --format messages

# Combine filters
python chatDemo.py -u <your_anl_username> --model claudesonnet45 --format prompt
```

### Requirements

- Python 3.10+
- `requests` library (already in the project's `requirements.txt`)
- An ANL domain username with Argo Gateway access

## Validation Checks

Each example validates:

| Check | Pass Condition |
|-------|---------------|
| HTTP Status | `200` |
| Valid JSON | Response parses as a dict |
| No Error | No `error` field in the response |
| Has Response Field | `response` key exists |
| Non-empty Response | Response text is non-empty |
| Response Is String | `response` value is a Python `str` |
| Contains Keywords | Response mentions "argonne" and "laboratory" |
| Latency | < 120 seconds |

## See Also

- [01 — Tool Calling](../01-tool-calling/) — tool/function calling with
  numerical computation
- [Main test suite](../../argoTestSuite.py) — comprehensive tests across all
  models, endpoints, and features
- [Project README](../../README.md)
