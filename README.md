# Argo Gateway API — Test Suite & Tools

Comprehensive test suite for the [Argo Gateway API](https://apps.inside.anl.gov/argoapi) at Argonne National Laboratory. Tests all models, endpoints, and features with deep response validation and rich terminal output.

## Supported models

**OpenAI** — GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o, o1, o3, o4-mini, GPT-4.1, GPT-5, GPT-5.1, GPT-5.2

**Google** — Gemini 2.5 Pro, Gemini 2.5 Flash

**Anthropic** — Claude Opus 4/4.1/4.5/4.6, Claude Sonnet 3.5v2/3.7/4/4.5, Claude Haiku 3.5/4.5

**Embeddings** — text-embedding-ada-002, text-embedding-3-large, text-embedding-3-small

## Requirements

- Python 3.10+
- ANL domain username with Argo access
- Dependencies: `httpx`, `requests`, `rich`

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Run full test suite against dev environment
python argoTestSuite.py -u <your_anl_username>

# Test a specific environment
python argoTestSuite.py -u <your_anl_username> --env prod

# Test only Anthropic models
python argoTestSuite.py -u <your_anl_username> --category anthropic

# Test a single model
python argoTestSuite.py -u <your_anl_username> --model gpt4o

# List all available models
python argoTestSuite.py -u <your_anl_username> --list-models
```

## CLI options

| Flag | Description |
|------|-------------|
| `-u, --user` | ANL domain username (required) |
| `--env` | Target environment: `prod`, `test`, `dev` (default: `dev`) |
| `--category` | Filter by vendor: `openai`, `google`, `anthropic` |
| `--model` | Test a specific model ID only |
| `--list-models` | List all models and exit |
| `--skip-stream` | Skip streaming tests |
| `--skip-compat` | Skip OpenAI-compatible and Anthropic Messages endpoint tests |
| `--skip-tools` | Skip tool/function calling tests |
| `--skip-embed` | Skip embedding tests |
| `-v, --verbose` | Show request/response details |

## What gets tested

- **Chat** — Messages format and prompt/system format
- **Multi-turn** — Context retention across conversation turns
- **Streaming** — Server-sent event delivery
- **OpenAI-compatible endpoint** — `/v1/chat/completions`
- **Anthropic Messages endpoint** — `/v1/messages`
- **Tool/function calling** — Function call and response handling
- **Embeddings** — Single and batch embedding generation
- **Error handling** — Invalid model, missing fields, empty prompts
- **Parameter constraints** — Temperature, max tokens, stop sequences

## Environments

| Environment | Base URL |
|-------------|----------|
| dev | `https://apps-dev.inside.anl.gov/argoapi` |
| test | `https://apps-test.inside.anl.gov/argoapi` |
| prod | `https://apps.inside.anl.gov/argoapi` |

## Claude Code integration

The Argo gateway can serve as a backend for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). See [CLAUDE_CODE_SETUP.md](CLAUDE_CODE_SETUP.md) for configuration instructions.
