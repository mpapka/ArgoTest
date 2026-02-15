# Using Claude Code with the Argo Gateway

Claude Code can be backed by the Argo Gateway API instead of the default Anthropic API. The Argo gateway's `/v1/messages` endpoint is fully compatible with the Anthropic Messages API format that Claude Code expects.

## Prerequisites

- An ANL domain username with access to the Argo Gateway
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed (`npm install -g @anthropic-ai/claude-code`)

## Configuration

Set two environment variables:

```bash
export ANTHROPIC_BASE_URL="https://apps-dev.inside.anl.gov/argoapi"
export ANTHROPIC_API_KEY="<your_anl_username>"
```

### Option A: Shell profile (per-user, all sessions)

Add the exports to `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export ANTHROPIC_BASE_URL="https://apps-dev.inside.anl.gov/argoapi"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="<your_anl_username>"' >> ~/.zshrc
source ~/.zshrc
```

### Option B: Claude Code settings (per-user, Claude Code only)

Create or edit `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://apps-dev.inside.anl.gov/argoapi",
    "ANTHROPIC_API_KEY": "<your_anl_username>"
  }
}
```

### Option C: One-off invocation

```bash
ANTHROPIC_BASE_URL="https://apps-dev.inside.anl.gov/argoapi" \
ANTHROPIC_API_KEY="<your_anl_username>" \
claude
```

## How it works

Claude Code communicates with its backend via the Anthropic Messages API:

1. Claude Code sends `POST` requests to `$ANTHROPIC_BASE_URL/v1/messages`
2. The `ANTHROPIC_API_KEY` value is sent as the `x-api-key` header — the same header the Argo gateway uses for authentication with ANL usernames
3. Claude Code sends standard Anthropic model IDs (e.g., `claude-opus-4-6`), which the Argo gateway accepts directly
4. The gateway returns responses in the standard Anthropic Messages API format, so all Claude Code features work normally

## Available environments

| Environment | Base URL                                      |
|-------------|-----------------------------------------------|
| dev         | `https://apps-dev.inside.anl.gov/argoapi`     |
| test        | `https://apps-test.inside.anl.gov/argoapi`    |
| prod        | `https://apps.inside.anl.gov/argoapi`         |

The `/v1/messages` endpoint is available in the **dev** environment.

## Verification

Run a quick test to confirm the connection:

```bash
claude --print "Say hello"
```

Or verify with curl:

```bash
curl -s -X POST "https://apps-dev.inside.anl.gov/argoapi/v1/messages" \
  -H "x-api-key: <your_anl_username>" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-opus-4-6",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

A successful response returns HTTP 200 with a JSON body containing `"type": "message"`.
