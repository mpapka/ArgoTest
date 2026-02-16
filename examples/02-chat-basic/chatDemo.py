#!/usr/bin/env python3
"""
Basic Chat Demo — Argo Gateway API
====================================
Demonstrates the two input formats supported by the Argo
``/api/v1/resource/chat/`` endpoint:

  1. **Messages format** — OpenAI-style array of role/content objects
  2. **Prompt format**   — simple ``system`` + ``prompt`` fields

Each format is tested against two model families:
  - OpenAI  (GPT-4o)            — standard parameter handling
  - Anthropic (Claude Sonnet 4.5) — requires max_tokens, single-param constraint

The demo sends a question the model can answer deterministically
("What is Argonne National Laboratory?") and validates that the
response contains expected keywords.

Usage:
    python chatDemo.py -u <your_anl_username>
    python chatDemo.py -u <your_anl_username> --env dev
    python chatDemo.py -u <your_anl_username> --model gpt4o
    python chatDemo.py -u <your_anl_username> --format messages
"""

import argparse
import json
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

BASE_URLS = {
    "prod": "https://apps.inside.anl.gov/argoapi",
    "test": "https://apps-test.inside.anl.gov/argoapi",
    "dev":  "https://apps-dev.inside.anl.gov/argoapi",
}

CHAT_PATH = "/api/v1/resource/chat/"
REQUEST_TIMEOUT = 120  # seconds

DEMO_MODELS = {
    "gpt4o": {
        "vendor":  "openai",
        "display": "GPT-4o",
        "params":  "standard",
    },
    "claudesonnet45": {
        "vendor":  "anthropic",
        "display": "Claude Sonnet 4.5",
        "params":  "anthropic_single",
    },
}

SYSTEM_MESSAGE = "You are Argo, a helpful scientific assistant at Argonne National Laboratory."
USER_QUESTION = "What is Argonne National Laboratory? Reply in two sentences."

# Keywords we expect in a correct response
EXPECTED_KEYWORDS = ["argonne", "laboratory"]

# ---------------------------------------------------------------------------
# Terminal colours (ANSI)
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def okLabel():
    return f"{GREEN}PASS{RESET}"


def failLabel():
    return f"{RED}FAIL{RESET}"


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def applyParams(payload, modelConfig):
    """Apply model-specific parameters to a payload."""
    paramStyle = modelConfig["params"]
    if paramStyle == "standard":
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_tokens": 300})
    elif paramStyle == "anthropic_single":
        payload.update({"temperature": 0.1, "max_tokens": 300})


def buildMessagesPayload(user, model, modelConfig):
    """Build a chat payload using the messages array format.

    Messages format uses an array of {role, content} objects — the same
    structure as the OpenAI Chat Completions API.
    """
    payload = {
        "user": user,
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_QUESTION},
        ],
    }
    applyParams(payload, modelConfig)
    return payload


def buildPromptPayload(user, model, modelConfig):
    """Build a chat payload using the prompt format.

    Prompt format uses separate ``system`` and ``prompt`` fields.
    The ``prompt`` field is a list of strings.
    """
    payload = {
        "user": user,
        "model": model,
        "system": SYSTEM_MESSAGE,
        "prompt": [USER_QUESTION],
    }
    applyParams(payload, modelConfig)
    return payload


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validateResponse(statusCode, data, duration, keywords):
    """Return a list of (checkName, passed, detail) tuples."""
    checks = []

    # HTTP 200
    ok = statusCode == 200
    checks.append(("HTTP Status", ok, f"Expected 200, got {statusCode}"))

    # Valid JSON
    isDict = isinstance(data, dict)
    hasRaw = isDict and "raw" in data
    ok = isDict and not hasRaw
    checks.append(("Valid JSON", ok,
                    "Response is valid JSON" if ok else "Response was not valid JSON"))

    # No error field
    hasError = "error" in data
    detail = f"Error: {str(data.get('error', ''))[:100]}" if hasError else "No error field"
    checks.append(("No Error", not hasError, detail))

    # Has response field
    hasResponse = "response" in data
    checks.append(("Has Response Field", hasResponse,
                    "Field 'response' present" if hasResponse else "Missing 'response' field"))

    # Non-empty response
    respText = str(data.get("response", "")).strip()
    ok = len(respText) > 0
    checks.append(("Non-empty Response", ok,
                    f"{len(respText)} chars" if ok else "Empty response"))

    # Response is a string
    respVal = data.get("response")
    isStr = isinstance(respVal, str)
    checks.append(("Response Is String", isStr,
                    f"Type: {type(respVal).__name__}" if not isStr else "Response is a string"))

    # Contains expected keywords
    respLower = respText.lower()
    for keyword in keywords:
        found = keyword.lower() in respLower
        checks.append((f"Contains '{keyword}'", found,
                        f"Found '{keyword}'" if found else f"'{keyword}' not found in response"))

    # Latency
    ok = duration < REQUEST_TIMEOUT
    checks.append(("Latency", ok, f"{duration:.2f}s (limit {REQUEST_TIMEOUT}s)"))

    return checks


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def runExample(user, env, model, modelConfig, formatName, payloadBuilder):
    """Run a single chat example. Returns True if all checks pass."""
    display = modelConfig["display"]
    vendor = modelConfig["vendor"]
    baseURL = BASE_URLS[env]
    url = baseURL + CHAT_PATH

    print(f"\n  {BOLD}{display}{RESET} + {CYAN}{formatName}{RESET} format"
          f"  ({DIM}{model}, vendor={vendor}{RESET})")
    print(f"  {'-' * 58}")

    payload = payloadBuilder(user, model, modelConfig)

    # Show the payload
    print(f"  {DIM}Endpoint:{RESET} {url}")
    print(f"  {DIM}Payload:{RESET}")
    for line in json.dumps(payload, indent=2).split("\n"):
        print(f"    {DIM}{line}{RESET}")

    # Send request
    print(f"\n  {DIM}Sending request ...{RESET}")
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        duration = time.time() - start
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
    except Exception as exc:
        duration = time.time() - start
        print(f"  {RED}REQUEST ERROR:{RESET} {exc}")
        return False

    # Show response snippet
    respText = str(data.get("response", ""))
    snippet = respText.replace("\n", " ").strip()
    if len(snippet) > 200:
        snippet = snippet[:200] + "..."
    print(f"  {DIM}Response ({duration:.2f}s):{RESET}")
    print(f"    {snippet}")

    # Validate
    checks = validateResponse(resp.status_code, data, duration, EXPECTED_KEYWORDS)

    print(f"\n  {BOLD}Checks:{RESET}")
    allOK = True
    for name, passed, detail in checks:
        icon = okLabel() if passed else failLabel()
        print(f"    {icon}  {name}: {detail}")
        if not passed:
            allOK = False

    return allOK


def runDemo(user, env, model, modelConfig, formats):
    """Run all format examples for one model. Returns True if all pass."""
    builders = {
        "messages": ("messages", buildMessagesPayload),
        "prompt":   ("prompt",   buildPromptPayload),
    }
    allOK = True
    for fmt in formats:
        formatName, builder = builders[fmt]
        if not runExample(user, env, model, modelConfig, formatName, builder):
            allOK = False
    return allOK


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Basic Chat Demo — Argo Gateway API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python chatDemo.py -u mdearing
  python chatDemo.py -u mdearing --env dev
  python chatDemo.py -u mdearing --model gpt4o
  python chatDemo.py -u mdearing --format messages
  python chatDemo.py -u mdearing --model claudesonnet45 --format prompt
""",
    )
    parser.add_argument("-u", "--user", required=True,
                        help="Your ANL domain username (e.g., mdearing)")
    parser.add_argument("--env", choices=["prod", "test", "dev"], default="dev",
                        help="Target environment (default: dev)")
    parser.add_argument("--model", choices=list(DEMO_MODELS.keys()),
                        help="Run only a specific model example")
    parser.add_argument("--format", choices=["messages", "prompt"], dest="fmt",
                        help="Run only a specific input format")
    args = parser.parse_args()

    print(f"\n{BOLD}Argo Gateway — Basic Chat Demo{RESET}")
    print(f"  User: {CYAN}{args.user}{RESET}")
    print(f"  Env:  {CYAN}{args.env}{RESET}")
    print(f"  URL:  {CYAN}{BASE_URLS[args.env]}{RESET}")

    # Determine which models and formats to run
    if args.model:
        modelsToRun = {args.model: DEMO_MODELS[args.model]}
    else:
        modelsToRun = DEMO_MODELS

    formats = [args.fmt] if args.fmt else ["messages", "prompt"]

    results = {}
    for modelID, config in modelsToRun.items():
        print(f"\n{'=' * 64}")
        print(f"  {BOLD}{config['display']}{RESET}  ({CYAN}{modelID}{RESET})")
        print(f"{'=' * 64}")
        results[modelID] = runDemo(args.user, args.env, modelID, config, formats)

    # Summary
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n{'=' * 64}")
    print(f"  {BOLD}Summary{RESET}")
    print(f"{'=' * 64}")
    for modelID, ok in results.items():
        icon = okLabel() if ok else failLabel()
        print(f"  {icon}  {DEMO_MODELS[modelID]['display']} ({modelID})")
    print()
    if failed == 0:
        print(f"  {GREEN}{BOLD}All {total} model(s) passed.{RESET}")
    else:
        print(f"  {RED}{BOLD}{failed}/{total} model(s) failed.{RESET}")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
