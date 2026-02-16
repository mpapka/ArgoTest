#!/usr/bin/env python3
"""
Multi-Turn Conversation Demo — Argo Gateway API
==================================================
Demonstrates context retention across multiple conversation turns
through the Argo ``/api/v1/resource/chat/`` endpoint.

The demo builds a 3-turn conversation where each turn depends on
information from the previous turn, proving the model retains context:

  Turn 1: Introduce a fact       ("My project is called Meridian.")
  Turn 2: Build on that fact     ("It studies climate patterns in the Arctic.")
  Turn 3: Recall from context    ("What is my project about?")

The model must reference both the project name and topic in its final
answer — something it can only do if it retained the full conversation
history.

Two model families are demonstrated:
  - OpenAI  (GPT-4o)            — standard parameter handling
  - Anthropic (Claude Sonnet 4.5) — requires max_tokens, single-param constraint

Usage:
    python multiTurnDemo.py -u <your_anl_username>
    python multiTurnDemo.py -u <your_anl_username> --env dev
    python multiTurnDemo.py -u <your_anl_username> --model gpt4o
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

# Conversation turns — each is (user_message, keywords_to_check_in_response)
# Turn 3's keywords validate that the model retained context from turns 1 & 2.
CONVERSATION_TURNS = [
    (
        "I'm working on a research project called Meridian. We're just getting started with the proposal.",
        ["meridian"],
    ),
    (
        "Meridian will focus on studying climate patterns in the Arctic using satellite data. What kinds of challenges might we face?",
        ["arctic"],
    ),
    (
        "Based on our conversation so far, can you summarize what my project is called and what it will study?",
        ["meridian", "arctic"],
    ),
]

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
# Payload builder
# ---------------------------------------------------------------------------

def applyParams(payload, modelConfig):
    """Apply model-specific parameters to a payload."""
    paramStyle = modelConfig["params"]
    if paramStyle == "standard":
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_tokens": 300})
    elif paramStyle == "anthropic_single":
        payload.update({"temperature": 0.1, "max_tokens": 300})


def buildPayload(user, model, modelConfig, messages):
    """Build a chat payload with the full conversation history."""
    payload = {
        "user": user,
        "model": model,
        "messages": messages,
    }
    applyParams(payload, modelConfig)
    return payload


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validateTurnResponse(statusCode, data, duration, keywords, turnNum):
    """Return a list of (checkName, passed, detail) tuples for one turn."""
    checks = []
    prefix = f"Turn {turnNum}"

    ok = statusCode == 200
    checks.append((f"{prefix}: HTTP Status", ok, f"Expected 200, got {statusCode}"))

    isDict = isinstance(data, dict)
    hasRaw = isDict and "raw" in data
    ok = isDict and not hasRaw
    checks.append((f"{prefix}: Valid JSON", ok,
                    "Response is valid JSON" if ok else "Response was not valid JSON"))

    hasError = "error" in data
    detail = f"Error: {str(data.get('error', ''))[:100]}" if hasError else "No error field"
    checks.append((f"{prefix}: No Error", not hasError, detail))

    respText = str(data.get("response", "")).strip()
    ok = len(respText) > 0
    checks.append((f"{prefix}: Non-empty Response", ok,
                    f"{len(respText)} chars" if ok else "Empty response"))

    # Keyword checks — these validate context retention
    respLower = respText.lower()
    for keyword in keywords:
        found = keyword.lower() in respLower
        checks.append((f"{prefix}: Contains '{keyword}'", found,
                        f"Found '{keyword}'" if found else f"'{keyword}' not found in response"))

    ok = duration < REQUEST_TIMEOUT
    checks.append((f"{prefix}: Latency", ok, f"{duration:.2f}s (limit {REQUEST_TIMEOUT}s)"))

    return checks


# ---------------------------------------------------------------------------
# Demo runner — multi-turn conversation
# ---------------------------------------------------------------------------

def runDemo(user, env, model, modelConfig):
    """Run a multi-turn conversation for one model.

    Sends each turn with the full conversation history, appending the
    model's reply after each turn.  Returns True if all checks pass.
    """
    display = modelConfig["display"]
    vendor = modelConfig["vendor"]
    baseURL = BASE_URLS[env]
    url = baseURL + CHAT_PATH
    allChecks = []

    print(f"\n{'=' * 68}")
    print(f"  {BOLD}{display}{RESET}  ({CYAN}{model}{RESET}, vendor={vendor})")
    print(f"{'=' * 68}")

    # Build the conversation history incrementally
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    totalDuration = 0.0

    for turnIdx, (userMessage, keywords) in enumerate(CONVERSATION_TURNS, start=1):
        turnNum = turnIdx
        print(f"\n  {BOLD}Turn {turnNum}:{RESET} {CYAN}{userMessage}{RESET}")

        # Append user message to history
        messages.append({"role": "user", "content": userMessage})

        # Send the full conversation history
        payload = buildPayload(user, model, modelConfig, messages)
        print(f"    {DIM}Messages in payload: {len(messages)}{RESET}")

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
            print(f"    {RED}REQUEST ERROR:{RESET} {exc}")
            allChecks.append((f"Turn {turnNum}: Request", False, str(exc)[:100]))
            break

        totalDuration += duration

        # Extract response text
        respText = str(data.get("response", "")).strip()
        snippet = respText.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        print(f"    {DIM}Response ({duration:.2f}s):{RESET} {snippet}")

        # Validate this turn
        turnChecks = validateTurnResponse(
            resp.status_code, data, duration, keywords, turnNum
        )
        allChecks.extend(turnChecks)

        # Append assistant response to history for the next turn
        messages.append({"role": "assistant", "content": respText})

    # Print all checks
    print(f"\n  {DIM}Total conversation time: {totalDuration:.2f}s{RESET}")
    print(f"\n  {BOLD}All Checks:{RESET}")
    allOK = True
    for name, passed, detail in allChecks:
        icon = okLabel() if passed else failLabel()
        print(f"    {icon}  {name}: {detail}")
        if not passed:
            allOK = False

    return allOK


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Turn Conversation Demo — Argo Gateway API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python multiTurnDemo.py -u mdearing
  python multiTurnDemo.py -u mdearing --env dev
  python multiTurnDemo.py -u mdearing --model gpt4o
  python multiTurnDemo.py -u mdearing --model claudesonnet45
""",
    )
    parser.add_argument("-u", "--user", required=True,
                        help="Your ANL domain username (e.g., mdearing)")
    parser.add_argument("--env", choices=["prod", "test", "dev"], default="dev",
                        help="Target environment (default: dev)")
    parser.add_argument("--model", choices=list(DEMO_MODELS.keys()),
                        help="Run only a specific model example")
    args = parser.parse_args()

    print(f"\n{BOLD}Argo Gateway — Multi-Turn Conversation Demo{RESET}")
    print(f"  User: {CYAN}{args.user}{RESET}")
    print(f"  Env:  {CYAN}{args.env}{RESET}")
    print(f"  URL:  {CYAN}{BASE_URLS[args.env]}{RESET}")

    if args.model:
        modelsToRun = {args.model: DEMO_MODELS[args.model]}
    else:
        modelsToRun = DEMO_MODELS

    results = {}
    for modelID, config in modelsToRun.items():
        results[modelID] = runDemo(args.user, args.env, modelID, config)

    # Summary
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n{'=' * 68}")
    print(f"  {BOLD}Summary{RESET}")
    print(f"{'=' * 68}")
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
