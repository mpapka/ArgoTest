#!/usr/bin/env python3
"""
Tool / Function Calling Demo — Argo Gateway API
=================================================
Complete round-trip demonstration of tool calling through the Argo
``/api/v1/resource/chat/`` endpoint.

Flow:
  1. Send a prompt with tool definitions to the model.
  2. Parse the model's tool-call request from the response.
  3. Execute the real Python implementation of the requested tool.
  4. Send the tool result back to the model.
  5. Receive and display the model's final natural-language answer.

Two model families are demonstrated:
  - OpenAI  (GPT-4o)            — standard parameter handling
  - Anthropic (Claude Sonnet 4.5) — requires max_tokens, single-param constraint

Usage:
    python toolCallingDemo.py -u <your_anl_username>
    python toolCallingDemo.py -u <your_anl_username> --env dev
    python toolCallingDemo.py -u <your_anl_username> --model gpt4o
    python toolCallingDemo.py -u <your_anl_username> --model claudesonnet45
"""

import argparse
import ast
import json
import math
import re
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

# ---------------------------------------------------------------------------
# Tool implementations (the actual functions the model can call)
# ---------------------------------------------------------------------------

SAFE_MATH_NAMES = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "pi": math.pi, "e": math.e, "abs": abs, "pow": pow,
}


def _simpsonsRule(exprStr, a, b, n=1000):
    """Numerically integrate *exprStr* over [a, b] using Simpson's rule."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    def f(x):
        return eval(exprStr, {"__builtins__": {}}, {**SAFE_MATH_NAMES, "x": x})
    total = f(a) + f(b)
    for i in range(1, n):
        coeff = 4 if i % 2 != 0 else 2
        total += coeff * f(a + i * h)
    return total * h / 3


def executeIntegrate(args):
    """Execute the integrate tool and return a result string."""
    exprStr = args.get("expression", "x")
    a = float(args.get("lower_bound", 0))
    b = float(args.get("upper_bound", 1))
    result = _simpsonsRule(exprStr, a, b)
    return json.dumps({
        "expression": exprStr,
        "lower_bound": a,
        "upper_bound": b,
        "result": result,
    })


def executeDerivative(args):
    """Approximate the derivative of *expression* at *point* using central differences."""
    exprStr = args.get("expression", "x")
    x0 = float(args.get("point", 0))
    h = 1e-8
    def f(x):
        return eval(exprStr, {"__builtins__": {}}, {**SAFE_MATH_NAMES, "x": x})
    result = (f(x0 + h) - f(x0 - h)) / (2 * h)
    return json.dumps({
        "expression": exprStr,
        "point": x0,
        "derivative": result,
    })


# Registry: tool name -> implementation
TOOL_REGISTRY = {
    "numericalIntegrate": executeIntegrate,
    "numericalDerivative": executeDerivative,
}

# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema sent to the model)
# ---------------------------------------------------------------------------

INTEGRATE_DESC = (
    "Compute the definite integral of a mathematical expression "
    "over a given interval using numerical methods. The expression "
    "should use 'x' as the variable.  Available functions: sin, "
    "cos, tan, exp, log, sqrt, pow.  Constants: pi, e."
)

INTEGRATE_PARAMS = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Math expression in x, e.g. 'sin(x) * exp(-x)'",
        },
        "lower_bound": {
            "type": "number",
            "description": "Lower limit of integration",
        },
        "upper_bound": {
            "type": "number",
            "description": "Upper limit of integration",
        },
    },
    "required": ["expression", "lower_bound", "upper_bound"],
}

DERIVATIVE_DESC = (
    "Approximate the derivative of a mathematical expression at a "
    "specific point using numerical methods.  The expression should "
    "use 'x' as the variable."
)

DERIVATIVE_PARAMS = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Math expression in x, e.g. 'x**3 - 2*x'",
        },
        "point": {
            "type": "number",
            "description": "The x value at which to evaluate the derivative",
        },
    },
    "required": ["expression", "point"],
}

# OpenAI-style: {"type": "function", "function": {name, description, parameters}}
OPENAI_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "numericalIntegrate",
            "description": INTEGRATE_DESC,
            "parameters": INTEGRATE_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "numericalDerivative",
            "description": DERIVATIVE_DESC,
            "parameters": DERIVATIVE_PARAMS,
        },
    },
]

# Anthropic-style: {name, description, input_schema}
ANTHROPIC_TOOL_DEFINITIONS = [
    {
        "name": "numericalIntegrate",
        "description": INTEGRATE_DESC,
        "input_schema": INTEGRATE_PARAMS,
    },
    {
        "name": "numericalDerivative",
        "description": DERIVATIVE_DESC,
        "input_schema": DERIVATIVE_PARAMS,
    },
]


def toolDefinitionsForVendor(vendor):
    """Return the correct tool definition format for the model vendor."""
    if vendor == "anthropic":
        return ANTHROPIC_TOOL_DEFINITIONS
    return OPENAI_TOOL_DEFINITIONS

# ---------------------------------------------------------------------------
# Prompt — asks something the LLM cannot compute on its own
# ---------------------------------------------------------------------------

USER_PROMPT = (
    "I need two things:\n"
    "1. Compute the definite integral of sin(x)*exp(-x) from 0 to 10.\n"
    "2. Find the derivative of x**3 - 2*x at x = 3.\n"
    "Use the provided tools and report the numerical results."
)

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

def buildPayload(user, model, modelConfig, messages):
    """Build a chat request payload with tool definitions."""
    vendor = modelConfig["vendor"]
    payload = {
        "user": user,
        "model": model,
        "messages": messages,
        "tools": toolDefinitionsForVendor(vendor),
    }
    paramStyle = modelConfig["params"]
    if paramStyle == "standard":
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024})
    elif paramStyle == "anthropic_single":
        payload.update({"temperature": 0.1, "max_tokens": 1024})
    return payload


def buildFollowUpPayload(user, model, modelConfig, messages):
    """Build a follow-up payload (tool result sent back, no tools field needed)."""
    payload = {
        "user": user,
        "model": model,
        "messages": messages,
    }
    paramStyle = modelConfig["params"]
    if paramStyle == "standard":
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024})
    elif paramStyle == "anthropic_single":
        payload.update({"temperature": 0.1, "max_tokens": 1024})
    return payload


# ---------------------------------------------------------------------------
# Response parsing — extract tool calls from model response
# ---------------------------------------------------------------------------

def _extractToolCallsFromList(tcList):
    """Extract (toolName, argsDict) tuples from a tool_calls list.

    Handles both formats:
      - OpenAI:    {function: {name, arguments}, type: "function"}
      - Anthropic: {name, input, type: "tool_use"}
    """
    toolCalls = []
    for tc in tcList:
        # OpenAI format
        if "function" in tc:
            fn = tc["function"]
            name = fn.get("name", "")
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
        # Anthropic format
        elif "name" in tc:
            name = tc.get("name", "")
            args = tc.get("input", {})
        else:
            continue
        if name:
            toolCalls.append((name, args if isinstance(args, dict) else {}))
    return toolCalls


def _tryParseResponseAsDict(responseText):
    """Try to parse the response field as a dict (JSON or Python literal)."""
    # Try JSON first
    try:
        return json.loads(responseText)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try Python literal (gateway sometimes returns single-quoted dicts)
    try:
        result = ast.literal_eval(responseText)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def parseToolCalls(data):
    """Parse tool calls from the Argo /chat/ response.

    The gateway may return tool calls in several formats depending on the
    model backend.  This function handles the common cases and returns a
    list of (toolName, argsDict) tuples.
    """
    toolCalls = []

    # Attempt 1: top-level tool_calls array
    if "tool_calls" in data and isinstance(data["tool_calls"], list):
        return _extractToolCallsFromList(data["tool_calls"])

    # Attempt 2: response field is a dict (or stringified dict) containing tool_calls
    responseRaw = data.get("response", "")
    if isinstance(responseRaw, dict):
        responseParsed = responseRaw
    else:
        responseParsed = _tryParseResponseAsDict(str(responseRaw))

    if isinstance(responseParsed, dict) and "tool_calls" in responseParsed:
        tcList = responseParsed["tool_calls"]
        if isinstance(tcList, list):
            return _extractToolCallsFromList(tcList)

    # Attempt 3: Anthropic-style — response contains tool_use content blocks
    if isinstance(responseParsed, dict) and "content" in responseParsed:
        content = responseParsed.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    args = block.get("input", {})
                    if name:
                        toolCalls.append((name, args if isinstance(args, dict) else {}))
            if toolCalls:
                return toolCalls

    # Attempt 4: JSON block(s) embedded in response text
    responseText = str(responseRaw)
    jsonBlocks = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', responseText)
    for block in jsonBlocks:
        try:
            parsed = json.loads(block)
            name = parsed.get("name", "")
            args = parsed.get("arguments", parsed.get("parameters", parsed.get("input", {})))
            if isinstance(args, str):
                args = json.loads(args)
            if name and name in TOOL_REGISTRY:
                toolCalls.append((name, args if isinstance(args, dict) else {}))
        except json.JSONDecodeError:
            continue
    if toolCalls:
        return toolCalls

    # Attempt 5: look for function names + arguments heuristically
    for toolName in TOOL_REGISTRY:
        if toolName in responseText:
            argsMatch = re.search(
                rf'{toolName}\s*\(([^)]*)\)', responseText
            )
            if argsMatch:
                argStr = argsMatch.group(1)
                argsDict = {}
                for pair in re.findall(r'(\w+)\s*=\s*(["\']?)([^,\'"]+)\2', argStr):
                    key, _, val = pair
                    try:
                        argsDict[key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        argsDict[key] = val
                toolCalls.append((toolName, argsDict))

    return toolCalls


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validateResponse(statusCode, data, duration, stepLabel=""):
    """Return a list of (checkName, passed, detail) tuples."""
    checks = []
    prefix = f"{stepLabel}: " if stepLabel else ""

    ok = statusCode == 200
    checks.append((f"{prefix}HTTP Status", ok, f"Expected 200, got {statusCode}"))

    isDict = isinstance(data, dict)
    hasRaw = isDict and "raw" in data
    ok = isDict and not hasRaw
    checks.append((f"{prefix}Valid JSON", ok,
                    "Response is valid JSON" if ok else "Response was not valid JSON"))

    hasError = "error" in data
    detail = f"Error: {str(data.get('error', ''))[:100]}" if hasError else "No error field"
    checks.append((f"{prefix}No Error", not hasError, detail))

    respText = str(data.get("response", "")).strip()
    ok = len(respText) > 0
    checks.append((f"{prefix}Non-empty Response", ok,
                    f"{len(respText)} chars" if ok else "Empty response"))

    ok = duration < REQUEST_TIMEOUT
    checks.append((f"{prefix}Latency", ok, f"{duration:.2f}s (limit {REQUEST_TIMEOUT}s)"))

    return checks


# ---------------------------------------------------------------------------
# Demo runner — full round-trip
# ---------------------------------------------------------------------------

def runDemo(user, env, model, modelConfig):
    """Run the complete tool-calling round-trip for one model.

    Returns True if all checks pass across all steps.
    """
    display = modelConfig["display"]
    vendor = modelConfig["vendor"]
    baseURL = BASE_URLS[env]
    url = baseURL + CHAT_PATH
    allChecks = []

    print(f"\n{'=' * 68}")
    print(f"  {BOLD}{display}{RESET}  ({CYAN}{model}{RESET}, vendor={vendor})")
    print(f"{'=' * 68}")

    # ── Step 1: Send prompt + tool definitions ────────────────────────
    print(f"\n{BOLD}Step 1:{RESET} Send prompt with tool definitions")
    messages = [
        {"role": "system", "content": "You are a helpful scientific assistant with access to numerical computation tools. Use the provided tools to answer the user's questions."},
        {"role": "user", "content": USER_PROMPT},
    ]
    payload = buildPayload(user, model, modelConfig, messages)
    print(f"  {DIM}Endpoint:{RESET} {url}")
    print(f"  {DIM}Prompt:{RESET} {USER_PROMPT[:80]}...")
    print(f"  {DIM}Tools:{RESET} numericalIntegrate, numericalDerivative")

    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        duration = time.time() - start
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
    except Exception as exc:
        print(f"\n  {RED}REQUEST ERROR:{RESET} {exc}")
        return False

    responseText = str(data.get("response", ""))
    snippet = responseText.replace("\n", " ").strip()
    if len(snippet) > 160:
        snippet = snippet[:160] + "..."
    print(f"  {DIM}Response ({duration:.2f}s):{RESET} {snippet}")

    step1Checks = validateResponse(resp.status_code, data, duration, "Step 1")
    allChecks.extend(step1Checks)

    # ── Step 2: Parse tool calls ──────────────────────────────────────
    print(f"\n{BOLD}Step 2:{RESET} Parse tool calls from response")
    toolCalls = parseToolCalls(data)

    if not toolCalls:
        print(f"  {YELLOW}No structured tool calls found — model may have answered directly.{RESET}")
        print(f"  {DIM}Full response:{RESET}")
        print(f"  {responseText[:400]}")
        allChecks.append(("Step 2: Tool Calls Parsed", False, "No tool calls found in response"))
        # Print final check summary
        printChecks(allChecks)
        return allPassed(allChecks)

    allChecks.append(("Step 2: Tool Calls Parsed", True,
                       f"Found {len(toolCalls)} tool call(s)"))
    for toolName, toolArgs in toolCalls:
        print(f"  {GREEN}Found:{RESET} {toolName}({json.dumps(toolArgs)})")

    # ── Step 3: Execute tools locally ─────────────────────────────────
    print(f"\n{BOLD}Step 3:{RESET} Execute tools locally")
    toolResults = []
    for toolName, toolArgs in toolCalls:
        executeFn = TOOL_REGISTRY.get(toolName)
        if not executeFn:
            result = json.dumps({"error": f"Unknown tool: {toolName}"})
            print(f"  {RED}Unknown tool:{RESET} {toolName}")
        else:
            try:
                result = executeFn(toolArgs)
                print(f"  {GREEN}Executed:{RESET} {toolName} -> {result}")
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
                print(f"  {RED}Error executing {toolName}:{RESET} {exc}")
        toolResults.append((toolName, toolArgs, result))

    allChecks.append(("Step 3: Tools Executed", True,
                       f"Executed {len(toolResults)} tool(s)"))

    # ── Step 4: Send tool results back to the model ───────────────────
    print(f"\n{BOLD}Step 4:{RESET} Send tool results back to the model")
    followUpMessages = list(messages)
    followUpMessages.append({"role": "assistant", "content": responseText})
    for toolName, toolArgs, result in toolResults:
        followUpMessages.append({
            "role": "user",
            "content": (
                f"Tool result for {toolName}({json.dumps(toolArgs)}):\n{result}"
            ),
        })

    followUpPayload = buildFollowUpPayload(user, model, modelConfig, followUpMessages)

    start = time.time()
    try:
        resp2 = requests.post(url, json=followUpPayload, timeout=REQUEST_TIMEOUT)
        duration2 = time.time() - start
        try:
            data2 = resp2.json()
        except Exception:
            data2 = {"raw": resp2.text[:500]}
    except Exception as exc:
        print(f"\n  {RED}REQUEST ERROR:{RESET} {exc}")
        allChecks.append(("Step 4: Follow-up Request", False, str(exc)[:100]))
        printChecks(allChecks)
        return allPassed(allChecks)

    finalResponse = str(data2.get("response", ""))
    finalSnippet = finalResponse.replace("\n", " ").strip()
    if len(finalSnippet) > 200:
        finalSnippet = finalSnippet[:200] + "..."

    step4Checks = validateResponse(resp2.status_code, data2, duration2, "Step 4")
    allChecks.extend(step4Checks)

    # ── Step 5: Display final answer ──────────────────────────────────
    print(f"\n{BOLD}Step 5:{RESET} Final answer from model")
    print(f"  {CYAN}{finalSnippet}{RESET}")

    # Check that the final answer references actual numbers
    hasNumber = bool(re.search(r'\d+\.\d+', finalResponse))
    allChecks.append(("Step 5: Contains Numerical Result", hasNumber,
                       "Final answer includes numerical value(s)" if hasNumber
                       else "No numerical values found in final answer"))

    totalDuration = duration + duration2
    print(f"\n  {DIM}Total round-trip time: {totalDuration:.2f}s{RESET}")

    # Print all checks
    printChecks(allChecks)
    return allPassed(allChecks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allPassed(checks):
    return all(passed for _, passed, _ in checks)


def printChecks(checks):
    print(f"\n{BOLD}Checks:{RESET}")
    for name, passed, detail in checks:
        icon = okLabel() if passed else failLabel()
        print(f"  {icon}  {name}: {detail}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tool / Function Calling Demo — Argo Gateway API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python toolCallingDemo.py -u mdearing
  python toolCallingDemo.py -u mdearing --env dev
  python toolCallingDemo.py -u mdearing --model gpt4o
  python toolCallingDemo.py -u mdearing --model claudesonnet45
""",
    )
    parser.add_argument("-u", "--user", required=True,
                        help="Your ANL domain username (e.g., mdearing)")
    parser.add_argument("--env", choices=["prod", "test", "dev"], default="dev",
                        help="Target environment (default: dev)")
    parser.add_argument("--model", choices=list(DEMO_MODELS.keys()),
                        help="Run only a specific model example")
    args = parser.parse_args()

    print(f"\n{BOLD}Argo Gateway — Tool Calling Demo{RESET}")
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
        print(f"  {GREEN}{BOLD}All {total} example(s) passed.{RESET}")
    else:
        print(f"  {RED}{BOLD}{failed}/{total} example(s) failed.{RESET}")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
