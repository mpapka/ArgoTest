#!/usr/bin/env python3
"""
Argo Gateway API — Comprehensive Test Suite with Validation
=============================================================
Tests all models, endpoints, and features of the Argo Gateway API
with deep response validation and extensive rich-formatted feedback.

Usage:
    python argo_test_suite.py --user <your_anl_domain_user>
    python argo_test_suite.py --user <your_anl_domain_user> --env dev
    python argo_test_suite.py --user <your_anl_domain_user> --env prod --category openai
    python argo_test_suite.py --user <your_anl_domain_user> --list-models
    python argo_test_suite.py --user <your_anl_domain_user> --model gpt4o
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
import requests
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

console = Console()

BASE_URLS = {
    "prod": "https://apps.inside.anl.gov/argoapi",
    "test": "https://apps-test.inside.anl.gov/argoapi",
    "dev": "https://apps-dev.inside.anl.gov/argoapi",
}

CHAT_PATH = "/api/v1/resource/chat/"
STREAM_PATH = "/api/v1/resource/streamchat/"
EMBED_PATH = "/api/v1/resource/embed/"
OPENAI_COMPAT_PATH = "/v1/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"

REQUEST_TIMEOUT = 120  # seconds


class ModelVendor(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


class ParamSupport(str, Enum):
    STANDARD = "standard"              # temperature, top_p, max_tokens
    COMPLETION_TOKENS = "completion"   # max_completion_tokens only, no temp/top_p
    MIXED = "mixed"                    # temperature, top_p, max_completion_tokens
    ANTHROPIC_STANDARD = "anthropic"   # requires max_tokens, temp, top_p
    ANTHROPIC_SINGLE = "anthropic_s"   # only one of temperature or top_p
    NO_MAX_TOKENS = "no_max_tokens"    # temp, top_p, max_completion_tokens but NOT max_tokens


@dataclass
class ModelSpec:
    modelId: str
    displayName: str
    vendor: ModelVendor
    paramSupport: ParamSupport
    maxInputTokens: int
    maxOutputTokens: int
    devOnly: bool = False
    supportsSystem: bool = True
    supportsTools: bool = True
    supportsStreaming: bool = True
    multimodal: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Model registry — every model from the documentation
# ---------------------------------------------------------------------------

ALL_MODELS: list[ModelSpec] = [
    # ── OpenAI ────────────────────────────────────────────────────────────
    ModelSpec("gpt35", "GPT-3.5 Turbo", ModelVendor.OPENAI, ParamSupport.STANDARD, 4096, 4096),
    ModelSpec("gpt35large", "GPT-3.5 Turbo 16k", ModelVendor.OPENAI, ParamSupport.STANDARD, 16384, 4096),
    ModelSpec("gpt4", "GPT-4", ModelVendor.OPENAI, ParamSupport.STANDARD, 8192, 4096),
    ModelSpec("gpt4large", "GPT-4 32k", ModelVendor.OPENAI, ParamSupport.STANDARD, 32768, 4096),
    ModelSpec("gpt4turbo", "GPT-4 Turbo", ModelVendor.OPENAI, ParamSupport.STANDARD, 128000, 4096),
    ModelSpec("gpt4o", "GPT-4o", ModelVendor.OPENAI, ParamSupport.STANDARD, 128000, 16384),
    ModelSpec("gpt4olatest", "GPT-4o Latest", ModelVendor.OPENAI, ParamSupport.STANDARD, 128000, 16384, devOnly=True),
    ModelSpec("gpto1preview", "GPT o1-preview", ModelVendor.OPENAI, ParamSupport.COMPLETION_TOKENS, 128000, 32768,
             supportsSystem=False, supportsTools=False, notes="Only user prompt + max_completion_tokens"),
    ModelSpec("gpto1mini", "GPT o1-mini", ModelVendor.OPENAI, ParamSupport.COMPLETION_TOKENS, 128000, 65536,
             devOnly=True, supportsSystem=False, supportsTools=False),
    ModelSpec("gpto1", "GPT o1", ModelVendor.OPENAI, ParamSupport.COMPLETION_TOKENS, 200000, 100000,
             devOnly=True, supportsSystem=False, supportsTools=False),
    ModelSpec("gpto3mini", "GPT o3-mini", ModelVendor.OPENAI, ParamSupport.COMPLETION_TOKENS, 200000, 100000,
             devOnly=True, supportsSystem=False),
    ModelSpec("gpto3", "GPT o3", ModelVendor.OPENAI, ParamSupport.COMPLETION_TOKENS, 200000, 100000, devOnly=True),
    ModelSpec("gpto4mini", "GPT o4-mini", ModelVendor.OPENAI, ParamSupport.COMPLETION_TOKENS, 200000, 100000, devOnly=True),
    ModelSpec("gpt41", "GPT-4.1", ModelVendor.OPENAI, ParamSupport.MIXED, 1000000, 16384, devOnly=True),
    ModelSpec("gpt41mini", "GPT-4.1 mini", ModelVendor.OPENAI, ParamSupport.MIXED, 1000000, 16384, devOnly=True),
    ModelSpec("gpt41nano", "GPT-4.1 nano", ModelVendor.OPENAI, ParamSupport.MIXED, 1000000, 16384, devOnly=True),
    ModelSpec("gpt5", "GPT-5", ModelVendor.OPENAI, ParamSupport.NO_MAX_TOKENS, 272000, 128000, devOnly=True),
    ModelSpec("gpt5mini", "GPT-5 mini", ModelVendor.OPENAI, ParamSupport.NO_MAX_TOKENS, 272000, 128000, devOnly=True),
    ModelSpec("gpt5nano", "GPT-5 nano", ModelVendor.OPENAI, ParamSupport.NO_MAX_TOKENS, 272000, 128000, devOnly=True),
    ModelSpec("gpt51", "GPT-5.1", ModelVendor.OPENAI, ParamSupport.NO_MAX_TOKENS, 400000, 128000, devOnly=True),
    ModelSpec("gpt52", "GPT-5.2", ModelVendor.OPENAI, ParamSupport.NO_MAX_TOKENS, 400000, 128000, devOnly=True),
    # ── Google ────────────────────────────────────────────────────────────
    ModelSpec("gemini25pro", "Gemini 2.5 Pro", ModelVendor.GOOGLE, ParamSupport.STANDARD, 1048576, 65536, devOnly=True),
    ModelSpec("gemini25flash", "Gemini 2.5 Flash", ModelVendor.GOOGLE, ParamSupport.STANDARD, 1048576, 64536, devOnly=True),
    # ── Anthropic ─────────────────────────────────────────────────────────
    ModelSpec("claudeopus46", "Claude Opus 4.6", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 128000, devOnly=True),
    ModelSpec("claudeopus45", "Claude Opus 4.5", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 64000, devOnly=True),
    ModelSpec("claudeopus41", "Claude Opus 4.1", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 32000, devOnly=True),
    ModelSpec("claudeopus4", "Claude Opus 4", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 32000, devOnly=True),
    ModelSpec("claudehaiku45", "Claude Haiku 4.5", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_SINGLE, 200000, 64000, devOnly=True),
    ModelSpec("claudesonnet45", "Claude Sonnet 4.5", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_SINGLE, 200000, 64000, devOnly=True),
    ModelSpec("claudesonnet4", "Claude Sonnet 4", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 64000, devOnly=True),
    ModelSpec("claudesonnet37", "Claude Sonnet 3.7", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 128000, devOnly=True),
    ModelSpec("claudesonnet35v2", "Claude Sonnet 3.5v2", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 8000, devOnly=True),
    ModelSpec("claudehaiku35", "Claude Haiku 3.5", ModelVendor.ANTHROPIC, ParamSupport.ANTHROPIC_STANDARD, 200000, 8000, devOnly=True),
]

EMBEDDING_MODELS: list[ModelSpec] = [
    ModelSpec("ada002", "text-embedding-ada-002", ModelVendor.OPENAI, ParamSupport.STANDARD, 8191, 1536),
    ModelSpec("v3large", "text-embedding-3-large", ModelVendor.OPENAI, ParamSupport.STANDARD, 8191, 3072),
    ModelSpec("v3small", "text-embedding-3-small", ModelVendor.OPENAI, ParamSupport.STANDARD, 8191, 1536),
]

EXPECTED_EMBED_DIMS = {"ada002": 1536, "v3large": 3072, "v3small": 1536}


# ---------------------------------------------------------------------------
# Validation data classes
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ValidationCheck:
    """A single validation assertion within a test."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    testName: str
    modelId: str
    status: TestStatus
    durationSec: float = 0.0
    message: str = ""
    responseSnippet: str = ""
    statusCode: Optional[int] = None
    checks: list[ValidationCheck] = field(default_factory=list)


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
    startTime: float = 0.0
    endTime: float = 0.0

    @property
    def totalTests(self) -> int:
        return len(self.results)

    @property
    def passedCount(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failedCount(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def errorCount(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def skippedCount(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def totalChecks(self) -> int:
        return sum(len(r.checks) for r in self.results)

    @property
    def passedChecks(self) -> int:
        return sum(1 for r in self.results for c in r.checks if c.passed)

    @property
    def failedChecks(self) -> int:
        return sum(1 for r in self.results for c in r.checks if not c.passed)

    @property
    def elapsedSec(self) -> float:
        return self.endTime - self.startTime


# ---------------------------------------------------------------------------
# Validators — deep response inspection
# ---------------------------------------------------------------------------

class ResponseValidator:
    """Validates API responses with multiple assertion checks."""

    @staticmethod
    def validateHttpStatus(code: int, expected: int = 200) -> ValidationCheck:
        return ValidationCheck(
            "http_status",
            code == expected,
            f"Expected {expected}, got {code}",
        )

    @staticmethod
    def validateJsonParseable(data: Any) -> ValidationCheck:
        isDict = isinstance(data, dict)
        hasRaw = isDict and "raw" in data
        return ValidationCheck(
            "json_parseable",
            isDict and not hasRaw,
            "Response is valid JSON" if (isDict and not hasRaw) else "Response was not valid JSON",
        )

    @staticmethod
    def validateHasResponseField(data: dict) -> ValidationCheck:
        hasField = "response" in data
        return ValidationCheck(
            "has_response_field",
            hasField,
            "Field 'response' present" if hasField else "Missing 'response' field in payload",
        )

    @staticmethod
    def validateResponseNotEmpty(data: dict) -> ValidationCheck:
        resp = data.get("response", "")
        notEmpty = bool(str(resp).strip())
        return ValidationCheck(
            "response_not_empty",
            notEmpty,
            f"Response has {len(str(resp))} chars" if notEmpty else "Response is empty",
        )

    @staticmethod
    def validateResponseIsString(data: dict) -> ValidationCheck:
        resp = data.get("response")
        isStr = isinstance(resp, str)
        return ValidationCheck(
            "response_is_string",
            isStr,
            f"Type: {type(resp).__name__}" if not isStr else "Response is a string",
        )

    @staticmethod
    def validateNoErrorField(data: dict) -> ValidationCheck:
        hasError = "error" in data
        detail = f"Error field present: {str(data.get('error', ''))[:120]}" if hasError else "No error field"
        return ValidationCheck("no_error_field", not hasError, detail)

    @staticmethod
    def validateResponseContains(data: dict, needle: str, caseSensitive: bool = False) -> ValidationCheck:
        resp = str(data.get("response", ""))
        if caseSensitive:
            found = needle in resp
        else:
            found = needle.lower() in resp.lower()
        return ValidationCheck(
            f"contains_{needle[:20]}",
            found,
            f"Found '{needle}'" if found else f"'{needle}' not found in response",
        )

    @staticmethod
    def validateResponseLength(data: dict, minLen: int = 1, maxLen: int = 50000) -> ValidationCheck:
        resp = str(data.get("response", ""))
        inRange = minLen <= len(resp) <= maxLen
        return ValidationCheck(
            "response_length",
            inRange,
            f"Length {len(resp)} (expected {minLen}-{maxLen})",
        )

    @staticmethod
    def validateLatency(durationSec: float, maxSec: float = 60.0) -> ValidationCheck:
        ok = durationSec <= maxSec
        return ValidationCheck(
            "latency",
            ok,
            f"{durationSec:.2f}s (limit {maxSec:.0f}s)",
        )

    # ── OpenAI-compat response validators ─────────────────────────────

    @staticmethod
    def validateOpenAIShape(data: dict) -> ValidationCheck:
        hasChoices = "choices" in data and isinstance(data.get("choices"), list) and len(data["choices"]) > 0
        return ValidationCheck(
            "openai_shape",
            hasChoices,
            "Has 'choices' array" if hasChoices else "Missing or empty 'choices'",
        )

    @staticmethod
    def validateOpenAIMessageContent(data: dict) -> ValidationCheck:
        try:
            content = data["choices"][0]["message"]["content"]
            ok = isinstance(content, str) and len(content.strip()) > 0
            return ValidationCheck(
                "openai_message_content",
                ok,
                f"Content: {content[:80]}" if ok else "Empty or missing message content",
            )
        except (KeyError, IndexError, TypeError):
            return ValidationCheck("openai_message_content", False, "Could not extract choices[0].message.content")

    @staticmethod
    def validateOpenAIModel(data: dict, expectedModel: str) -> ValidationCheck:
        returnedModel = data.get("model", "")
        return ValidationCheck(
            "openai_model_field",
            bool(returnedModel),
            f"Model: {returnedModel}" if returnedModel else "No 'model' field in response",
        )

    @staticmethod
    def validateOpenAIUsage(data: dict) -> ValidationCheck:
        usage = data.get("usage", {})
        hasPrompt = "prompt_tokens" in usage
        hasCompletion = "completion_tokens" in usage or "total_tokens" in usage
        ok = hasPrompt and hasCompletion
        return ValidationCheck(
            "openai_usage",
            ok,
            f"Usage: {json.dumps(usage)}" if ok else f"Incomplete usage object: {usage}",
        )

    # ── Anthropic messages response validators ────────────────────────

    @staticmethod
    def validateAnthropicShape(data: dict) -> ValidationCheck:
        hasContent = "content" in data and isinstance(data.get("content"), list)
        return ValidationCheck(
            "anthropic_shape",
            hasContent,
            "Has 'content' array" if hasContent else "Missing 'content' array",
        )

    @staticmethod
    def validateAnthropicTextBlock(data: dict) -> ValidationCheck:
        try:
            blocks = data.get("content", [])
            textBlocks = [b for b in blocks if b.get("type") == "text"]
            ok = len(textBlocks) > 0 and len(textBlocks[0].get("text", "").strip()) > 0
            txt = textBlocks[0]["text"][:80] if ok else ""
            return ValidationCheck(
                "anthropic_text_block",
                ok,
                f"Text: {txt}" if ok else "No text block in content",
            )
        except (KeyError, IndexError, TypeError):
            return ValidationCheck("anthropic_text_block", False, "Could not parse content blocks")

    @staticmethod
    def validateAnthropicRole(data: dict) -> ValidationCheck:
        role = data.get("role", "")
        ok = role == "assistant"
        return ValidationCheck("anthropic_role", ok, f"Role: {role}")

    # ── Embedding validators ──────────────────────────────────────────

    @staticmethod
    def validateEmbeddingShape(data: dict) -> ValidationCheck:
        hasData = "data" in data and isinstance(data.get("data"), list) and len(data["data"]) > 0
        return ValidationCheck(
            "embedding_shape",
            hasData,
            f"Has 'data' array with {len(data.get('data', []))} items" if hasData else "Missing 'data' array",
        )

    @staticmethod
    def validateEmbeddingDimension(data: dict, expectedDim: int) -> ValidationCheck:
        try:
            emb = data["data"][0]
            if isinstance(emb, dict):
                vec = emb.get("embedding", [])
            elif isinstance(emb, list):
                vec = emb
            else:
                return ValidationCheck("embedding_dim", False, f"Unexpected type: {type(emb).__name__}")
            ok = len(vec) == expectedDim
            return ValidationCheck(
                "embedding_dim",
                ok,
                f"Dim={len(vec)} (expected {expectedDim})",
            )
        except (KeyError, IndexError):
            return ValidationCheck("embedding_dim", False, "Could not extract embedding vector")

    @staticmethod
    def validateEmbeddingValues(data: dict) -> ValidationCheck:
        try:
            emb = data["data"][0]
            vec = emb.get("embedding", emb) if isinstance(emb, dict) else emb
            allFloats = all(isinstance(v, (int, float)) for v in vec[:10])
            nonZero = any(v != 0 for v in vec[:10])
            ok = allFloats and nonZero
            return ValidationCheck(
                "embedding_values",
                ok,
                "Values are valid floats" if ok else "Values are not valid floats or all zero",
            )
        except (KeyError, IndexError, TypeError):
            return ValidationCheck("embedding_values", False, "Could not inspect embedding values")

    @staticmethod
    def validateEmbeddingCount(data: dict, expectedCount: int) -> ValidationCheck:
        items = data.get("data", [])
        ok = len(items) == expectedCount
        return ValidationCheck(
            "embedding_count",
            ok,
            f"Got {len(items)} embeddings (expected {expectedCount})",
        )

    # ── Streaming validators ──────────────────────────────────────────

    @staticmethod
    def validateStreamReceived(text: str) -> ValidationCheck:
        ok = bool(text.strip())
        return ValidationCheck("stream_received", ok, f"Received {len(text)} chars" if ok else "No stream content")

    @staticmethod
    def validateStreamMultipleChunks(chunkCount: int) -> ValidationCheck:
        ok = chunkCount > 1
        return ValidationCheck("stream_chunks", ok, f"{chunkCount} chunks received")

    # ── Tool calling validators ───────────────────────────────────────

    @staticmethod
    def validateToolCallResponse(data: dict) -> ValidationCheck:
        resp = str(data.get("response", ""))
        # The model either calls the tool (function_call/tool_calls in response) or describes
        # wanting to call it. Either way, the response should not be empty.
        ok = len(resp.strip()) > 0
        return ValidationCheck("tool_response", ok, f"Tool response: {resp[:80]}" if ok else "Empty tool response")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def buildChatPayloadMessages(user: str, model: str, spec: ModelSpec) -> dict:
    payload: dict[str, Any] = {
        "user": user,
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant named Argo."},
            {"role": "user", "content": "Reply with exactly: TEST_OK"},
        ],
    }
    if not spec.supportsSystem:
        payload["messages"] = [{"role": "user", "content": "Reply with exactly: TEST_OK"}]
    _applyParams(payload, spec)
    return payload


def buildChatPayloadPrompt(user: str, model: str, spec: ModelSpec) -> dict:
    payload: dict[str, Any] = {"user": user, "model": model, "prompt": ["Reply with exactly: TEST_OK"]}
    if spec.supportsSystem:
        payload["system"] = "You are a helpful assistant named Argo."
    _applyParams(payload, spec)
    return payload


def buildMultiTurnPayload(user: str, model: str, spec: ModelSpec) -> dict:
    messages = []
    if spec.supportsSystem:
        messages.append({"role": "system", "content": "You are Argo, a helpful assistant."})
    messages.extend([
        {"role": "user", "content": "My name is TestUser."},
        {"role": "assistant", "content": "Hello TestUser! How can I help you today?"},
        {"role": "user", "content": "What is my name? Reply only with the name."},
    ])
    payload: dict[str, Any] = {"user": user, "model": model, "messages": messages}
    _applyParams(payload, spec)
    return payload


def buildEmbeddingPayload(user: str, model: str) -> dict:
    return {"user": user, "model": model, "prompt": ["Argonne National Laboratory is a science and engineering research center."]}


def buildEmbeddingBatchPayload(user: str, model: str) -> dict:
    return {"user": user, "model": model, "prompt": [
        "First test sentence for embeddings.",
        "Second test sentence for embeddings.",
        "Third test sentence for embeddings.",
    ]}


def buildOpenAICompatPayload(user: str, model: str, spec: ModelSpec) -> dict:
    messages = []
    if spec.supportsSystem:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": "Reply with exactly: TEST_OK"})
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if spec.paramSupport in (ParamSupport.STANDARD, ParamSupport.ANTHROPIC_STANDARD, ParamSupport.ANTHROPIC_SINGLE):
        payload["temperature"] = 0.1
        payload["max_tokens"] = 100
    elif spec.paramSupport == ParamSupport.COMPLETION_TOKENS:
        payload["max_completion_tokens"] = 100
    elif spec.paramSupport in (ParamSupport.MIXED, ParamSupport.NO_MAX_TOKENS):
        payload["temperature"] = 0.1
        payload["max_completion_tokens"] = 100
    return payload


def buildAnthropicMessagesPayload(model: str, spec: ModelSpec) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: TEST_OK"}],
        "max_tokens": 100,
    }
    if spec.supportsSystem:
        payload["system"] = "You are a helpful assistant."
    return payload


def buildToolCallingPayload(user: str, model: str, spec: ModelSpec) -> dict:
    messages = []
    if spec.supportsSystem:
        messages.append({"role": "system", "content": "You are a helpful assistant with access to tools."})
    messages.append({"role": "user", "content": "What is the weather in Chicago?"})
    tools = [{
        "type": "function",
        "function": {
            "name": "getWeather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    }]
    payload: dict[str, Any] = {"user": user, "model": model, "messages": messages, "tools": tools}
    _applyParams(payload, spec)
    return payload


def _applyParams(payload: dict, spec: ModelSpec) -> None:
    if spec.paramSupport == ParamSupport.STANDARD:
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_tokens": 200})
    elif spec.paramSupport == ParamSupport.COMPLETION_TOKENS:
        payload["max_completion_tokens"] = 200
    elif spec.paramSupport == ParamSupport.MIXED:
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_completion_tokens": 200})
    elif spec.paramSupport == ParamSupport.ANTHROPIC_STANDARD:
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_tokens": 200})
    elif spec.paramSupport == ParamSupport.ANTHROPIC_SINGLE:
        payload.update({"temperature": 0.1, "max_tokens": 200})
    elif spec.paramSupport == ParamSupport.NO_MAX_TOKENS:
        payload.update({"temperature": 0.1, "top_p": 0.9, "max_completion_tokens": 200})


def extractResponseText(data: dict) -> str:
    if "response" in data:
        return str(data["response"])[:300]
    if "choices" in data and data["choices"]:
        ch = data["choices"][0]
        if "message" in ch and "content" in ch["message"]:
            return str(ch["message"]["content"])[:300]
        if "delta" in ch and "content" in ch["delta"]:
            return str(ch["delta"]["content"])[:300]
    if "content" in data and isinstance(data["content"], list):
        parts = [c.get("text", "") for c in data["content"] if c.get("type") == "text"]
        return "".join(parts)[:300]
    if "data" in data and isinstance(data["data"], list):
        first = data["data"][0]
        if isinstance(first, dict) and "embedding" in first:
            emb = first["embedding"]
            return f"[embedding dim={len(emb)}]"
        if isinstance(first, list):
            return f"[embedding dim={len(first)}]"
    return str(data)[:300]


def snippetFor(text: str, maxLen: int = 120) -> str:
    cleaned = text.replace("\n", " ").strip()
    return cleaned[:maxLen] + "..." if len(cleaned) > maxLen else cleaned


def statusIcon(status: TestStatus) -> str:
    return {
        TestStatus.PASSED: "[bold green]PASS[/]",
        TestStatus.FAILED: "[bold red]FAIL[/]",
        TestStatus.ERROR: "[bold yellow]ERROR[/]",
        TestStatus.SKIPPED: "[dim]SKIP[/]",
    }.get(status, "?")


def checkIcon(passed: bool) -> str:
    return "[green]OK[/]" if passed else "[red]X[/]"


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

class ArgoTestRunner:
    def __init__(self, user: str, env: str, verbose: bool = False):
        self.user = user
        self.env = env
        self.verbose = verbose
        self.baseUrl = BASE_URLS[env]
        self.suite = TestSuite()
        self.v = ResponseValidator()

    # -- HTTP helpers --

    def _post(self, path: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> tuple[int, dict]:
        url = self.baseUrl + path
        if self.verbose:
            console.print(f"  [dim]POST {url}[/]")
            console.print(f"  [dim]Payload: {json.dumps(payload, indent=2)[:500]}[/]")
        resp = requests.post(url, json=payload, timeout=timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
        if self.verbose:
            console.print(f"  [dim]Status: {resp.status_code}[/]")
            console.print(f"  [dim]Response: {json.dumps(data, default=str)[:500]}[/]")
        return resp.status_code, data

    def _postOpenAICompat(self, payload: dict, timeout: int = REQUEST_TIMEOUT) -> tuple[int, dict]:
        url = self.baseUrl + OPENAI_COMPAT_PATH
        headers = {"Authorization": f"Bearer {self.user}", "Content-Type": "application/json"}
        if self.verbose:
            console.print(f"  [dim]POST {url}[/]")
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
        return resp.status_code, data

    def _postAnthropicMessages(self, payload: dict, timeout: int = REQUEST_TIMEOUT) -> tuple[int, dict]:
        url = self.baseUrl + ANTHROPIC_MESSAGES_PATH
        headers = {"x-api-key": self.user, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.verbose:
            console.print(f"  [dim]POST {url}[/]")
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
        return resp.status_code, data

    def _streamPost(self, path: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> tuple[bool, str, float, int]:
        """Returns (success, collected_text, duration, chunk_count)."""
        url = self.baseUrl + path
        collected = []
        chunkCount = 0
        start = time.time()
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        return False, f"HTTP {resp.status_code}", time.time() - start, 0
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            chunkCount += 1
                            try:
                                parsed = json.loads(chunk)
                                txt = extractResponseText(parsed)
                                if txt:
                                    collected.append(txt)
                            except json.JSONDecodeError:
                                collected.append(chunk[:100])
                        else:
                            collected.append(line[:100])
                            chunkCount += 1
        except Exception as exc:
            return False, str(exc)[:300], time.time() - start, chunkCount
        duration = time.time() - start
        fullText = " ".join(collected)
        return bool(fullText.strip()), fullText[:300], duration, chunkCount

    # -- record & display helper --

    def _record(self, testName: str, modelId: str, checks: list[ValidationCheck],
                durationSec: float = 0.0, statusCode: Optional[int] = None,
                responseSnippet: str = "", overrideStatus: Optional[TestStatus] = None) -> TestResult:
        allPassed = all(c.passed for c in checks)
        status = overrideStatus if overrideStatus else (TestStatus.PASSED if allPassed else TestStatus.FAILED)

        passedChecks = sum(1 for c in checks if c.passed)
        failedChecks = [c for c in checks if not c.passed]

        message = f"{passedChecks}/{len(checks)} checks passed"
        if failedChecks:
            message += " | Failed: " + ", ".join(c.name for c in failedChecks)

        result = TestResult(
            testName=testName, modelId=modelId, status=status,
            durationSec=durationSec, message=message,
            responseSnippet=responseSnippet, statusCode=statusCode, checks=checks,
        )
        self.suite.results.append(result)

        # Print result line
        icon = statusIcon(status)
        timing = f"[dim]({durationSec:.2f}s)[/]" if durationSec > 0 else ""
        checksStr = f"[dim][{passedChecks}/{len(checks)}][/]"
        console.print(f"  {icon} {checksStr} {testName} [cyan]{modelId}[/] {timing}")

        # Print check details
        for c in checks:
            ci = checkIcon(c.passed)
            console.print(f"       {ci} {c.name}: {escape(c.detail[:100])}")

        return result

    def _recordSkip(self, testName: str, modelId: str, reason: str) -> TestResult:
        result = TestResult(testName=testName, modelId=modelId, status=TestStatus.SKIPPED, message=reason)
        self.suite.results.append(result)
        console.print(f"  [dim]SKIP[/] {testName} [cyan]{modelId}[/] [dim]{reason}[/]")
        return result

    def _recordError(self, testName: str, modelId: str, exc: Exception, durationSec: float = 0.0) -> TestResult:
        result = TestResult(
            testName=testName, modelId=modelId, status=TestStatus.ERROR,
            durationSec=durationSec, message=str(exc)[:200],
        )
        self.suite.results.append(result)
        console.print(f"  [bold yellow]ERROR[/] {testName} [cyan]{modelId}[/] [yellow]{escape(str(exc)[:120])}[/]")
        return result

    # ──────────────────────────────────────────────────────────────────
    # Individual tests with deep validation
    # ──────────────────────────────────────────────────────────────────

    def testChatMessages(self, spec: ModelSpec) -> TestResult:
        """Test /chat/ with messages object — validates response structure."""
        payload = buildChatPayloadMessages(self.user, spec.modelId, spec)
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateNoErrorField(data),
                self.v.validateHasResponseField(data),
                self.v.validateResponseNotEmpty(data),
                self.v.validateResponseIsString(data),
                self.v.validateLatency(dur),
            ]
            return self._record("chat/messages", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("chat/messages", spec.modelId, exc, time.time() - start)

    def testChatPrompt(self, spec: ModelSpec) -> TestResult:
        """Test /chat/ with system + prompt fields — validates response structure."""
        payload = buildChatPayloadPrompt(self.user, spec.modelId, spec)
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateNoErrorField(data),
                self.v.validateHasResponseField(data),
                self.v.validateResponseNotEmpty(data),
                self.v.validateResponseIsString(data),
                self.v.validateLatency(dur),
            ]
            return self._record("chat/prompt", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("chat/prompt", spec.modelId, exc, time.time() - start)

    def testMultiTurn(self, spec: ModelSpec) -> TestResult:
        """Test multi-turn conversation — validates context retention."""
        payload = buildMultiTurnPayload(self.user, spec.modelId, spec)
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateNoErrorField(data),
                self.v.validateHasResponseField(data),
                self.v.validateResponseNotEmpty(data),
                self.v.validateResponseContains(data, "TestUser"),
                self.v.validateLatency(dur),
            ]
            return self._record("chat/multi-turn", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("chat/multi-turn", spec.modelId, exc, time.time() - start)

    def testStreaming(self, spec: ModelSpec) -> TestResult:
        """Test /streamchat/ — validates streaming response delivery."""
        if not spec.supportsStreaming:
            return self._recordSkip("stream", spec.modelId, "Streaming not supported")
        payload = buildChatPayloadMessages(self.user, spec.modelId, spec)
        start = time.time()
        try:
            ok, text, dur, chunkCount = self._streamPost(STREAM_PATH, payload)
            checks = [
                self.v.validateStreamReceived(text),
                self.v.validateStreamMultipleChunks(chunkCount),
                self.v.validateLatency(dur),
            ]
            return self._record("stream", spec.modelId, checks, dur, responseSnippet=snippetFor(text))
        except Exception as exc:
            return self._recordError("stream", spec.modelId, exc, time.time() - start)

    def testOpenAICompat(self, spec: ModelSpec) -> TestResult:
        """Test OpenAI-compatible /v1/chat/completions — validates OpenAI response shape."""
        if self.env != "dev":
            return self._recordSkip("openai-compat", spec.modelId, "Only available in dev")
        payload = buildOpenAICompatPayload(self.user, spec.modelId, spec)
        start = time.time()
        try:
            code, data = self._postOpenAICompat(payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateOpenAIShape(data),
                self.v.validateOpenAIMessageContent(data),
                self.v.validateOpenAIModel(data, spec.modelId),
                self.v.validateOpenAIUsage(data),
                self.v.validateLatency(dur),
            ]
            return self._record("openai-compat", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("openai-compat", spec.modelId, exc, time.time() - start)

    def testAnthropicMessages(self, spec: ModelSpec) -> TestResult:
        """Test Anthropic /v1/messages — validates Anthropic response shape."""
        if spec.vendor != ModelVendor.ANTHROPIC:
            return self._recordSkip("anthropic-messages", spec.modelId, "Not an Anthropic model")
        if self.env != "dev":
            return self._recordSkip("anthropic-messages", spec.modelId, "Only available in dev")
        payload = buildAnthropicMessagesPayload(spec.modelId, spec)
        start = time.time()
        try:
            code, data = self._postAnthropicMessages(payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateAnthropicShape(data),
                self.v.validateAnthropicTextBlock(data),
                self.v.validateAnthropicRole(data),
                self.v.validateLatency(dur),
            ]
            return self._record("anthropic-messages", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("anthropic-messages", spec.modelId, exc, time.time() - start)

    def testToolCalling(self, spec: ModelSpec) -> TestResult:
        """Test tool/function calling via /chat/ — validates tool interaction."""
        if not spec.supportsTools:
            return self._recordSkip("tool-calling", spec.modelId, "Tools not supported")
        payload = buildToolCallingPayload(self.user, spec.modelId, spec)
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateNoErrorField(data),
                self.v.validateToolCallResponse(data),
                self.v.validateLatency(dur),
            ]
            return self._record("tool-calling", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("tool-calling", spec.modelId, exc, time.time() - start)

    def testEmbedding(self, spec: ModelSpec) -> TestResult:
        """Test /embed/ single string — validates embedding dimensions."""
        payload = buildEmbeddingPayload(self.user, spec.modelId)
        expectedDim = EXPECTED_EMBED_DIMS.get(spec.modelId, spec.maxOutputTokens)
        start = time.time()
        try:
            code, data = self._post(EMBED_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateEmbeddingShape(data),
                self.v.validateEmbeddingDimension(data, expectedDim),
                self.v.validateEmbeddingValues(data),
                self.v.validateLatency(dur),
            ]
            return self._record("embed/single", spec.modelId, checks, dur, code, snippetFor(extractResponseText(data)))
        except Exception as exc:
            return self._recordError("embed/single", spec.modelId, exc, time.time() - start)

    def testEmbeddingBatch(self, spec: ModelSpec) -> TestResult:
        """Test /embed/ with 3 strings — validates batch output count."""
        payload = buildEmbeddingBatchPayload(self.user, spec.modelId)
        expectedDim = EXPECTED_EMBED_DIMS.get(spec.modelId, spec.maxOutputTokens)
        start = time.time()
        try:
            code, data = self._post(EMBED_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateJsonParseable(data),
                self.v.validateEmbeddingShape(data),
                self.v.validateEmbeddingCount(data, 3),
                self.v.validateEmbeddingDimension(data, expectedDim),
                self.v.validateLatency(dur),
            ]
            return self._record("embed/batch", spec.modelId, checks, dur, code)
        except Exception as exc:
            return self._recordError("embed/batch", spec.modelId, exc, time.time() - start)

    # ──────────────────────────────────────────────────────────────────
    # Validation / negative tests
    # ──────────────────────────────────────────────────────────────────

    def testInvalidModel(self) -> TestResult:
        """Verify the API rejects an invalid model name."""
        payload = {"user": self.user, "model": "nonexistent_xyz", "prompt": ["test"], "temperature": 0.1, "max_tokens": 50}
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                self.v.validateHttpStatus(code, expected=422),  # expect rejection
                ValidationCheck("rejected", code != 200, f"HTTP {code} — {'rejected as expected' if code != 200 else 'incorrectly accepted'}"),
            ]
            # If the API returns any non-200, that counts as correct behavior
            if code != 200:
                checks = [
                    ValidationCheck("http_non_200", True, f"Correctly returned HTTP {code}"),
                    ValidationCheck("rejected", True, "Invalid model rejected"),
                ]
            else:
                checks = [
                    ValidationCheck("http_non_200", False, f"Expected non-200 but got {code}"),
                    ValidationCheck("rejected", False, "API accepted invalid model — should reject"),
                ]
            return self._record("validation/invalid-model", "N/A", checks, dur, code)
        except Exception as exc:
            return self._recordError("validation/invalid-model", "N/A", exc, time.time() - start)

    def testMissingUser(self) -> TestResult:
        """Verify behavior when user field is missing."""
        payload = {"model": "gpt4o", "prompt": ["test"], "temperature": 0.1, "max_tokens": 50}
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                ValidationCheck("missing_user_handled", True, f"HTTP {code} — API handled missing user field"),
            ]
            return self._record("validation/missing-user", "gpt4o", checks, dur, code)
        except Exception as exc:
            return self._recordError("validation/missing-user", "gpt4o", exc, time.time() - start)

    def testMissingModel(self) -> TestResult:
        """Verify behavior when model field is missing."""
        payload = {"user": self.user, "prompt": ["test"], "temperature": 0.1, "max_tokens": 50}
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            rejected = code != 200
            checks = [
                ValidationCheck("missing_model_rejected", rejected,
                                f"HTTP {code} — {'rejected' if rejected else 'accepted (should reject)'}"),
            ]
            return self._record("validation/missing-model", "N/A", checks, dur, code)
        except Exception as exc:
            return self._recordError("validation/missing-model", "N/A", exc, time.time() - start)

    def testEmptyPrompt(self) -> TestResult:
        """Verify behavior with empty prompt."""
        payload = {"user": self.user, "model": "gpt4o", "prompt": [""], "temperature": 0.1, "max_tokens": 50}
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                ValidationCheck("empty_prompt_handled", True, f"HTTP {code} — API handled empty prompt gracefully"),
                self.v.validateJsonParseable(data),
            ]
            return self._record("validation/empty-prompt", "gpt4o", checks, dur, code)
        except Exception as exc:
            return self._recordError("validation/empty-prompt", "gpt4o", exc, time.time() - start)

    def testStopSequence(self) -> TestResult:
        """Test that stop sequences truncate the response."""
        payload = {
            "user": self.user, "model": "gpt4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20, each number separated by a comma."},
            ],
            "stop": [","], "temperature": 0.0, "max_tokens": 200,
        }
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            resp = str(data.get("response", ""))
            # With stop=[","], response should be very short (just "1" or similar)
            stopRespected = "," not in resp
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateHasResponseField(data),
                ValidationCheck("stop_respected", stopRespected,
                                f"No comma in response" if stopRespected else f"Comma found — stop not honored: {resp[:60]}"),
                ValidationCheck("response_short", len(resp) < 100,
                                f"Response length: {len(resp)} chars (expected short due to stop)"),
            ]
            return self._record("feature/stop-sequence", "gpt4o", checks, dur, code, snippetFor(resp))
        except Exception as exc:
            return self._recordError("feature/stop-sequence", "gpt4o", exc, time.time() - start)

    def testTemperatureRange(self) -> TestResult:
        """Test with temperature = 0 (deterministic) and temperature = 2 (max)."""
        checks = []
        totalDur = 0.0
        for temp in [0.0, 2.0]:
            payload = {
                "user": self.user, "model": "gpt4o",
                "messages": [{"role": "user", "content": "Say hello."}],
                "temperature": temp, "max_tokens": 50,
            }
            start = time.time()
            try:
                code, data = self._post(CHAT_PATH, payload)
                dur = time.time() - start
                totalDur += dur
                checks.append(ValidationCheck(
                    f"temp_{temp}", code == 200, f"temp={temp} -> HTTP {code} ({dur:.2f}s)",
                ))
            except Exception as exc:
                totalDur += time.time() - start
                checks.append(ValidationCheck(f"temp_{temp}", False, str(exc)[:80]))
        return self._record("feature/temperature-range", "gpt4o", checks, totalDur)

    def testMaxTokensLimit(self) -> TestResult:
        """Test that max_tokens = 10 produces a short response."""
        payload = {
            "user": self.user, "model": "gpt4o",
            "messages": [{"role": "user", "content": "Write a very long essay about the history of physics."}],
            "temperature": 0.1, "max_tokens": 10,
        }
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            resp = str(data.get("response", ""))
            # 10 tokens is roughly 40-80 chars
            isShort = len(resp) < 200
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateHasResponseField(data),
                ValidationCheck("response_truncated", isShort,
                                f"Length: {len(resp)} chars ({'truncated' if isShort else 'too long — max_tokens may not be honored'})"),
            ]
            return self._record("feature/max-tokens-limit", "gpt4o", checks, dur, code, snippetFor(resp))
        except Exception as exc:
            return self._recordError("feature/max-tokens-limit", "gpt4o", exc, time.time() - start)

    def testAnthropicSingleParamConstraint(self) -> TestResult:
        """Test that Anthropic models with ANTHROPIC_SINGLE reject both temp and top_p."""
        singleModels = [m for m in ALL_MODELS if m.paramSupport == ParamSupport.ANTHROPIC_SINGLE]
        if not singleModels:
            return self._recordSkip("feature/anthropic-single-param", "N/A", "No ANTHROPIC_SINGLE models available")
        spec = singleModels[0]
        # Send with ONLY temperature — should succeed
        payload = {
            "user": self.user, "model": spec.modelId,
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0.5, "max_tokens": 50,
        }
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            checks = [
                ValidationCheck("single_temp_ok", code == 200,
                                f"temp-only: HTTP {code} ({'accepted' if code == 200 else 'rejected'})"),
                self.v.validateLatency(dur),
            ]
            return self._record("feature/anthropic-single-param", spec.modelId, checks, dur, code)
        except Exception as exc:
            return self._recordError("feature/anthropic-single-param", spec.modelId, exc, time.time() - start)

    def testReasoningModelConstraint(self) -> TestResult:
        """Test that o-series models work without temperature/top_p."""
        reasoningModels = [m for m in ALL_MODELS
                          if m.paramSupport == ParamSupport.COMPLETION_TOKENS and not m.devOnly]
        if not reasoningModels:
            return self._recordSkip("feature/reasoning-constraint", "N/A", "No available reasoning models in this env")
        spec = reasoningModels[0]
        payload = {
            "user": self.user, "model": spec.modelId,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
            "max_completion_tokens": 100,
        }
        start = time.time()
        try:
            code, data = self._post(CHAT_PATH, payload)
            dur = time.time() - start
            resp = str(data.get("response", ""))
            has4 = "4" in resp
            checks = [
                self.v.validateHttpStatus(code),
                self.v.validateHasResponseField(data),
                ValidationCheck("reasoning_correct", has4, f"Contains '4': {has4} (response: {resp[:60]})"),
                self.v.validateLatency(dur, maxSec=90.0),
            ]
            return self._record("feature/reasoning-constraint", spec.modelId, checks, dur, code, snippetFor(resp))
        except Exception as exc:
            return self._recordError("feature/reasoning-constraint", spec.modelId, exc, time.time() - start)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def filterModels(models: list[ModelSpec], env: str,
                 category: Optional[str] = None,
                 modelFilter: Optional[str] = None) -> list[ModelSpec]:
    result = []
    for m in models:
        if m.devOnly and env == "prod":
            continue
        if category:
            if category == "openai" and m.vendor != ModelVendor.OPENAI:
                continue
            if category == "google" and m.vendor != ModelVendor.GOOGLE:
                continue
            if category == "anthropic" and m.vendor != ModelVendor.ANTHROPIC:
                continue
        if modelFilter and m.modelId != modelFilter:
            continue
        result.append(m)
    return result


def printModelTable(models: list[ModelSpec], title: str = "Available Chat Models") -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Model ID", style="cyan", min_width=16)
    table.add_column("Display Name", style="white")
    table.add_column("Vendor", style="magenta")
    table.add_column("Params", style="green")
    table.add_column("Max In", justify="right", style="yellow")
    table.add_column("Max Out", justify="right", style="yellow")
    table.add_column("Dev Only", justify="center")
    table.add_column("Tools", justify="center")
    table.add_column("Stream", justify="center")
    for m in models:
        table.add_row(
            m.modelId, m.displayName, m.vendor.value, m.paramSupport.value,
            f"{m.maxInputTokens:,}", f"{m.maxOutputTokens:,}",
            "Yes" if m.devOnly else "", "Yes" if m.supportsTools else "No",
            "Yes" if m.supportsStreaming else "No",
        )
    console.print(table)


def printSummary(suite: TestSuite) -> None:
    console.print()
    console.print(Rule("[bold]Test Summary[/]"))
    console.print()

    # ── High-level stats ──────────────────────────────────────────────
    statsTable = Table(show_header=False, box=None, padding=(0, 2))
    statsTable.add_column("Label", style="bold")
    statsTable.add_column("Value", justify="right")
    statsTable.add_row("Total Tests", str(suite.totalTests))
    statsTable.add_row("[green]Passed[/]", str(suite.passedCount))
    statsTable.add_row("[red]Failed[/]", str(suite.failedCount))
    statsTable.add_row("[yellow]Errors[/]", str(suite.errorCount))
    statsTable.add_row("[dim]Skipped[/]", str(suite.skippedCount))
    statsTable.add_row("", "")
    statsTable.add_row("Total Checks", str(suite.totalChecks))
    statsTable.add_row("[green]Checks Passed[/]", str(suite.passedChecks))
    statsTable.add_row("[red]Checks Failed[/]", str(suite.failedChecks))
    statsTable.add_row("", "")
    statsTable.add_row("Total Time", f"{suite.elapsedSec:.1f}s")
    console.print(Panel(statsTable, title="Results Overview", border_style="blue"))

    # ── What's Working — services & capabilities that passed ──────────
    console.print()
    console.print(Rule("[bold green]What's Working[/]"))
    workingTree = Tree("[green]Operational[/]")

    workingModels = sorted(set(r.modelId for r in suite.results if r.status == TestStatus.PASSED and r.modelId != "N/A"))
    if workingModels:
        modelsNode = workingTree.add(f"[bold]Models responding ({len(workingModels)})[/]")
        for mid in workingModels:
            passedTests = [r.testName for r in suite.results if r.modelId == mid and r.status == TestStatus.PASSED]
            modelsNode.add(f"[cyan]{mid}[/] — {', '.join(passedTests)}")

    workingEndpoints = set()
    for r in suite.results:
        if r.status == TestStatus.PASSED:
            if r.testName.startswith("chat/"):
                workingEndpoints.add("/chat/")
            elif r.testName.startswith("stream"):
                workingEndpoints.add("/streamchat/")
            elif r.testName.startswith("embed"):
                workingEndpoints.add("/embed/")
            elif r.testName.startswith("openai-compat"):
                workingEndpoints.add("/v1/chat/completions")
            elif r.testName.startswith("anthropic"):
                workingEndpoints.add("/v1/messages")
    if workingEndpoints:
        epNode = workingTree.add(f"[bold]Endpoints operational ({len(workingEndpoints)})[/]")
        for ep in sorted(workingEndpoints):
            epNode.add(f"[green]{ep}[/]")

    workingFeatures = [r.testName for r in suite.results if r.testName.startswith("feature/") and r.status == TestStatus.PASSED]
    if workingFeatures:
        featNode = workingTree.add(f"[bold]Features validated ({len(workingFeatures)})[/]")
        for f in workingFeatures:
            featNode.add(f"[green]{f}[/]")

    workingValidation = [r.testName for r in suite.results if r.testName.startswith("validation/") and r.status == TestStatus.PASSED]
    if workingValidation:
        valNode = workingTree.add(f"[bold]Validation checks ({len(workingValidation)})[/]")
        for v in workingValidation:
            valNode.add(f"[green]{v}[/]")

    console.print(workingTree)

    # ── What's Not Working — detailed breakdown of failures ───────────
    problems = [r for r in suite.results if r.status in (TestStatus.FAILED, TestStatus.ERROR)]
    if problems:
        console.print()
        console.print(Rule("[bold red]What's Not Working[/]"))
        brokenTree = Tree("[red]Issues Found[/]")

        failedModels: dict[str, list[TestResult]] = {}
        for r in problems:
            failedModels.setdefault(r.modelId, []).append(r)

        for mid, results in sorted(failedModels.items()):
            modelNode = brokenTree.add(f"[bold red]{mid}[/]")
            for r in results:
                statusStr = "[red]FAIL[/]" if r.status == TestStatus.FAILED else "[yellow]ERROR[/]"
                httpStr = f"HTTP {r.statusCode}" if r.statusCode else ""
                testNode = modelNode.add(f"{statusStr} {r.testName} {httpStr}")
                # Show which checks failed
                failedChecks = [c for c in r.checks if not c.passed]
                if failedChecks:
                    for c in failedChecks:
                        testNode.add(f"[red]X[/] {c.name}: {escape(c.detail[:100])}")
                elif r.message:
                    testNode.add(f"[dim]{escape(r.message[:120])}[/]")

        console.print(brokenTree)

        # Also show a compact table
        console.print()
        failTable = Table(title="Failure & Error Detail", show_lines=True)
        failTable.add_column("Test", style="cyan")
        failTable.add_column("Model", style="magenta")
        failTable.add_column("Status")
        failTable.add_column("HTTP", justify="center")
        failTable.add_column("Failed Checks", max_width=50)
        failTable.add_column("Message", max_width=40)

        for r in problems:
            statusStr = "[red]FAIL[/]" if r.status == TestStatus.FAILED else "[yellow]ERR[/]"
            failedNames = ", ".join(c.name for c in r.checks if not c.passed) if r.checks else "-"
            failTable.add_row(
                r.testName, r.modelId, statusStr,
                str(r.statusCode) if r.statusCode else "-",
                escape(failedNames[:50]),
                escape(r.message[:40]),
            )
        console.print(failTable)
    else:
        console.print()
        console.print(Panel("[bold green]No failures or errors detected![/]", border_style="green"))

    # ── Per-model breakdown ───────────────────────────────────────────
    console.print()
    console.print(Rule("Per-Model Breakdown"))
    modelIds = sorted(set(r.modelId for r in suite.results))
    breakdownTable = Table(show_lines=True)
    breakdownTable.add_column("Model", style="cyan", min_width=18)
    breakdownTable.add_column("Pass", justify="right", style="green")
    breakdownTable.add_column("Fail", justify="right", style="red")
    breakdownTable.add_column("Error", justify="right", style="yellow")
    breakdownTable.add_column("Skip", justify="right", style="dim")
    breakdownTable.add_column("Checks OK", justify="right", style="green")
    breakdownTable.add_column("Checks X", justify="right", style="red")
    breakdownTable.add_column("Avg Time", justify="right")
    breakdownTable.add_column("Health")

    for mid in modelIds:
        mr = [r for r in suite.results if r.modelId == mid]
        p = sum(1 for r in mr if r.status == TestStatus.PASSED)
        f = sum(1 for r in mr if r.status == TestStatus.FAILED)
        e = sum(1 for r in mr if r.status == TestStatus.ERROR)
        s = sum(1 for r in mr if r.status == TestStatus.SKIPPED)
        cOk = sum(1 for r in mr for c in r.checks if c.passed)
        cFail = sum(1 for r in mr for c in r.checks if not c.passed)
        times = [r.durationSec for r in mr if r.durationSec > 0]
        avgTime = f"{sum(times) / len(times):.2f}s" if times else "-"
        total = p + f + e
        if total == 0:
            health = "[dim]N/A[/]"
        elif f == 0 and e == 0:
            health = "[bold green]Healthy[/]"
        elif p > f + e:
            health = "[yellow]Degraded[/]"
        else:
            health = "[bold red]Down[/]"
        breakdownTable.add_row(mid, str(p), str(f), str(e), str(s), str(cOk), str(cFail), avgTime, health)
    console.print(breakdownTable)

    # ── Endpoint health summary ───────────────────────────────────────
    console.print()
    console.print(Rule("Endpoint Health"))
    endpointMap: dict[str, list[TestResult]] = {}
    for r in suite.results:
        if r.testName.startswith("chat/"):
            endpointMap.setdefault("Chat /chat/", []).append(r)
        elif r.testName.startswith("stream"):
            endpointMap.setdefault("Stream /streamchat/", []).append(r)
        elif r.testName.startswith("embed"):
            endpointMap.setdefault("Embed /embed/", []).append(r)
        elif r.testName.startswith("openai-compat"):
            endpointMap.setdefault("OpenAI-compat /v1/chat/completions", []).append(r)
        elif r.testName.startswith("anthropic"):
            endpointMap.setdefault("Anthropic /v1/messages", []).append(r)
        elif r.testName.startswith("tool"):
            endpointMap.setdefault("Tool Calling /chat/", []).append(r)
        elif r.testName.startswith("feature/") or r.testName.startswith("validation/"):
            endpointMap.setdefault("Validation & Features", []).append(r)

    epTable = Table(show_lines=True)
    epTable.add_column("Endpoint", style="bold")
    epTable.add_column("Tests", justify="right")
    epTable.add_column("Pass", justify="right", style="green")
    epTable.add_column("Fail", justify="right", style="red")
    epTable.add_column("Error", justify="right", style="yellow")
    epTable.add_column("Status")

    for epName, results in sorted(endpointMap.items()):
        p = sum(1 for r in results if r.status == TestStatus.PASSED)
        f = sum(1 for r in results if r.status == TestStatus.FAILED)
        e = sum(1 for r in results if r.status == TestStatus.ERROR)
        total = p + f + e
        if total == 0:
            status = "[dim]No data[/]"
        elif f == 0 and e == 0:
            status = "[bold green]Operational[/]"
        elif p > 0:
            status = "[yellow]Partial[/]"
        else:
            status = "[bold red]Down[/]"
        epTable.add_row(epName, str(len(results)), str(p), str(f), str(e), status)
    console.print(epTable)

    # ── Final verdict ─────────────────────────────────────────────────
    console.print()
    if suite.failedCount == 0 and suite.errorCount == 0:
        console.print(Panel(
            f"[bold green]ALL {suite.totalTests} TESTS PASSED[/]\n"
            f"[green]{suite.passedChecks}/{suite.totalChecks} validation checks passed[/]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]{suite.failedCount} FAILED, {suite.errorCount} ERRORS[/] "
            f"out of {suite.totalTests} tests\n"
            f"[red]{suite.failedChecks}/{suite.totalChecks} validation checks failed[/]",
            border_style="red",
        ))


def runAllTests(runner: ArgoTestRunner, chatModels: list[ModelSpec],
                embedModels: list[ModelSpec], skipStream: bool = False,
                skipCompat: bool = False, skipTools: bool = False) -> None:
    totalSteps = (
        len(chatModels) * 3
        + (len(chatModels) if not skipStream else 0)
        + (len(chatModels) if not skipCompat else 0)
        + (len([m for m in chatModels if m.vendor == ModelVendor.ANTHROPIC]) if not skipCompat else 0)
        + (len([m for m in chatModels if m.supportsTools]) if not skipTools else 0)
        + len(embedModels) * 2
        + 8  # validation & feature tests
    )

    runner.suite.startTime = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running Argo tests...", total=totalSteps)

        # ── Chat: messages ────────────────────────────────────────────
        console.print()
        console.print(Rule("[bold cyan]Chat Endpoint — Messages Format[/]"))
        for spec in chatModels:
            runner.testChatMessages(spec)
            progress.advance(task)

        # ── Chat: prompt ──────────────────────────────────────────────
        console.print()
        console.print(Rule("[bold cyan]Chat Endpoint — Prompt Format[/]"))
        for spec in chatModels:
            runner.testChatPrompt(spec)
            progress.advance(task)

        # ── Multi-turn ────────────────────────────────────────────────
        console.print()
        console.print(Rule("[bold cyan]Multi-Turn Conversations[/]"))
        for spec in chatModels:
            runner.testMultiTurn(spec)
            progress.advance(task)

        # ── Streaming ─────────────────────────────────────────────────
        if not skipStream:
            console.print()
            console.print(Rule("[bold cyan]Streaming Endpoint[/]"))
            for spec in chatModels:
                runner.testStreaming(spec)
                progress.advance(task)

        # ── OpenAI-compatible ─────────────────────────────────────────
        if not skipCompat:
            console.print()
            console.print(Rule("[bold cyan]OpenAI-Compatible Endpoint[/]"))
            for spec in chatModels:
                runner.testOpenAICompat(spec)
                progress.advance(task)

            console.print()
            console.print(Rule("[bold cyan]Anthropic Messages Endpoint[/]"))
            for spec in chatModels:
                if spec.vendor == ModelVendor.ANTHROPIC:
                    runner.testAnthropicMessages(spec)
                    progress.advance(task)

        # ── Tool calling ──────────────────────────────────────────────
        if not skipTools:
            console.print()
            console.print(Rule("[bold cyan]Tool / Function Calling[/]"))
            for spec in chatModels:
                if spec.supportsTools:
                    runner.testToolCalling(spec)
                    progress.advance(task)

        # ── Embeddings ────────────────────────────────────────────────
        console.print()
        console.print(Rule("[bold cyan]Embedding Endpoint[/]"))
        for spec in embedModels:
            runner.testEmbedding(spec)
            progress.advance(task)
            runner.testEmbeddingBatch(spec)
            progress.advance(task)

        # ── Validation & negative tests ───────────────────────────────
        console.print()
        console.print(Rule("[bold cyan]Validation & Negative Tests[/]"))
        runner.testInvalidModel()
        progress.advance(task)
        runner.testMissingUser()
        progress.advance(task)
        runner.testMissingModel()
        progress.advance(task)
        runner.testEmptyPrompt()
        progress.advance(task)

        # ── Feature tests ─────────────────────────────────────────────
        console.print()
        console.print(Rule("[bold cyan]Feature Tests[/]"))
        runner.testStopSequence()
        progress.advance(task)
        runner.testTemperatureRange()
        progress.advance(task)
        runner.testMaxTokensLimit()
        progress.advance(task)
        runner.testAnthropicSingleParamConstraint()
        progress.advance(task)
        runner.testReasoningModelConstraint()

    runner.suite.endTime = time.time()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Argo Gateway API — Comprehensive Test Suite with Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python argo_test_suite.py --user mdearing
  python argo_test_suite.py --user mdearing --env dev
  python argo_test_suite.py --user mdearing --env dev --category anthropic
  python argo_test_suite.py --user mdearing --model gpt4o
  python argo_test_suite.py --user mdearing --list-models
  python argo_test_suite.py --user mdearing --skip-stream --skip-compat
        """,
    )
    parser.add_argument("--user", required=True, help="Your ANL domain username (e.g., mdearing)")
    parser.add_argument("--env", choices=["prod", "test", "dev"], default="dev",
                        help="Target environment (default: dev)")
    parser.add_argument("--category", choices=["openai", "google", "anthropic"],
                        help="Only test models from this vendor")
    parser.add_argument("--model", help="Only test a specific model ID (e.g., gpt4o)")
    parser.add_argument("--list-models", action="store_true", help="List all models and exit")
    parser.add_argument("--skip-stream", action="store_true", help="Skip streaming tests")
    parser.add_argument("--skip-compat", action="store_true",
                        help="Skip OpenAI-compatible and Anthropic Messages tests")
    parser.add_argument("--skip-tools", action="store_true", help="Skip tool/function calling tests")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show request/response details")
    return parser


def main() -> None:
    parser = buildParser()
    args = parser.parse_args()

    # Banner
    console.print()
    console.print(Panel.fit(
        "[bold white]Argo Gateway API[/]\n[dim]Comprehensive Test Suite with Validation[/]",
        border_style="bright_blue",
        padding=(1, 4),
    ))
    console.print()

    if args.list_models:
        chatFiltered = filterModels(ALL_MODELS, args.env, args.category, args.model)
        printModelTable(chatFiltered, f"Chat Models ({args.env})")
        console.print()
        printModelTable(EMBEDDING_MODELS, "Embedding Models")
        return

    # Config display
    configTree = Tree("[bold]Configuration[/]")
    configTree.add(f"User: [cyan]{args.user}[/]")
    configTree.add(f"Environment: [cyan]{args.env}[/]")
    configTree.add(f"Base URL: [cyan]{BASE_URLS[args.env]}[/]")
    if args.category:
        configTree.add(f"Category filter: [cyan]{args.category}[/]")
    if args.model:
        configTree.add(f"Model filter: [cyan]{args.model}[/]")
    flagsNode = configTree.add("Flags")
    flagsNode.add(f"Streaming: {'[red]skip[/]' if args.skip_stream else '[green]enabled[/]'}")
    flagsNode.add(f"Compat endpoints: {'[red]skip[/]' if args.skip_compat else '[green]enabled[/]'}")
    flagsNode.add(f"Tool calling: {'[red]skip[/]' if args.skip_tools else '[green]enabled[/]'}")
    flagsNode.add(f"Embeddings: {'[red]skip[/]' if args.skip_embed else '[green]enabled[/]'}")
    flagsNode.add(f"Verbose: {'[green]on[/]' if args.verbose else '[dim]off[/]'}")
    console.print(configTree)
    console.print()

    # Filter models
    chatModels = filterModels(ALL_MODELS, args.env, args.category, args.model)
    embedModels = EMBEDDING_MODELS if not args.skip_embed else []
    if args.model:
        embedModels = [m for m in embedModels if m.modelId == args.model]

    if not chatModels and not embedModels:
        console.print("[bold red]No models matched your filters.[/]")
        console.print("Use --list-models to see available models.")
        sys.exit(1)

    printModelTable(chatModels, f"Chat Models to Test ({len(chatModels)})")
    if embedModels:
        console.print()
        printModelTable(embedModels, f"Embedding Models to Test ({len(embedModels)})")

    console.print()
    console.print(f"[bold]Starting {len(chatModels)} chat model(s) + {len(embedModels)} embedding model(s) tests...[/]")
    console.print()

    runner = ArgoTestRunner(args.user, args.env, verbose=args.verbose)
    runAllTests(runner, chatModels, embedModels,
                skipStream=args.skip_stream,
                skipCompat=args.skip_compat,
                skipTools=args.skip_tools)

    printSummary(runner.suite)

    if runner.suite.failedCount > 0 or runner.suite.errorCount > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
