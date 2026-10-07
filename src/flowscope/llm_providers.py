"""Free / local LLM back-ends that look like the Anthropic client to the rest of FlowScope.

`panel_interpreter.py` and `critic.py` only ever call
    client.messages.create(model=..., system=..., tools=[tool], tool_choice=..., messages=[...])
and read a `tool_use` block from `response.content`. `OllamaClient` implements exactly that
surface on top of a local Ollama server, so the existing stages (and their tests) are unchanged
and can run with no API key and no cost.

Structured output: instead of relying on model-specific tool calling (not every local model
supports it), the tool's JSON schema is sent through Ollama's `format` option, which constrains
decoding to that schema. The result is still validated here, because a constrained model can
satisfy the grammar and still return an out-of-vocabulary value.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

DEFAULT_HOST = "http://127.0.0.1:11434"


class LLMFormatError(ValueError):
    """The model answered, but not with valid structured output for the requested tool."""


def _validate(value, schema: dict, path: str = "$") -> None:
    """Minimal JSON-schema check: type, required keys, enum, array items. Raises LLMFormatError."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            raise LLMFormatError(f"{path}: expected object")
        for key in schema.get("required", []):
            if key not in value:
                raise LLMFormatError(f"{path}: missing required key {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}")
    elif t == "array":
        if not isinstance(value, list):
            raise LLMFormatError(f"{path}: expected array")
        for n, item in enumerate(value):
            _validate(item, schema.get("items", {}), f"{path}[{n}]")
    elif t == "string":
        if not isinstance(value, str):
            raise LLMFormatError(f"{path}: expected string")
    elif t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LLMFormatError(f"{path}: expected number")
    elif t == "boolean":
        if not isinstance(value, bool):
            raise LLMFormatError(f"{path}: expected boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMFormatError(f"{path}: {value!r} not in {schema['enum']}")


class _Messages:
    def __init__(self, owner: "OllamaClient"):
        self._owner = owner

    def create(self, *, model, messages, system=None, tools=None, tool_choice=None,
               max_tokens=1024, **_ignored):
        if not tools:
            raise LLMFormatError("OllamaClient needs exactly one tool to define the output schema")
        tool = tools[0]
        if tool_choice and tool_choice.get("name") not in (None, tool["name"]):
            tool = next((t for t in tools if t["name"] == tool_choice["name"]), tool)
        schema = tool["input_schema"]

        chat = []
        if system:
            chat.append({"role": "system", "content": system})
        for m in messages:
            chat.append({"role": m["role"], "content": m["content"]})

        payload = {
            "model": model,
            "messages": chat,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": max_tokens, "seed": 0},
        }
        t0 = time.perf_counter()
        body = self._owner._post("/api/chat", payload)
        self._owner.last_latency_s = time.perf_counter() - t0

        raw = (body.get("message") or {}).get("content", "")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMFormatError(f"model output is not valid JSON: {raw[:120]!r}") from e
        _validate(parsed, schema)
        block = SimpleNamespace(type="tool_use", name=tool["name"], input=parsed)
        return SimpleNamespace(content=[block], stop_reason=body.get("done_reason", "stop"))


class OllamaClient:
    """Drop-in for `anthropic.Anthropic()` in FlowScope's LLM stages (local, free, no key)."""

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 300.0):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.last_latency_s: float | None = None
        self.messages = _Messages(self)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (local host)
                return json.loads(r.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` "
                f"(or `brew services start ollama`) and make sure the model is pulled."
            ) from e

    def available_models(self) -> list[str]:
        req = urllib.request.Request(self.host + "/api/tags")
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
            return [m["name"] for m in json.loads(r.read()).get("models", [])]


# --- Gemini (Google AI Studio free tier) --------------------------------------------------------

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class QuotaExhausted(RuntimeError):
    """The free-tier quota is used up (e.g. requests per day). Finished items are already saved."""


class ProviderUnavailable(RuntimeError):
    """The provider's server failed (HTTP 5xx) even after retries. This says nothing about the model's
    answer quality, so callers should NOT record it as a model failure; retry the item later."""


def _to_gemini_schema(schema: dict) -> dict:
    """JSON-schema subset -> Gemini's OpenAPI-style `responseSchema` (upper-case TYPE enum).
    Only used as a fallback if the endpoint rejects `responseJsonSchema`."""
    out = {}
    for k, v in schema.items():
        if k == "type":
            out["type"] = v.upper()
        elif k == "properties":
            out["properties"] = {p: _to_gemini_schema(s) for p, s in v.items()}
        elif k == "items":
            out["items"] = _to_gemini_schema(v)
        elif k in ("enum", "required", "description"):
            out[k] = v
    return out


class _GeminiMessages:
    def __init__(self, owner: "GeminiClient"):
        self._owner = owner

    def create(self, *, model, messages, system=None, tools=None, tool_choice=None,
               max_tokens=1024, **_ignored):
        if not tools:
            raise LLMFormatError("GeminiClient needs exactly one tool to define the output schema")
        tool = tools[0]
        if tool_choice and tool_choice.get("name") not in (None, tool["name"]):
            tool = next((t for t in tools if t["name"] == tool_choice["name"]), tool)
        schema = tool["input_schema"]

        body = {
            "contents": [{"role": "user" if m["role"] == "user" else "model",
                          "parts": [{"text": m["content"]}]} for m in messages],
            # Thinking models spend output tokens on reasoning first, so leave headroom.
            "generationConfig": {"temperature": 0, "maxOutputTokens": max(max_tokens, 4096),
                                 "responseMimeType": "application/json"},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        t0 = time.perf_counter()
        resp = None
        for schema_field, schema_value in (("responseJsonSchema", schema),
                                           ("responseSchema", _to_gemini_schema(schema))):
            body["generationConfig"].pop("responseJsonSchema", None)
            body["generationConfig"].pop("responseSchema", None)
            body["generationConfig"][schema_field] = schema_value
            status, resp = self._owner._post(f"/models/{model}:generateContent", body)
            if status != 400:
                break
        self._owner.last_latency_s = time.perf_counter() - t0
        if status != 200:
            raise LLMFormatError(f"Gemini HTTP {status}: {str(resp)[:160]}")

        try:
            parts = resp["candidates"][0]["content"]["parts"]
            raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError) as e:
            raise LLMFormatError(f"no text in Gemini response: {str(resp)[:160]}") from e
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMFormatError(f"model output is not valid JSON: {raw[:120]!r}") from e
        _validate(parsed, schema)
        block = SimpleNamespace(type="tool_use", name=tool["name"], input=parsed)
        return SimpleNamespace(content=[block],
                               stop_reason=resp["candidates"][0].get("finishReason", "STOP"))


class GeminiClient:
    """Drop-in for `anthropic.Anthropic()` backed by the Gemini API free tier.

    The key is read from GEMINI_API_KEY and sent in the `x-goog-api-key` header (never in the URL,
    so it cannot leak into logs or error messages). Requests are spaced out (`min_interval_s`) to stay
    under free-tier per-minute limits; 429s are retried with back-off, and a daily-quota 429 raises
    QuotaExhausted so a resumable run can stop cleanly.
    """

    def __init__(self, api_key: str | None = None, min_interval_s: float = 7.0,
                 timeout: float = 180.0, max_retries: int = 4):
        import os
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Put it in the project's .env file.")
        self.min_interval_s = min_interval_s
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_latency_s: float | None = None
        self._last_call = 0.0
        self.messages = _GeminiMessages(self)

    def _request(self, method: str, path: str, payload: dict | None = None):
        req = urllib.request.Request(
            GEMINI_BASE + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except Exception:  # noqa: BLE001
                return e.code, {"error": str(e)}
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            # No answer in time / connection dropped: a server-side problem, treated like a 504
            # so it is retried and, if it persists, skipped (never scored as the model's mistake).
            return 504, {"error": f"{type(e).__name__}: {e}"}

    def _post(self, path: str, payload: dict):
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            status, body = self._request("POST", path, payload)
            if status in (500, 502, 503, 504):          # server-side trouble, not the model's answer
                if attempt == self.max_retries:
                    raise ProviderUnavailable(f"Gemini HTTP {status} after {attempt + 1} tries")
                time.sleep(min(60, 8 * (attempt + 1)))
                continue
            if status != 429:
                return status, body
            text = json.dumps(body).lower()
            if "per day" in text or "perday" in text or "daily" in text:
                raise QuotaExhausted("Gemini free-tier daily quota reached; re-run tomorrow to resume.")
            time.sleep(min(60, 10 * (attempt + 1)))
        raise QuotaExhausted("Gemini kept returning 429 (rate limit); stopped. Re-run later to resume.")

    def available_models(self) -> list[str]:
        status, body = self._request("GET", "/models?pageSize=200")
        if status != 200:
            raise ConnectionError(f"Gemini model list failed: HTTP {status} {str(body)[:160]}")
        return sorted(m["name"].split("/", 1)[-1] for m in body.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))
