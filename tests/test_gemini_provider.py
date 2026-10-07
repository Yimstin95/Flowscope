"""GeminiClient tests: no network, no real key — urlopen is replaced by a fake."""

import io
import json

import pandas as pd
import pytest

from flowscope import llm_providers as lp
from flowscope.critic import interpret_anomaly

GOOD = {"verdict": "likely_biological", "confidence": "medium",
        "key_markers": ["CD3", "CD4"], "reasoning": "CD3+CD4+ T helper pattern"}
FAKE_KEY = "test-key-not-real"


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok_body(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}, "finishReason": "STOP"}]}


def _summary():
    return pd.DataFrame({"cluster": [0, 1], "n_events": [10, 20], "frequency_pct": [33.0, 67.0],
                         "CD3": [5.0, 1.0], "CD4": [4.0, 0.5]})


def _install(monkeypatch, responses, seen):
    """responses: list of (status, body) returned in order."""
    queue = list(responses)

    def fake(req, timeout=None):
        seen.append(req)
        status, body = queue.pop(0)
        if status != 200:
            raise lp.urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(json.dumps(body).encode()))
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(lp.urllib.request, "urlopen", fake)
    monkeypatch.setattr(lp.time, "sleep", lambda s: None)


def test_request_shape_and_key_never_in_url(monkeypatch):
    seen = []
    _install(monkeypatch, [(200, _ok_body(GOOD))], seen)
    out = interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="gemini-x")
    assert out["verdict"] == "likely_biological"
    req = seen[0]
    assert FAKE_KEY not in req.full_url                      # key goes in a header, not the URL
    assert req.get_header("X-goog-api-key") == FAKE_KEY
    assert req.full_url.endswith("/models/gemini-x:generateContent")
    body = json.loads(req.data)
    cfg = body["generationConfig"]
    assert cfg["temperature"] == 0 and cfg["responseMimeType"] == "application/json"
    assert cfg["responseJsonSchema"]["properties"]["verdict"]["enum"][0] == "likely_artifact"
    assert "systemInstruction" in body


def test_falls_back_to_openapi_schema_on_400(monkeypatch):
    seen = []
    _install(monkeypatch, [(400, {"error": "unknown field"}), (200, _ok_body(GOOD))], seen)
    interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="m")
    second = json.loads(seen[1].data)["generationConfig"]
    assert "responseJsonSchema" not in second and second["responseSchema"]["type"] == "OBJECT"


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    seen = []
    _install(monkeypatch, [(429, {"error": {"message": "per minute"}}), (200, _ok_body(GOOD))], seen)
    interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="m")
    assert len(seen) == 2


def test_daily_quota_stops_cleanly(monkeypatch):
    _install(monkeypatch, [(429, {"error": {"message": "Quota exceeded: requests per day"}})], [])
    with pytest.raises(lp.QuotaExhausted):
        interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="m")


def test_invalid_output_is_a_format_error(monkeypatch):
    bad = {"candidates": [{"content": {"parts": [{"text": "{\"verdict\": \"maybe\"}"}]}}]}
    _install(monkeypatch, [(200, bad)], [])
    with pytest.raises(lp.LLMFormatError):
        interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="m")


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=".env"):
        lp.GeminiClient()


def test_server_errors_are_retried_not_blamed_on_the_model(monkeypatch):
    seen = []
    _install(monkeypatch, [(503, {"error": "high demand"}), (500, {"error": "internal"}), (200, _ok_body(GOOD))], seen)
    out = interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="m")
    assert out["verdict"] == "likely_biological" and len(seen) == 3


def test_persistent_server_error_raises_provider_unavailable(monkeypatch):
    _install(monkeypatch, [(503, {"error": "x"})] * 5, [])
    with pytest.raises(lp.ProviderUnavailable):
        interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0, max_retries=4), model="m")


def test_read_timeout_is_treated_as_server_trouble(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return _Resp(json.dumps(_ok_body(GOOD)).encode())

    monkeypatch.setattr(lp.urllib.request, "urlopen", fake)
    monkeypatch.setattr(lp.time, "sleep", lambda s: None)
    out = interpret_anomaly(_summary(), 0, client=lp.GeminiClient(api_key=FAKE_KEY, min_interval_s=0), model="m")
    assert out["verdict"] == "likely_biological" and len(calls) == 2
