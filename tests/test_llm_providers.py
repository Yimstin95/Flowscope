"""OllamaClient tests: no server, no network — urlopen is replaced by a fake."""

import io
import json

import pandas as pd
import pytest

from flowscope import llm_providers as lp
from flowscope.critic import INTERPRET_TOOL, interpret_anomaly


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, content, captured=None):
    def fake(req, timeout=None):
        if captured is not None:
            captured.append(json.loads(req.data))
        return _FakeResp(json.dumps({"message": {"content": content}, "done_reason": "stop"}).encode())

    monkeypatch.setattr(lp.urllib.request, "urlopen", fake)


GOOD = {"verdict": "likely_artifact", "confidence": "medium",
        "key_markers": ["viability"], "reasoning": "high viability dye, diffuse low lineage markers"}


def _summary():
    return pd.DataFrame({
        "cluster": [0, 1, 2], "n_events": [100, 200, 300], "frequency_pct": [10.0, 30.0, 60.0],
        "CD3": [1.0, 5.0, 4.0], "viability": [6.0, 0.5, 0.7],
    })


def test_returns_tool_use_block_and_sends_schema(monkeypatch):
    sent = []
    _patch_urlopen(monkeypatch, json.dumps(GOOD), sent)
    client = lp.OllamaClient()
    out = interpret_anomaly(_summary(), 0, client=client, model="llama3.1:8b")
    assert out["verdict"] == "likely_artifact"
    assert sent[0]["format"] == INTERPRET_TOOL["input_schema"]  # constrained decoding schema
    assert sent[0]["options"]["temperature"] == 0               # deterministic
    assert sent[0]["messages"][0]["role"] == "system"
    assert client.last_latency_s is not None


def test_invalid_json_is_a_format_error(monkeypatch):
    _patch_urlopen(monkeypatch, "I think this is an artifact.")
    with pytest.raises(lp.LLMFormatError):
        interpret_anomaly(_summary(), 0, client=lp.OllamaClient())


def test_out_of_vocabulary_value_is_rejected(monkeypatch):
    bad = dict(GOOD, verdict="probably_real")
    _patch_urlopen(monkeypatch, json.dumps(bad))
    with pytest.raises(lp.LLMFormatError, match="not in"):
        interpret_anomaly(_summary(), 0, client=lp.OllamaClient())


def test_missing_required_key_is_rejected(monkeypatch):
    bad = {k: v for k, v in GOOD.items() if k != "reasoning"}
    _patch_urlopen(monkeypatch, json.dumps(bad))
    with pytest.raises(lp.LLMFormatError, match="missing required"):
        interpret_anomaly(_summary(), 0, client=lp.OllamaClient())


def test_unreachable_server_gives_a_helpful_error(monkeypatch):
    def boom(req, timeout=None):
        raise lp.urllib.error.URLError("refused")

    monkeypatch.setattr(lp.urllib.request, "urlopen", boom)
    with pytest.raises(ConnectionError, match="ollama serve"):
        interpret_anomaly(_summary(), 0, client=lp.OllamaClient())


def test_custom_system_prompt_reaches_the_model(monkeypatch):
    from flowscope.critic import INTERPRET_SYSTEM_PROMPT, INTERPRET_SYSTEM_PROMPT_V2, INTERPRET_SYSTEM_PROMPT_V3, PROMPTS

    assert PROMPTS == {"v1": INTERPRET_SYSTEM_PROMPT, "v2": INTERPRET_SYSTEM_PROMPT_V2, "v3": INTERPRET_SYSTEM_PROMPT_V3}
    assert "T helper cells CD3+CD4+" in INTERPRET_SYSTEM_PROMPT_V3 and "T helper cells" not in INTERPRET_SYSTEM_PROMPT_V2
    assert "NKT" not in INTERPRET_SYSTEM_PROMPT_V3 and "plasmablast" not in INTERPRET_SYSTEM_PROMPT_V3.lower()  # no test-only classes
    sent = []
    _patch_urlopen(monkeypatch, json.dumps(GOOD), sent)
    interpret_anomaly(_summary(), 0, client=lp.OllamaClient(), model="m", system_prompt=INTERPRET_SYSTEM_PROMPT_V2)
    assert sent[0]["messages"][0]["content"] == INTERPRET_SYSTEM_PROMPT_V2
    sent.clear()
    interpret_anomaly(_summary(), 0, client=lp.OllamaClient(), model="m")   # default stays v1
    assert sent[0]["messages"][0]["content"] == INTERPRET_SYSTEM_PROMPT
