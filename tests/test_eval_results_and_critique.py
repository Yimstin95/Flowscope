"""Display-table loader + critique() model/prompt pass-through."""

import json
from types import SimpleNamespace

import pandas as pd

from flowscope.critic import INTERPRET_SYSTEM_PROMPT, INTERPRET_SYSTEM_PROMPT_V2, critique
from flowscope.eval_results import load_eval_results


def test_load_eval_results(tmp_path):
    raw = {
        "models": {
            "gemini__gemma-4-31b-it": {"agreement": 0.97, "agreement_ci": [0.92, 1.0], "dangerous_miss": 0.04,
                                       "lost_population": 0.03, "abstention_rate": 0.0, "median_latency_s": 42,
                                       "n_scored": 63},
            "gemma3_4b": {"agreement": 0.0, "agreement_ci": [0.0, 0.0], "dangerous_miss": 0.0,
                          "lost_population": 0.0, "abstention_rate": 1.0, "median_latency_s": 16, "n_scored": 63},
        },
        "by_true_class": {"gemini__gemma-4-31b-it": {"CD4_T": {"sum": 5, "count": 5}},
                          "gemma3_4b": {"CD4_T": {"sum": 0, "count": 5}}},
        "detector": {"artifact_recall": 0.38},
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(raw))
    out = load_eval_results(str(p))
    m = out["models"]
    assert list(m["model"]) == ["gemma-4-31b-it", "gemma3:4b"]          # best first, local name restored
    assert list(m["runs on"]) == ["Gemini API (free tier)", "local (Ollama)"]
    assert m.loc[0, "95% CI"] == "92%–100%"
    assert out["by_type"].loc["CD4_T", "gemma-4-31b-it"] == "5/5"
    assert out["detector"]["artifact_recall"] == 0.38


def test_missing_results_file_returns_none(tmp_path):
    assert load_eval_results(str(tmp_path / "nope.json")) is None


class _Capture:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        block = SimpleNamespace(type="tool_use", name="record_cluster_interpretation",
                                input={"verdict": "likely_artifact", "confidence": "low",
                                       "key_markers": [], "reasoning": "x"})
        return SimpleNamespace(content=[block], stop_reason="tool_use")


def _summary():
    rng = [1.0] * 9 + [9.0]   # cluster 9 is an obvious frequency outlier
    return pd.DataFrame({"cluster": range(10), "n_events": [100] * 10, "frequency_pct": rng,
                         "CD3": [1.0] * 9 + [8.0], "CD19": [1.0] * 10})


def test_critique_passes_model_and_prompt_through():
    cap = _Capture()
    res = critique(_summary(), client=cap, model="gemma-4-31b-it", system_prompt=INTERPRET_SYSTEM_PROMPT_V2)
    assert res.critiques and cap.calls
    assert all(c["model"] == "gemma-4-31b-it" and c["system"] == INTERPRET_SYSTEM_PROMPT_V2 for c in cap.calls)


def test_critique_defaults_to_original_prompt():
    cap = _Capture()
    critique(_summary(), client=cap)
    assert all(c["system"] == INTERPRET_SYSTEM_PROMPT for c in cap.calls)


class _FlakyClient(_Capture):
    """Fails on the first call only (like a provider outage on one cluster)."""

    def create(self, **kw):
        if not self.calls:
            self.calls.append(kw)
            raise RuntimeError("Gemini HTTP 500 after 5 tries")
        return super().create(**kw)


def _two_flag_summary():
    return pd.DataFrame({"cluster": range(10), "n_events": [100] * 10,
                         "frequency_pct": [1.0] * 8 + [9.0, 9.5],
                         "CD3": [1.0] * 8 + [8.0, 0.1], "CD19": [1.0] * 8 + [0.1, 8.0]})


def test_one_failing_cluster_does_not_lose_the_others():
    from flowscope.reporter import render_report

    res = critique(_two_flag_summary(), client=_FlakyClient(), on_error="skip")
    failed = [c for c in res.critiques if c.interpretation_error]
    ok = [c for c in res.critiques if c.interpretation]
    assert len(failed) == 1 and len(ok) >= 1
    md = render_report(res.annotated, res.critiques, {"sample_label": "t"})
    assert "AI service failed for this cluster" in md and "Review this cluster manually" in md


def test_default_still_raises_on_failure():
    import pytest

    with pytest.raises(RuntimeError):
        critique(_two_flag_summary(), client=_FlakyClient())
