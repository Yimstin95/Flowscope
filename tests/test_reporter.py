"""Tests for the Reporter (stage 4).

The deterministic template is tested directly; the narrative LLM call uses a
fake in-process client (no key, no network).
"""

import numpy as np
import pandas as pd

from flowscope.critic import ClusterCritique, detect_anomalies
from flowscope.reporter import (
    DISCLAIMER,
    SUMMARY_TOOL,
    generate_narrative,
    generate_report,
    render_report,
)


def _annotated_summary():
    rng = np.random.default_rng(0)
    markers = ["G610-A (CD3)", "UV730-A (CD19)", "V510-A (AViD)"]
    rows = []
    for cl in range(15):
        row = {"cluster": cl, "n_events": int(rng.uniform(800, 1500))}
        for m in markers:
            row[m] = float(rng.uniform(3.5, 5.5))
        rows.append(row)
    outlier = {"cluster": 15, "n_events": 40, **{m: 0.2 for m in markers}}
    rows.append(outlier)
    df = pd.DataFrame(rows)
    df["frequency_pct"] = df["n_events"] / df["n_events"].sum() * 100
    df = df[["cluster", "n_events", "frequency_pct"] + markers]
    return detect_anomalies(df, seed=1)


class FakeToolUseBlock:
    type = "tool_use"
    name = "compose_report_summary"

    def __init__(self, payload):
        self.input = payload


class FakeResponse:
    stop_reason = "tool_use"

    def __init__(self, content):
        self.content = content


class FakeClient:
    def __init__(self, payload):
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return FakeResponse([FakeToolUseBlock(payload)])

        self.messages = _Messages()


META = {
    "sample_label": "FR-FCM-ZZEB/Sample1.fcs",
    "total_events": 428831,
    "analyzed_events": 20000,
    "n_clusters": 16,
    "generated_at": "2026-07-17 00:00 UTC",
}


def test_render_report_is_deterministic_and_has_disclaimer():
    summary = _annotated_summary()
    critiques = [ClusterCritique(cluster=15, frequency_pct=1.0, flagged_by="isolation-forest")]
    md = render_report(summary, critiques, META)
    # Disclaimer appears (header + footer), sample id present, table rendered.
    assert md.count(DISCLAIMER) >= 2
    assert "FR-FCM-ZZEB/Sample1.fcs" in md
    assert "## Cluster frequencies" in md
    assert "Cluster 15" in md
    # Deterministic: same inputs → identical output.
    assert md == render_report(summary, critiques, META)


def test_render_report_without_narrative_omits_summary_section():
    summary = _annotated_summary()
    md = render_report(summary, [], META, narrative=None)
    assert "## Summary (AI-proposed)" not in md
    assert DISCLAIMER in md  # still present


def test_render_report_includes_narrative_when_present():
    summary = _annotated_summary()
    narrative = {
        "overview": "Sample shows the expected PBMC lineage structure.",
        "notable_findings": ["Cluster 15 is a rare all-low debris-like population."],
        "recommended_followups": ["Manually gate cluster 15 on viability."],
    }
    md = render_report(summary, [], META, narrative=narrative)
    assert "## Summary (AI-proposed)" in md
    assert "expected PBMC lineage structure" in md
    assert "Manually gate cluster 15 on viability." in md


def test_generate_narrative_uses_fake_client_and_pins_model():
    summary = _annotated_summary()
    payload = {"overview": "ok", "notable_findings": [], "recommended_followups": []}
    client = FakeClient(payload)
    result = generate_narrative(summary, [], client=client)
    assert result == payload
    req = client.calls[0]
    assert req["model"] == "claude-sonnet-4-6"
    assert req["tool_choice"] == {"type": "tool", "name": "compose_report_summary"}
    assert req["tools"] == [SUMMARY_TOOL]


def test_generate_report_end_to_end_with_fake_client():
    summary = _annotated_summary()
    critiques = [
        ClusterCritique(
            cluster=15,
            frequency_pct=1.0,
            flagged_by="isolation-forest",
            interpretation={
                "verdict": "likely_artifact",
                "confidence": "high",
                "key_markers": ["V510-A (AViD)"],
                "reasoning": "All markers low; debris/dead-cell pattern.",
            },
        )
    ]
    payload = {
        "overview": "Structure looks normal aside from one debris cluster.",
        "notable_findings": ["Cluster 15 debris-like."],
        "recommended_followups": ["Verify viability gating."],
    }
    client = FakeClient(payload)
    md = generate_report(summary, critiques, META, client=client, with_narrative=True)
    assert "likely_artifact" in md
    assert "Structure looks normal" in md
    assert DISCLAIMER in md
