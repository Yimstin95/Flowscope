"""Tests for the Critic / anomaly interpreter (stage 3).

Anomaly detection is pure Python and tested against a synthetic cluster summary
with one deliberately planted outlier. The LLM interpretation is tested with a
fake in-process client (no key, no network). The prompt is exercised live by
scripts/validate_critic.py.
"""

import numpy as np
import pandas as pd
import pytest

from flowscope.critic import (
    INTERPRET_TOOL,
    ClusterCritique,
    cluster_marker_profile,
    critique,
    detect_anomalies,
    interpret_anomaly,
)


def _summary_with_outlier():
    """15 'normal' clusters with moderate frequency and marker medians, plus one
    debris-like outlier: tiny frequency and everything low (a classic dead-cell
    /debris signature)."""
    rng = np.random.default_rng(0)
    rows = []
    markers = ["B710-A (CD8)", "V610-A (CD56)", "G610-A (CD3)", "UV730-A (CD19)", "V510-A (AViD)"]
    for cl in range(15):
        row = {"cluster": cl, "n_events": int(rng.uniform(800, 1500))}
        for m in markers:
            row[m] = float(rng.uniform(3.5, 5.5))
        rows.append(row)
    # Planted outlier: cluster 15, very rare, all markers low.
    outlier = {"cluster": 15, "n_events": 40}
    for m in markers:
        outlier[m] = 0.2
    rows.append(outlier)

    df = pd.DataFrame(rows)
    df["frequency_pct"] = df["n_events"] / df["n_events"].sum() * 100
    # reorder columns to the canonical meta-first layout
    cols = ["cluster", "n_events", "frequency_pct"] + markers
    return df[cols]


class FakeToolUseBlock:
    type = "tool_use"
    name = "record_cluster_interpretation"

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


def test_detect_anomalies_flags_planted_outlier():
    summary = _summary_with_outlier()
    annotated = detect_anomalies(summary, seed=1)
    flagged = set(annotated.loc[annotated["anomaly"], "cluster"])
    assert 15 in flagged
    # The outlier should be caught by the isolation forest (unusual marker profile).
    out_row = annotated.loc[annotated["cluster"] == 15].iloc[0]
    assert out_row["iforest_flag"]
    assert out_row["flagged_by"]  # non-empty reason string


def test_detect_anomalies_adds_expected_columns():
    annotated = detect_anomalies(_summary_with_outlier())
    for col in ("freq_zscore", "iforest_flag", "iforest_score", "anomaly", "flagged_by"):
        assert col in annotated.columns


def test_cluster_marker_profile_ranks_by_deviation():
    summary = _summary_with_outlier()
    profile = cluster_marker_profile(summary, 15, top_n=5)
    # Outlier is low on everything, so top deviations are all "low".
    assert profile[0].direction == "low"
    # Deviations sorted by magnitude, descending.
    mags = [abs(d.deviation) for d in profile]
    assert mags == sorted(mags, reverse=True)


def test_interpret_anomaly_parses_tool_call_and_pins_model():
    summary = _summary_with_outlier()
    payload = {
        "verdict": "likely_artifact",
        "confidence": "high",
        "key_markers": ["V510-A (AViD)"],
        "reasoning": "All markers low with tiny frequency — dead-cell/debris pattern.",
    }
    client = FakeClient(payload)
    result = interpret_anomaly(summary, 15, client=client)
    assert result["verdict"] == "likely_artifact"
    req = client.calls[0]
    assert req["model"] == "claude-sonnet-4-6"
    assert req["tool_choice"] == {"type": "tool", "name": "record_cluster_interpretation"}
    assert req["tools"] == [INTERPRET_TOOL]
    # The prompt actually described the flagged cluster's markers.
    assert "AViD" in req["messages"][0]["content"]


def test_critique_detection_only_without_client():
    summary = _summary_with_outlier()
    result = critique(summary, interpret=False)
    assert result.annotated["anomaly"].any()
    assert all(isinstance(c, ClusterCritique) for c in result.critiques)
    # No interpretation was attempted.
    assert all(c.interpretation is None for c in result.critiques)
    # Cluster 15 is among the critiques.
    assert 15 in {c.cluster for c in result.critiques}


def test_critique_with_fake_client_interprets_each_flagged_cluster():
    summary = _summary_with_outlier()
    payload = {
        "verdict": "needs_confirmation",
        "confidence": "low",
        "key_markers": [],
        "reasoning": "Ambiguous.",
    }
    client = FakeClient(payload)
    result = critique(summary, client=client, interpret=True)
    assert result.critiques
    for c in result.critiques:
        assert c.interpretation == payload
    # One LLM call per flagged cluster.
    assert len(client.calls) == len(result.critiques)
