"""Tests for the Panel Interpreter (stage 1).

These use a fake in-process client so they run with no API key and no network.
The prompt itself is exercised against the live model by
scripts/validate_panel_interpreter.py, which needs ANTHROPIC_API_KEY.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flowscope.executor import load_fcs
from flowscope.panel_interpreter import (
    GATING_TOOL,
    PanelParameter,
    build_user_message,
    interpret_panel,
    panel_from_sample,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SYNTHETIC_FCS = os.path.join(DATA_DIR, "synthetic_pbmc_demo.fcs")
SYNTHETIC_SPILL = os.path.join(DATA_DIR, "synthetic_pbmc_demo.spillover.csv")
REAL_SAMPLE = os.path.join(DATA_DIR, "FR-FCM-ZZEB", "Sample1.fcs")


class FakeToolUseBlock:
    type = "tool_use"
    name = "propose_gating_strategy"

    def __init__(self, steps):
        self.input = {"steps": steps}


class FakeResponse:
    stop_reason = "tool_use"

    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, response, capture):
        self._response = response
        self._capture = capture

    def create(self, **kwargs):
        self._capture.update(kwargs)
        return self._response


class FakeClient:
    """Records the request kwargs and returns a canned response."""

    def __init__(self, response):
        self.last_request = {}
        self.messages = FakeMessages(response, self.last_request)


@pytest.fixture(scope="module", autouse=True)
def ensure_synthetic_data():
    if not os.path.exists(SYNTHETIC_FCS):
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_synthetic_fcs.py")
        subprocess.run([sys.executable, script], check=True)


def test_panel_from_synthetic_sample_tags_scatter_and_fluorescence():
    sample = load_fcs(SYNTHETIC_FCS, spillover_path=SYNTHETIC_SPILL)
    panel = panel_from_sample(sample)
    by_channel = {p.channel: p for p in panel}
    assert by_channel["FSC-A"].kind == "scatter/time"
    assert by_channel["Time"].kind == "scatter/time"
    # Synthetic file has no PnS markers, so fluorescence channels have marker=None
    assert by_channel["CD3-FITC"].kind == "fluorescence"


def test_build_user_message_lists_markers():
    panel = [
        PanelParameter("FSC-A"),
        PanelParameter("G610-A", "CD3"),
        PanelParameter("UV730-A", "CD19"),
    ]
    msg = build_user_message(panel, context="Healthy donor PBMC")
    assert "CD3 (G610-A)" in msg
    assert "CD19 (UV730-A)" in msg
    assert "Healthy donor PBMC" in msg


def test_interpret_panel_parses_and_sorts_steps():
    # Deliberately out of order to confirm sorting by `step`.
    steps = [
        {"step": 2, "marker_pair": ["CD3", "CD19"], "gate_type": "quadrant", "rationale": "lineage split"},
        {"step": 1, "marker_pair": ["FSC-A", "SSC-A"], "gate_type": "polygon", "rationale": "cells vs debris"},
    ]
    client = FakeClient(FakeResponse([FakeToolUseBlock(steps)]))
    panel = [PanelParameter("FSC-A"), PanelParameter("SSC-A"), PanelParameter("G610-A", "CD3")]

    result = interpret_panel(panel, client=client)

    assert [s["step"] for s in result] == [1, 2]
    # The request forced the tool and pinned the model.
    req = client.last_request
    assert req["model"] == "claude-sonnet-4-6"
    assert req["tool_choice"] == {"type": "tool", "name": "propose_gating_strategy"}
    assert req["tools"] == [GATING_TOOL]


def test_interpret_panel_raises_without_tool_call():
    text_block = SimpleNamespace(type="text", text="no tool here")
    client = FakeClient(FakeResponse([text_block]))
    with pytest.raises(ValueError):
        interpret_panel([PanelParameter("FSC-A")], client=client)


@pytest.mark.skipif(
    not os.path.exists(REAL_SAMPLE),
    reason="Real FR-FCM-ZZEB Sample1.fcs not present",
)
def test_panel_from_real_sample_includes_omip024_markers():
    sample = load_fcs(REAL_SAMPLE)
    panel = panel_from_sample(sample)
    markers = {p.marker for p in panel if p.marker}
    for expected in ("CD3", "CD4", "CD8", "CD19", "CD56"):
        assert expected in markers
