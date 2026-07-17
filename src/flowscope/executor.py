"""Stage 2: Executor — preprocessing.

Pure Python, no LLM calls. This module handles FCS parsing, compensation,
and the arcsinh transform (build step 2). The FlowSOM-style clustering, UMAP
embedding, and per-cluster summary statistics (build step 3) live in the
sibling module `clustering.py` — kept separate only for readability; both are
part of the deterministic, no-LLM Executor stage.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from flowio import FlowData


@dataclass
class FCSSample:
    """A parsed FCS file: event data plus the channel/parameter metadata
    needed to interpret it (compensation, transform, gating)."""

    events: pd.DataFrame
    channel_names: list[str]
    metadata: dict = field(default_factory=dict)
    spillover: pd.DataFrame | None = None
    markers: dict[str, str] = field(default_factory=dict)

    def fluor_channels(self) -> list[str]:
        """Fluorescence channels, taken as the ones named in the spillover
        matrix (falls back to empty if no spillover is available)."""
        if self.spillover is None:
            return []
        return list(self.spillover.columns)

    def label(self, channel: str) -> str:
        """Human-readable label for a channel: 'G610-A (CD3)' when a marker
        (PnS) is known, otherwise just the detector name."""
        marker = self.markers.get(channel)
        return f"{channel} ({marker})" if marker else channel


def load_fcs(path: str, spillover_path: str | None = None) -> FCSSample:
    """Parse an FCS file into an FCSSample.

    If the FCS file has an embedded $SPILLOVER/$SPILL keyword, that matrix is
    used. Otherwise, pass `spillover_path` to a CSV with a header row and
    first column both naming the fluorescence channels (row i, column j =
    fraction of channel i's true signal detected in channel j).
    """
    flow_data = FlowData(path)
    channel_names = [
        flow_data.channels[i]["pnn"] for i in range(1, flow_data.channel_count + 1)
    ]
    markers = {
        flow_data.channels[i]["pnn"]: flow_data.channels[i].get("pns", "").strip()
        for i in range(1, flow_data.channel_count + 1)
        if flow_data.channels[i].get("pns", "").strip()
    }
    events = np.reshape(flow_data.events, (-1, flow_data.channel_count))
    df = pd.DataFrame(events, columns=channel_names)

    spillover = _extract_embedded_spillover(flow_data, channel_names)
    if spillover is None and spillover_path is not None:
        spillover = _load_spillover_csv(spillover_path)

    return FCSSample(
        events=df,
        channel_names=channel_names,
        metadata=dict(flow_data.text),
        spillover=spillover,
        markers=markers,
    )


def _extract_embedded_spillover(flow_data: FlowData, channel_names: list[str]) -> pd.DataFrame | None:
    raw = flow_data.text.get("spillover") or flow_data.text.get("spill")
    if not raw:
        return None
    fields = raw.split(",")
    n = int(fields[0])
    names = fields[1 : 1 + n]
    values = np.array(fields[1 + n :], dtype=float).reshape(n, n)
    return pd.DataFrame(values, index=names, columns=names)


def _load_spillover_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def apply_compensation(sample: FCSSample) -> FCSSample:
    """Undo detector spillover: compensated = raw @ inverse(spillover)^T.

    Channels not present in the spillover matrix (e.g. FSC-A, SSC-A, Time)
    are passed through unchanged.
    """
    if sample.spillover is None:
        raise ValueError("No spillover matrix available (embedded or supplied) — cannot compensate")

    spill_channels = list(sample.spillover.columns)
    missing = set(spill_channels) - set(sample.events.columns)
    if missing:
        raise ValueError(f"Spillover matrix references channels not in FCS data: {missing}")

    spill_matrix = sample.spillover.loc[spill_channels, spill_channels].to_numpy()
    inverse = np.linalg.inv(spill_matrix)

    compensated = sample.events.copy()
    raw_block = sample.events[spill_channels].to_numpy()
    compensated_block = raw_block @ inverse.T
    compensated[spill_channels] = compensated_block

    return FCSSample(
        events=compensated,
        channel_names=sample.channel_names,
        metadata=sample.metadata,
        spillover=sample.spillover,
        markers=sample.markers,
    )


def arcsinh_transform(
    sample: FCSSample, channels: list[str] | None = None, cofactor: float = 150.0
) -> FCSSample:
    """Apply arcsinh(x / cofactor) to fluorescence channels.

    `cofactor` controls where the transform switches from roughly linear
    (near zero) to roughly logarithmic (large x) behavior; ~150 is a common
    starting point for conventional fluorescence flow cytometry (as opposed
    to CyTOF, which typically uses ~5).
    """
    channels = channels or sample.channel_names
    transformed = sample.events.copy()
    transformed[channels] = np.arcsinh(sample.events[channels].to_numpy() / cofactor)
    return FCSSample(
        events=transformed,
        channel_names=sample.channel_names,
        metadata=sample.metadata,
        spillover=sample.spillover,
        markers=sample.markers,
    )
