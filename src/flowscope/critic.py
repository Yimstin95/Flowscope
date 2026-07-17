"""Stage 3: Critic / anomaly interpreter (build step 5).

Two halves, mirroring the project's core design principle:

  (a) `detect_anomalies` — pure Python, NO LLM. Flags clusters whose frequency
      or marker profile is statistically unusual, via a z-score on cluster
      frequency and an IsolationForest over the full per-cluster feature vector.

  (b) `interpret_anomaly` — the stage-3 Claude call, made ONLY for clusters the
      detector flagged. It argues "likely artifact vs. likely biological vs.
      needs confirmation" from the cluster's marker-expression pattern, and is
      prompted to prefer "needs confirmation" over a confident guess.

`critique` wires the two together. As with the Panel Interpreter, the model is
pinned to claude-sonnet-4-6, structured output is forced via tool use, and the
API key is read from ANTHROPIC_API_KEY (never hardcoded). Output is a reference
signal for analyst review, not a diagnostic determination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .panel_interpreter import MODEL  # reuse the pinned model id

META_COLS = ("cluster", "n_events", "frequency_pct")
# Columns added by detect_anomalies — not markers, must be excluded from any
# marker computation when an already-annotated summary is passed back in.
ANNOTATION_COLS = ("freq_zscore", "iforest_flag", "iforest_score", "anomaly", "flagged_by")


def _marker_columns(summary: pd.DataFrame) -> list[str]:
    reserved = set(META_COLS) | set(ANNOTATION_COLS)
    return [c for c in summary.columns if c not in reserved]


def detect_anomalies(
    summary: pd.DataFrame,
    z_thresh: float = 2.5,
    contamination: float | str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    """Flag anomalous clusters. Pure Python — no LLM call.

    Returns a copy of `summary` with added columns:
      - `freq_zscore`     : z-score of frequency_pct across clusters
      - `iforest_flag`    : True if IsolationForest labels the cluster an outlier
      - `iforest_score`   : IsolationForest decision score (lower = more anomalous)
      - `anomaly`         : True if flagged by either method
      - `flagged_by`      : comma-joined method names, or "" if not flagged

    Two complementary signals: the z-score catches clusters that are unusually
    rare or common; the IsolationForest catches clusters whose *marker profile*
    is unusual even when their frequency is ordinary.
    """
    out = summary.copy()
    markers = _marker_columns(summary)

    freq = out["frequency_pct"].to_numpy(dtype=float)
    std = freq.std()
    out["freq_zscore"] = (freq - freq.mean()) / (std if std > 0 else 1.0)

    features = out[["frequency_pct", *markers]].to_numpy(dtype=float)
    scaled = _standardize(features)
    iforest = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed
    )
    labels = iforest.fit_predict(scaled)  # -1 outlier, 1 inlier
    out["iforest_flag"] = labels == -1
    out["iforest_score"] = iforest.decision_function(scaled)

    zflag = out["freq_zscore"].abs() > z_thresh
    out["anomaly"] = zflag | out["iforest_flag"]

    def _reasons(row) -> str:
        r = []
        if abs(row["freq_zscore"]) > z_thresh:
            r.append("frequency-zscore")
        if row["iforest_flag"]:
            r.append("isolation-forest")
        return ", ".join(r)

    out["flagged_by"] = out.apply(_reasons, axis=1)
    return out


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


@dataclass
class MarkerDeviation:
    marker: str
    median: float
    direction: str  # "high" or "low"
    deviation: float  # signed, in cross-cluster std units


def cluster_marker_profile(
    summary: pd.DataFrame, cluster_id, top_n: int = 8
) -> list[MarkerDeviation]:
    """For one cluster, rank its markers by how far the cluster's median sits
    from the cross-cluster median, in cross-cluster std units. This is the
    evidence the LLM reasons over — 'CD3 high, CD19 low, viability high', etc.
    """
    markers = _marker_columns(summary)
    row = summary.loc[summary["cluster"] == cluster_id].iloc[0]
    overall_mean = summary[markers].mean()
    overall_std = summary[markers].std().replace(0, 1.0)

    devs = []
    for m in markers:
        signed = (row[m] - overall_mean[m]) / overall_std[m]
        devs.append(
            MarkerDeviation(
                marker=m,
                median=float(row[m]),
                direction="high" if signed >= 0 else "low",
                deviation=float(signed),
            )
        )
    devs.sort(key=lambda d: abs(d.deviation), reverse=True)
    return devs[:top_n]


# --- Stage-3 LLM interpretation ---------------------------------------------

INTERPRET_TOOL = {
    "name": "record_cluster_interpretation",
    "description": (
        "Record an interpretation of one anomalous flow cytometry cluster: whether "
        "it looks like a technical artifact or a biologically meaningful population, "
        "with the marker evidence behind that read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["likely_artifact", "likely_biological", "needs_confirmation"],
                "description": (
                    "Use 'needs_confirmation' whenever the marker evidence is ambiguous "
                    "or insufficient — do not force a confident call."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "key_markers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Markers that most drive the interpretation.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Concise argument from the marker pattern, naming the specific "
                    "artifact considered (e.g. compensation spillover, dead-cell/debris, "
                    "doublets) or the candidate biological population."
                ),
            },
        },
        "required": ["verdict", "confidence", "key_markers", "reasoning"],
    },
}

INTERPRET_SYSTEM_PROMPT = """You are helping a trained flow cytometry analyst triage a \
cluster that a statistical detector flagged as anomalous.

Reason ONLY from the marker-expression evidence provided. Weigh common technical \
artifacts against genuine biology:
- Compensation spillover: a cluster that is 'high' on markers whose fluorochromes sit on \
adjacent detectors, without a coherent lineage pattern.
- Dead cells / debris: high viability-dye signal, low or smeared scatter, diffuse low \
expression across lineage markers.
- Doublets: co-expression of markers that should be mutually exclusive across two cell types.
- Genuine rare population: a coherent, interpretable marker combination (e.g. a bona fide \
CD3+CD4+CD8+ or CD56bright subset).

If the evidence does not clearly favor artifact or biology, choose 'needs_confirmation' — \
that is the correct answer under uncertainty, not a failure. Never assert a clinical or \
diagnostic conclusion; this is a reference signal for the analyst to verify. Record your \
read by calling record_cluster_interpretation."""


def build_interpretation_message(
    cluster_id, frequency_pct: float, n_events: int, profile: list[MarkerDeviation]
) -> str:
    lines = [
        f"Anomalous cluster {cluster_id}: {frequency_pct:.2f}% of analyzed events "
        f"({n_events} events).",
        "",
        "Marker profile (deviation from the cross-cluster median, in std units; "
        "median is on the arcsinh-transformed scale):",
    ]
    for d in profile:
        lines.append(
            f"  - {d.marker}: {d.direction} (median {d.median:.2f}, "
            f"{d.deviation:+.1f} std)"
        )
    lines += [
        "",
        "Is this cluster more likely a technical artifact or a biologically meaningful "
        "population? Call record_cluster_interpretation with your read.",
    ]
    return "\n".join(lines)


def interpret_anomaly(
    summary: pd.DataFrame,
    cluster_id,
    *,
    client=None,
    model: str = MODEL,
    max_tokens: int = 1024,
    top_n: int = 8,
) -> dict:
    """Stage-3 Claude call for a single flagged cluster. Returns the
    {verdict, confidence, key_markers, reasoning} dict.

    `client` defaults to `anthropic.Anthropic()` (reads ANTHROPIC_API_KEY);
    injectable for tests.
    """
    if client is None:
        import anthropic  # noqa: PLC0415 — lazy so importing this module needs no key

        client = anthropic.Anthropic()

    row = summary.loc[summary["cluster"] == cluster_id].iloc[0]
    profile = cluster_marker_profile(summary, cluster_id, top_n=top_n)
    message = build_interpretation_message(
        cluster_id, float(row["frequency_pct"]), int(row["n_events"]), profile
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=INTERPRET_SYSTEM_PROMPT,
        tools=[INTERPRET_TOOL],
        tool_choice={"type": "tool", "name": "record_cluster_interpretation"},
        messages=[{"role": "user", "content": message}],
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "record_cluster_interpretation":
            return dict(block.input)

    raise ValueError(
        "Model did not return a record_cluster_interpretation tool call; "
        f"stop_reason={getattr(response, 'stop_reason', None)}"
    )


@dataclass
class ClusterCritique:
    cluster: object
    frequency_pct: float
    flagged_by: str
    interpretation: dict | None = None  # None if not interpreted (no client)


@dataclass
class CritiqueResult:
    annotated: pd.DataFrame  # summary + anomaly columns
    critiques: list[ClusterCritique] = field(default_factory=list)


def critique(
    summary: pd.DataFrame,
    *,
    client=None,
    interpret: bool = True,
    z_thresh: float = 2.5,
    contamination: float | str = "auto",
    seed: int = 42,
) -> CritiqueResult:
    """Detect anomalous clusters, then (if `interpret`) ask Claude to interpret
    each flagged one. Set `interpret=False` (or leave `client=None` in an
    environment with no key) to run detection only.
    """
    annotated = detect_anomalies(
        summary, z_thresh=z_thresh, contamination=contamination, seed=seed
    )
    flagged = annotated[annotated["anomaly"]]

    critiques: list[ClusterCritique] = []
    for _, row in flagged.iterrows():
        interp = None
        if interpret:
            interp = interpret_anomaly(annotated, row["cluster"], client=client)
        critiques.append(
            ClusterCritique(
                cluster=row["cluster"],
                frequency_pct=float(row["frequency_pct"]),
                flagged_by=row["flagged_by"],
                interpretation=interp,
            )
        )
    return CritiqueResult(annotated=annotated, critiques=critiques)
