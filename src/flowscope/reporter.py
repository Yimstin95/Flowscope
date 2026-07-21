"""Stage 4: Reporter (build step 6).

Turns the stage-2 clustering output and the stage-3 Critic interpretations into
a structured, LIMS-entry-style Markdown report.

Split of responsibilities:
  - A deterministic template assembles every factual section (sample/analysis
    parameters, the per-cluster frequency table, the flagged clusters with their
    interpretations) and ALWAYS appends the mandatory disclaimer. The model can
    never drop it because the template, not the model, writes it.
  - The Claude call (stage 4) writes only the analyst-facing narrative summary
    on top — overview, notable findings, recommended follow-ups.

Model pinned to claude-sonnet-4-6; narrative is forced via tool use; API key
read from ANTHROPIC_API_KEY. Set `with_narrative=False` (or omit the client) to
produce a fully-useful report without any LLM call.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from . import __version__ as FLOWSCOPE_VERSION
from .critic import ANNOTATION_COLS, META_COLS, ClusterCritique
from .panel_interpreter import MODEL

DISCLAIMER = (
    "AI-proposed; final review performed by the analyst. FlowScope is a "
    "research/education tool and this report is a reference signal, not a "
    "diagnostic determination."
)


def compute_file_checksum(path: str, algorithm: str = "sha256") -> str:
    """Hash of the raw FCS file bytes, recorded in the report so a later reader
    can confirm they're looking at results from the exact same input file, byte
    for byte (not just a file with the same name)."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Stage-4 LLM narrative ---------------------------------------------------

SUMMARY_TOOL = {
    "name": "compose_report_summary",
    "description": (
        "Compose the analyst-facing narrative summary for a flow cytometry "
        "clustering report, grounded only in the provided figures."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "overview": {
                "type": "string",
                "description": (
                    "2-4 plain-language sentences summarizing the sample's cluster "
                    "structure and overall picture."
                ),
            },
            "notable_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Each flagged cluster or noteworthy observation, one per item.",
            },
            "recommended_followups": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Concrete next steps for the analyst (e.g. manually gate a cluster on "
                    "viability, re-check compensation for a spillover-suspect cluster)."
                ),
            },
        },
        "required": ["overview", "notable_findings", "recommended_followups"],
    },
}

SUMMARY_SYSTEM_PROMPT = """You are drafting the narrative summary section of a flow \
cytometry analysis report for a trained analyst.

Ground every statement in the figures provided — cluster frequencies and the Critic's \
per-cluster interpretations. Do not invent populations, markers, or numbers. Phrase \
findings as observations and suggestions for the analyst to verify, never as clinical or \
diagnostic conclusions. Keep it concise and factual. Record the summary by calling \
compose_report_summary."""


def _marker_columns(summary: pd.DataFrame) -> list[str]:
    reserved = set(META_COLS) | set(ANNOTATION_COLS)
    return [c for c in summary.columns if c not in reserved]


def _summary_context(summary: pd.DataFrame, critiques: list[ClusterCritique]) -> str:
    lines = ["Cluster frequencies (%):"]
    for _, row in summary.sort_values("cluster").iterrows():
        flag = " [FLAGGED]" if row.get("anomaly") else ""
        lines.append(f"  cluster {row['cluster']}: {row['frequency_pct']:.2f}%{flag}")
    if critiques:
        lines.append("")
        lines.append("Critic interpretations of flagged clusters:")
        for c in critiques:
            interp = c.interpretation or {}
            lines.append(
                f"  cluster {c.cluster}: verdict={interp.get('verdict', 'n/a')} "
                f"(confidence {interp.get('confidence', 'n/a')}); "
                f"{interp.get('reasoning', 'no interpretation available')}"
            )
    return "\n".join(lines)


def generate_narrative(
    summary: pd.DataFrame,
    critiques: list[ClusterCritique],
    *,
    client=None,
    model: str = MODEL,
    max_tokens: int = 1500,
) -> dict:
    """Stage-4 Claude call. Returns
    {overview, notable_findings, recommended_followups}."""
    if client is None:
        import anthropic  # noqa: PLC0415 — lazy so importing this module needs no key

        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SUMMARY_SYSTEM_PROMPT,
        tools=[SUMMARY_TOOL],
        tool_choice={"type": "tool", "name": "compose_report_summary"},
        messages=[{"role": "user", "content": _summary_context(summary, critiques)}],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "compose_report_summary":
            return dict(block.input)

    raise ValueError(
        "Model did not return a compose_report_summary tool call; "
        f"stop_reason={getattr(response, 'stop_reason', None)}"
    )


# --- Deterministic report template ------------------------------------------


def render_report(
    summary: pd.DataFrame,
    critiques: list[ClusterCritique],
    meta: dict,
    narrative: dict | None = None,
) -> str:
    """Assemble the Markdown report. Deterministic — no LLM call here. The
    mandatory disclaimer is always appended."""
    generated_at = meta.get("generated_at") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    markers = _marker_columns(summary)
    n_flagged = int(summary["anomaly"].sum()) if "anomaly" in summary.columns else 0

    lines = [
        "# FlowScope Analysis Report",
        "",
        f"> **{DISCLAIMER}**",
        "",
        "## Entry",
        "",
        f"- **Sample:** {meta.get('sample_label', 'n/a')}",
        f"- **Generated:** {generated_at}",
        f"- **Total events:** {meta.get('total_events', 'n/a')}",
        f"- **Events analyzed (subsample):** {meta.get('analyzed_events', 'n/a')}",
        f"- **Metaclusters:** {meta.get('n_clusters', len(summary))}",
        f"- **Clusters flagged anomalous:** {n_flagged}",
        f"- **Panel markers:** {len(markers)}",
        "",
    ]

    # Analysis Provenance: deterministic, no LLM involved. Exists so a report
    # can be reproduced bit-for-bit later, or defended in review, without
    # anyone having to remember (or re-derive) what settings produced it.
    lines += [
        "## Analysis Provenance",
        "",
        f"- **Input file:** {meta.get('file_path', 'n/a')}",
        f"- **File SHA-256:** `{meta.get('file_sha256', 'n/a')}`",
        f"- **Compensation source:** {meta.get('spillover_source', 'n/a')}",
        f"- **arcsinh cofactor:** {meta.get('cofactor', 'n/a')}",
        f"- **Events subsampled:** {meta.get('subsample_n', 'n/a')}",
        f"- **Random seed:** {meta.get('seed', 'n/a')}",
        f"- **FlowScope version:** {FLOWSCOPE_VERSION}",
        f"- **AI model (if used):** {meta.get('ai_model') or 'none — detection-only report'}",
        "",
        "_Re-running with the same input file and these exact settings reproduces "
        "this result bit-for-bit (the pipeline is deterministic given a fixed seed)._",
        "",
    ]

    if narrative:
        lines += ["## Summary (AI-proposed)", "", narrative.get("overview", ""), ""]
        if narrative.get("notable_findings"):
            lines.append("**Notable findings:**")
            lines += [f"- {x}" for x in narrative["notable_findings"]]
            lines.append("")
        if narrative.get("recommended_followups"):
            lines.append("**Recommended follow-ups (for analyst review):**")
            lines += [f"- {x}" for x in narrative["recommended_followups"]]
            lines.append("")

    # Cluster frequency table
    lines += ["## Cluster frequencies", "", "| Cluster | Events | Frequency (%) | Flagged |", "|---|---|---|---|"]
    for _, row in summary.sort_values("cluster").iterrows():
        flagged = row.get("flagged_by", "") if "flagged_by" in summary.columns else ""
        lines.append(
            f"| {row['cluster']} | {int(row['n_events'])} | "
            f"{row['frequency_pct']:.2f} | {flagged or '—'} |"
        )
    lines.append("")

    # Flagged cluster detail
    if critiques:
        lines += ["## Flagged clusters — interpretation", ""]
        for c in critiques:
            interp = c.interpretation
            lines.append(f"### Cluster {c.cluster} ({c.frequency_pct:.2f}% of analyzed events)")
            lines.append(f"- **Flagged by:** {c.flagged_by or 'n/a'}")
            if interp:
                lines.append(
                    f"- **Verdict:** {interp.get('verdict', 'n/a')} "
                    f"(confidence: {interp.get('confidence', 'n/a')})"
                )
                km = interp.get("key_markers") or []
                lines.append(f"- **Key markers:** {', '.join(km) or '(none)'}")
                lines.append(f"- **Reasoning:** {interp.get('reasoning', '')}")
            else:
                lines.append("- **Interpretation:** not generated (detection only)")
            lines.append("")

    lines += ["---", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(lines)


def generate_report(
    summary: pd.DataFrame,
    critiques: list[ClusterCritique],
    meta: dict,
    *,
    client=None,
    with_narrative: bool = True,
) -> str:
    """End-to-end stage-4 report. Runs the narrative LLM call when
    `with_narrative` is True, then renders the full Markdown report.

    `summary` should be the anomaly-annotated summary from `critic.detect_anomalies`
    (or `critic.critique(...).annotated`); `critiques` are the per-cluster
    ClusterCritique objects.
    """
    narrative = None
    if with_narrative:
        narrative = generate_narrative(summary, critiques, client=client)
        meta = {**meta, "ai_model": MODEL}
    return render_report(summary, critiques, meta, narrative=narrative)
