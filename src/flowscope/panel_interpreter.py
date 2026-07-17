"""Stage 1: Panel Interpreter (build step 4).

The first of the three LLM stages. Given a flow cytometry panel (the markers
and detectors on the instrument), it asks Claude to propose a *hierarchical
gating strategy* — the ordered sequence of 2D gates an analyst would draw to
walk from raw events down to the populations of interest.

Design constraints (from the project spec):
  - Only stages 1/3/4 call the Claude API. This is stage 1.
  - Structured output is forced via tool use (`tool_choice`), not free-form
    text, so the caller always gets a valid list of
    {step, marker_pair, gate_type, rationale} dicts.
  - The API key comes from the ANTHROPIC_API_KEY environment variable
    (read by `anthropic.Anthropic()`); it is never hardcoded.
  - Model is pinned to claude-sonnet-4-6 per the project spec.

Scope note: the output is a *suggested* gating strategy for an analyst to
review and adjust, for research/education use — not a diagnostic determination.
The system prompt makes that framing explicit to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pinned per the project spec. claude-sonnet-5 is the current-generation Sonnet
# if this is ever revised, but the spec fixes 4.6.
MODEL = "claude-sonnet-4-6"

SCATTER_CHANNELS = {"FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "Time"}


@dataclass
class PanelParameter:
    """One measured parameter on the instrument.

    `channel` is the detector name (FCS $PnN, e.g. 'G610-A'); `marker` is the
    antibody/target it reports (FCS $PnS, e.g. 'CD3'), or None for scatter/time.
    """

    channel: str
    marker: str | None = None

    @property
    def kind(self) -> str:
        return "scatter/time" if self.channel in SCATTER_CHANNELS else "fluorescence"

    @property
    def label(self) -> str:
        return f"{self.marker} ({self.channel})" if self.marker else self.channel


def panel_from_sample(sample) -> list[PanelParameter]:
    """Build the panel description from a parsed FCSSample (see executor.py).

    Includes scatter/time channels (gating starts on scatter) as well as the
    fluorescence channels with their marker labels.
    """
    return [
        PanelParameter(channel=ch, marker=sample.markers.get(ch) or None)
        for ch in sample.channel_names
    ]


# --- Tool definition: forces the model to emit structured gating steps -------

GATING_TOOL = {
    "name": "propose_gating_strategy",
    "description": (
        "Record a proposed hierarchical gating strategy for the given flow "
        "cytometry panel. Each step is one 2D (or 1D) gate in the hierarchy, "
        "in the order an analyst would apply them from raw events down to the "
        "populations of interest."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "Ordered gating steps, from broadest (scatter) to most specific.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "integer",
                            "description": "1-based position of this gate in the hierarchy.",
                        },
                        "marker_pair": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "The parameter(s) plotted for this gate, using the marker "
                                "or channel names exactly as given in the panel, e.g. "
                                "['FSC-A','SSC-A'] or ['CD3','CD19']. One entry for a 1D "
                                "gate, two for a 2D gate."
                            ),
                        },
                        "gate_type": {
                            "type": "string",
                            "description": (
                                "Shape/kind of gate, e.g. 'polygon', 'rectangle', "
                                "'threshold', 'quadrant', 'ellipse'."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "Why this gate at this step: the population it isolates and/or "
                                "the artifact (debris, doublets, dead cells) it excludes."
                            ),
                        },
                    },
                    "required": ["step", "marker_pair", "gate_type", "rationale"],
                },
            }
        },
        "required": ["steps"],
    },
}

SYSTEM_PROMPT = """You are assisting a trained flow cytometry analyst by proposing a \
hierarchical gating strategy for a given antibody panel.

Follow standard immunophenotyping practice:
- Start from scatter (FSC/SSC) to separate cells from debris, then remove doublets \
using height-vs-area scatter if those parameters are present.
- If a viability dye is in the panel, add a live/dead gate before lineage gating.
- Gate major lineages (e.g. T cells on CD3, B cells on CD19, NK cells on CD56/CD16) \
before sub-populations (e.g. CD4 vs CD8 within CD3+ T cells).
- Only reference markers/channels that are actually in the provided panel, and write \
them exactly as given.
- Order steps parent-before-child: a gate may only rely on populations defined by \
earlier steps.

This is a SUGGESTED strategy for the analyst to review and adjust — a starting point, \
not a final or validated gating scheme, and not a diagnostic determination. Record your \
proposal by calling the propose_gating_strategy tool."""


def _format_panel(panel: list[PanelParameter]) -> str:
    lines = [f"- {p.label} [{p.kind}]" for p in panel]
    return "\n".join(lines)


def build_user_message(panel: list[PanelParameter], context: str | None = None) -> str:
    parts = [
        "Here is the panel (one parameter per line, marker then detector channel):",
        "",
        _format_panel(panel),
    ]
    if context:
        parts += ["", f"Additional context: {context}"]
    parts += [
        "",
        "Propose a hierarchical gating strategy for this panel by calling "
        "propose_gating_strategy.",
    ]
    return "\n".join(parts)


def interpret_panel(
    panel: list[PanelParameter],
    *,
    client=None,
    model: str = MODEL,
    max_tokens: int = 4096,
    context: str | None = None,
) -> list[dict]:
    """Ask Claude to propose a gating strategy for `panel`.

    Returns the list of {step, marker_pair, gate_type, rationale} dicts.

    `client` defaults to `anthropic.Anthropic()`, which reads ANTHROPIC_API_KEY
    from the environment. It's injectable so tests can pass a fake client
    without any network call or key.
    """
    if client is None:
        import anthropic  # noqa: PLC0415 — lazy so importing this module needs no key

        client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        tools=[GATING_TOOL],
        tool_choice={"type": "tool", "name": "propose_gating_strategy"},
        messages=[{"role": "user", "content": build_user_message(panel, context)}],
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "propose_gating_strategy":
            steps = block.input.get("steps", [])
            return sorted(steps, key=lambda s: s.get("step", 0))

    raise ValueError(
        "Model did not return a propose_gating_strategy tool call; "
        f"stop_reason={getattr(response, 'stop_reason', None)}"
    )
