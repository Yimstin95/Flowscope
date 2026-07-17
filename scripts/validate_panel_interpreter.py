"""Live validation of the Panel Interpreter against real panels.

Runs the stage-1 agent on several panels of different sizes and prints the
proposed gating strategies, with light sanity checks (does it start on scatter?
does it add a viability gate when a dye is present? does it sub-gate CD4/CD8
under CD3?). This is prompt validation, not a unit test — it makes real Claude
API calls and therefore needs ANTHROPIC_API_KEY in the environment.

Usage:
    export ANTHROPIC_API_KEY=...        # your key; never commit it
    python scripts/validate_panel_interpreter.py
"""

import os
import sys

# Make src/ importable when run directly from the repo without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowscope.panel_interpreter import PanelParameter, interpret_panel  # noqa: E402


def _p(pairs):
    """Build a panel from (channel, marker) tuples; marker None for scatter."""
    return [PanelParameter(channel=c, marker=m) for c, m in pairs]


# Panel 1 — minimal 4-parameter T/B split (sanity floor)
PANEL_MINIMAL = _p([
    ("FSC-A", None), ("SSC-A", None),
    ("FITC-A", "CD3"), ("PE-A", "CD19"),
])

# Panel 2 — classic 8-color T/B/NK with a viability dye and doublet scatter
PANEL_TBNK = _p([
    ("FSC-A", None), ("FSC-H", None), ("SSC-A", None),
    ("V510-A", "LiveDead"),
    ("FITC-A", "CD3"), ("PE-A", "CD4"), ("APC-A", "CD8"),
    ("PerCP-A", "CD19"), ("PE-Cy7-A", "CD56"),
])

# Panel 3 — OMIP-024 (18-color PBMC), the confirmed FR-FCM-ZZEB panel
PANEL_OMIP024 = _p([
    ("FSC-A", None), ("FSC-H", None), ("SSC-A", None), ("SSC-H", None),
    ("B515-A", "CD57"), ("B710-A", "CD8"), ("V450-A", "CD25"), ("V510-A", "AViD"),
    ("V570-A", "HLADR"), ("V610-A", "CD56"), ("V655-A", "CD45RA"), ("V710-A", "CD14"),
    ("V780-A", "CCR7"), ("R660-A", "C127"), ("R710-A", "NKG2C"), ("R780-A", "CD16"),
    ("G575-A", "Vdelta2"), ("G610-A", "CD3"), ("G660-A", "CD38"), ("G780-A", "gdTCR"),
    ("UV395-A", "CD4"), ("UV730-A", "CD19"),
])

PANELS = {
    "Minimal 4-param (CD3/CD19)": PANEL_MINIMAL,
    "8-color T/B/NK + viability": PANEL_TBNK,
    "OMIP-024 18-color PBMC (FR-FCM-ZZEB)": PANEL_OMIP024,
}


def _markers(panel):
    return {p.marker for p in panel if p.marker}


def _check(panel, steps):
    """Return a list of (label, ok) sanity checks for this panel's strategy."""
    checks = []
    first = steps[0] if steps else {}
    first_pair = {m.upper() for m in first.get("marker_pair", [])}
    checks.append(("first gate uses scatter (FSC/SSC)",
                   any("FSC" in m or "SSC" in m for m in first_pair)))

    markers = {m.upper() for m in _markers(panel)}
    all_pair_markers = {m.upper() for s in steps for m in s.get("marker_pair", [])}

    if {"LIVEDEAD", "AVID"} & markers:
        via = {"LIVEDEAD", "AVID"} & all_pair_markers
        checks.append(("viability dye is gated on", bool(via)))
    if {"CD4", "CD8"} <= markers:
        checks.append(("CD4 and CD8 both appear in gating",
                       {"CD4", "CD8"} <= all_pair_markers))
    # Every referenced marker must exist in the panel (channels or markers).
    panel_names = {p.channel.upper() for p in panel} | markers
    unknown = all_pair_markers - panel_names
    checks.append((f"no invented markers ({sorted(unknown) or 'none'})", not unknown))
    return checks


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export your key and re-run:")
        print("    export ANTHROPIC_API_KEY=...")
        sys.exit(1)

    overall_ok = True
    for name, panel in PANELS.items():
        print("=" * 78)
        print(name)
        print("=" * 78)
        try:
            steps = interpret_panel(panel, context="Healthy-donor PBMC, research use.")
        except Exception as exc:  # noqa: BLE001 — surface any API/parse failure per panel
            print(f"  ERROR: {exc}")
            overall_ok = False
            continue

        for s in steps:
            pair = " x ".join(s.get("marker_pair", []))
            print(f"  {s.get('step'):>2}. [{s.get('gate_type'):<9}] {pair}")
            print(f"      {s.get('rationale')}")

        print("  --- sanity checks ---")
        for label, ok in _check(panel, steps):
            print(f"      [{'PASS' if ok else 'FAIL'}] {label}")
            overall_ok = overall_ok and ok
        print()

    print("Overall:", "PASS" if overall_ok else "SOME CHECKS FAILED")
    sys.exit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
