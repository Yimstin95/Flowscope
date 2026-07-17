"""Live validation of the Critic (stage 3) on real data, end to end.

Runs the full deterministic pipeline on a real FR-FCM-ZZEB sample (parse ->
compensate -> transform -> cluster -> per-cluster summary), flags anomalous
clusters with the pure-Python detector, then asks Claude to interpret each
flagged cluster. Prints the detector output and each interpretation.

Makes real Claude API calls -> needs ANTHROPIC_API_KEY. Falls back to the
synthetic fixture if the real data isn't present.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/validate_critic.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowscope.critic import critique  # noqa: E402
from flowscope.pipeline import run_pipeline  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
REAL = os.path.join(DATA_DIR, "FR-FCM-ZZEB", "Sample1.fcs")
SYNTH = os.path.join(DATA_DIR, "synthetic_pbmc_demo.fcs")
SYNTH_SPILL = os.path.join(DATA_DIR, "synthetic_pbmc_demo.spillover.csv")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export your key and re-run.")
        sys.exit(1)

    if os.path.exists(REAL):
        fcs, spill, label = REAL, None, "FR-FCM-ZZEB/Sample1.fcs (real)"
    else:
        fcs, spill, label = SYNTH, SYNTH_SPILL, "synthetic_pbmc_demo.fcs (fallback)"

    print(f"Running pipeline on {label} ...")
    out = run_pipeline(fcs, spillover_path=spill, subsample_n=20000, n_metaclusters=15, with_umap=False)

    result = critique(out.summary, interpret=True)

    annotated = result.annotated
    flagged = annotated[annotated["anomaly"]]
    print(f"\n{len(flagged)} of {len(annotated)} clusters flagged as anomalous.\n")
    print(annotated[["cluster", "frequency_pct", "freq_zscore", "iforest_flag", "flagged_by"]]
          .round(3).to_string(index=False))

    print("\n--- LLM interpretations (flagged clusters only) ---")
    for c in result.critiques:
        interp = c.interpretation or {}
        print(f"\nCluster {c.cluster}  ({c.frequency_pct:.2f}%, flagged by: {c.flagged_by})")
        print(f"  verdict    : {interp.get('verdict')}  [confidence: {interp.get('confidence')}]")
        print(f"  key markers: {', '.join(interp.get('key_markers') or []) or '(none)'}")
        print(f"  reasoning  : {interp.get('reasoning')}")


if __name__ == "__main__":
    main()
