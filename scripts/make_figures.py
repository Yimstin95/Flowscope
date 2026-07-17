"""Generate the README figures: a UMAP colored by cluster and a marker
heatmap, saved as PNGs under docs/.

Runs the deterministic stage-2 pipeline on the real FR-FCM-ZZEB Sample1.fcs if
present, else the synthetic fixture. No API key needed — figures come from the
deterministic pipeline only.

Usage:
    python scripts/make_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flowscope.pipeline import run_pipeline  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data", "raw")
DOCS = os.path.join(HERE, "..", "docs")
REAL = os.path.join(DATA_DIR, "FR-FCM-ZZEB", "Sample1.fcs")
SYNTH = os.path.join(DATA_DIR, "synthetic_pbmc_demo.fcs")
SYNTH_SPILL = os.path.join(DATA_DIR, "synthetic_pbmc_demo.spillover.csv")


def main():
    os.makedirs(DOCS, exist_ok=True)
    if os.path.exists(REAL):
        fcs, spill, src = REAL, None, "FR-FCM-ZZEB/Sample1.fcs (real OMIP-024 PBMC)"
    else:
        fcs, spill, src = SYNTH, SYNTH_SPILL, "synthetic_pbmc_demo.fcs"

    print(f"Rendering figures from {src} ...")
    out = run_pipeline(fcs, spillover_path=spill, subsample_n=20000, n_metaclusters=15, with_umap=True)
    labels = out.result.labels
    n_meta = out.result.n_metaclusters

    # --- UMAP colored by cluster ---
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7, 6))
    for cl in range(n_meta):
        mask = labels == cl
        if mask.any():
            ax.scatter(out.umap[mask, 0], out.umap[mask, 1], s=3, color=cmap(cl % 20), label=str(cl))
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"FlowScope — UMAP by cluster\n{src}")
    ax.legend(title="cluster", markerscale=3, fontsize=7, ncol=2, loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    umap_path = os.path.join(DOCS, "umap.png")
    fig.savefig(umap_path, dpi=120)
    print(f"  wrote {os.path.relpath(umap_path, HERE)}")

    # --- Marker heatmap ---
    summary = out.summary
    marker_cols = [c for c in summary.columns if c not in ("cluster", "n_events", "frequency_pct")]
    expr = summary.set_index("cluster")[marker_cols]
    fig2, ax2 = plt.subplots(figsize=(max(8, len(marker_cols) * 0.5), max(4, n_meta * 0.35)))
    im = ax2.imshow(expr.to_numpy(), aspect="auto", cmap="viridis")
    ax2.set_xticks(range(len(marker_cols)))
    ax2.set_xticklabels(marker_cols, rotation=90, fontsize=7)
    ax2.set_yticks(range(len(expr)))
    ax2.set_yticklabels(expr.index, fontsize=8)
    ax2.set_ylabel("cluster")
    ax2.set_title("Per-cluster median marker expression (arcsinh scale)")
    fig2.colorbar(im, ax=ax2, label="median (arcsinh)")
    fig2.tight_layout()
    heatmap_path = os.path.join(DOCS, "heatmap.png")
    fig2.savefig(heatmap_path, dpi=120)
    print(f"  wrote {os.path.relpath(heatmap_path, HERE)}")


if __name__ == "__main__":
    main()
