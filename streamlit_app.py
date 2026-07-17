"""FlowScope — Streamlit MVP (build steps 3 & 6).

Visualizes the deterministic stage-2 pipeline (parse → compensate → transform →
FlowSOM-style cluster → UMAP), the deterministic stage-3 anomaly detection, and,
when an ANTHROPIC_API_KEY is available, the stage-3 LLM interpretation of flagged
clusters plus the stage-4 report.

Everything shown without a key is fully deterministic (no AI). The AI sections
are opt-in via a button and are always framed as reference signals for analyst
review. Nothing here is diagnostic — see the banner in the app.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from flowscope.critic import critique
from flowscope.pipeline import run_pipeline
from flowscope.reporter import generate_report, render_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

st.set_page_config(page_title="FlowScope", layout="wide")

st.title("FlowScope")
st.caption(
    "Automated flow cytometry clustering + anomaly interpretation — "
    "research / education / portfolio tool."
)
st.warning(
    "⚠️ **Not a diagnostic tool.** FlowScope produces reference signals for an "
    "analyst to review, not clinical conclusions. Clustering here is fully "
    "deterministic (no AI); the AI interpretation stages come later in the "
    "pipeline and are always subject to expert review.",
    icon="⚠️",
)


def _discover_fcs_files() -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.fcs"), recursive=True))


@st.cache_data(show_spinner="Running pipeline (parse → compensate → transform → cluster → UMAP)…")
def _cached_pipeline(path: str, cofactor: float, subsample_n: int, n_metaclusters: int, seed: int):
    out = run_pipeline(
        path,
        cofactor=cofactor,
        subsample_n=subsample_n,
        n_metaclusters=n_metaclusters,
        seed=seed,
        with_umap=True,
    )
    # Return plain, cache-friendly objects.
    return {
        "summary": out.summary,
        "labels": out.result.labels,
        "umap": out.umap,
        "channels": out.result.channels,
        "markers": out.sample.markers,
        "n_metaclusters": out.result.n_metaclusters,
        "n_events_total": out.sample.events.shape[0],
        "n_sub": len(out.result.event_index),
    }


with st.sidebar:
    st.header("Input")
    files = _discover_fcs_files()
    if not files:
        st.error(
            f"No .fcs files found under {DATA_DIR}. Add the FR-FCM-ZZEB data "
            "(or run scripts/generate_synthetic_fcs.py for the synthetic demo)."
        )
        st.stop()

    rel_files = [os.path.relpath(f, DATA_DIR) for f in files]
    # Default to the first stained specimen ("Sample*") rather than a
    # single-stain compensation control, since the samples are the actual
    # analysis targets. Falls back to the first file if none match.
    default_idx = next(
        (i for i, f in enumerate(rel_files) if "sample" in os.path.basename(f).lower()),
        0,
    )
    choice = st.selectbox("FCS file", rel_files, index=default_idx)
    path = os.path.join(DATA_DIR, choice)

    st.header("Parameters")
    cofactor = st.slider("arcsinh cofactor", 5, 500, 150, step=5,
                         help="~150 for conventional fluorescence, ~5 for CyTOF.")
    subsample_n = st.select_slider("Events subsampled", [5000, 10000, 20000, 50000], value=20000)
    n_metaclusters = st.slider("Metaclusters", 5, 30, 15,
                               help="Number of populations SOM nodes are collapsed into.")
    seed = st.number_input("Random seed", value=42, step=1)
    run = st.button("Run analysis", type="primary")


# Persist the chosen parameters so results survive reruns triggered by other
# buttons on the page (report generation, downloads). Without this, clicking any
# button re-runs the script with `run` False and the results vanish.
if run:
    st.session_state["params"] = (
        path, float(cofactor), int(subsample_n), int(n_metaclusters), int(seed)
    )

if "params" not in st.session_state:
    st.info("Pick a file and parameters in the sidebar, then click **Run analysis**.")
    st.stop()

data = _cached_pipeline(*st.session_state["params"])

summary = data["summary"]
labels = data["labels"]
umap = data["umap"]
n_meta = data["n_metaclusters"]

c1, c2, c3 = st.columns(3)
c1.metric("Total events", f"{data['n_events_total']:,}")
c2.metric("Analyzed (subsample)", f"{data['n_sub']:,}")
c3.metric("Metaclusters", n_meta)

st.subheader("UMAP embedding, colored by cluster")
st.caption(
    "2D UMAP of the subsampled, compensated, transformed events. Proximity ≈ "
    "phenotypic similarity; the layout itself carries no absolute meaning."
)
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(7, 6))
for cl in range(n_meta):
    mask = labels == cl
    if mask.any():
        ax.scatter(umap[mask, 0], umap[mask, 1], s=3, color=cmap(cl % 20), label=str(cl))
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(title="cluster", markerscale=3, fontsize=7, ncol=2, loc="best")
ax.set_aspect("equal")
st.pyplot(fig, use_container_width=False)

st.subheader("Cluster frequencies")
freq = summary[["cluster", "n_events", "frequency_pct"]].copy()
st.bar_chart(freq.set_index("cluster")["frequency_pct"], y_label="frequency (%)")

st.subheader("Marker expression heatmap (median, transformed scale)")
st.caption("Rows = clusters, columns = markers. Brighter = higher median expression.")
marker_cols = [c for c in summary.columns if c not in ("cluster", "n_events", "frequency_pct")]
expr = summary.set_index("cluster")[marker_cols]
fig2, ax2 = plt.subplots(figsize=(max(8, len(marker_cols) * 0.5), max(4, n_meta * 0.35)))
im = ax2.imshow(expr.to_numpy(), aspect="auto", cmap="viridis")
ax2.set_xticks(range(len(marker_cols)))
ax2.set_xticklabels(marker_cols, rotation=90, fontsize=7)
ax2.set_yticks(range(len(expr)))
ax2.set_yticklabels(expr.index, fontsize=8)
ax2.set_ylabel("cluster")
fig2.colorbar(im, ax=ax2, label="median (arcsinh)")
fig2.tight_layout()
st.pyplot(fig2, use_container_width=False)

st.subheader("Per-cluster summary table")
st.dataframe(summary.round(3), use_container_width=True)

# --- Stage 3: anomaly detection (deterministic, no AI) ----------------------
st.subheader("Anomaly detection (deterministic)")
st.caption(
    "Clusters flagged by frequency z-score and/or IsolationForest over the "
    "per-cluster feature vector. No AI is involved in this step."
)
detected = critique(summary, interpret=False).annotated
flagged_df = detected[detected["anomaly"]]
n_flagged = len(flagged_df)
if n_flagged:
    st.write(f"**{n_flagged}** of {len(detected)} clusters flagged as anomalous:")
    st.dataframe(
        detected.loc[
            detected["anomaly"],
            ["cluster", "frequency_pct", "freq_zscore", "iforest_flag", "flagged_by"],
        ].round(3),
        use_container_width=True,
    )
else:
    st.info("No clusters flagged as anomalous at the current thresholds.")

meta = {
    "sample_label": os.path.relpath(st.session_state["params"][0], DATA_DIR),
    "total_events": data["n_events_total"],
    "analyzed_events": data["n_sub"],
    "n_clusters": n_meta,
}

# --- Stage 3 + 4: AI interpretation and report (opt-in, needs API key) ------
st.subheader("AI interpretation & report")
st.caption(
    "Interprets the flagged clusters with Claude (artifact vs. biological, with "
    "marker evidence) and drafts a LIMS-style report. Reference signals for "
    "analyst review — never diagnostic."
)
has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if not has_key:
    st.info(
        "Set `ANTHROPIC_API_KEY` (e.g. in the git-ignored `.env` file) to enable "
        "the AI interpretation and report. You can still download the "
        "deterministic report below."
    )

col_ai, col_det = st.columns(2)
gen_ai = col_ai.button("Generate AI report", type="primary", disabled=not has_key)
gen_det = col_det.button("Build report without AI (detection only)")

if gen_ai:
    try:
        with st.spinner("Interpreting flagged clusters and drafting report (Claude)…"):
            result = critique(detected, interpret=True)
            report_md = generate_report(
                result.annotated, result.critiques, meta, with_narrative=True
            )
        st.success("Report generated.")
        st.markdown(report_md)
        st.download_button("Download report (.md)", report_md, file_name="flowscope_report.md")
    except Exception as exc:  # noqa: BLE001 — surface API/billing errors to the user
        st.error(f"AI report failed: {exc}")

if gen_det:
    # Detection-only critiques (no interpretation), rendered without a narrative.
    result = critique(detected, interpret=False)
    report_md = render_report(result.annotated, result.critiques, meta, narrative=None)
    st.markdown(report_md)
    st.download_button("Download report (.md)", report_md, file_name="flowscope_report.md")
