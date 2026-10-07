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
import hashlib
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Defensive fallback: the flowscope package (src/flowscope/) is normally made
# importable via `pip install -e .` (requirements.txt carries a `-e .` line
# specifically so Streamlit Community Cloud's `pip install -r requirements.txt`
# picks it up too). If that install step was ever skipped, fall back to adding
# src/ onto sys.path directly rather than hard-crashing with ModuleNotFoundError.
try:
    import flowscope  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# On Streamlit Community Cloud, secrets are supplied via st.secrets (pasted in
# the app's dashboard), not a .env file. Bridge it into the environment so the
# Anthropic SDK (which reads ANTHROPIC_API_KEY from os.environ) finds it either
# way, without changing any downstream code. st.secrets raises if no secrets
# file exists at all (e.g. a bare local clone with just .env), so guard it.
for _key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    if not os.environ.get(_key):
        try:
            os.environ[_key] = st.secrets[_key]
        except (FileNotFoundError, KeyError, st.errors.StreamlitAPIException):
            pass

from flowscope.critic import critique
from flowscope.eval_results import load_eval_results
from flowscope.llm_providers import GeminiClient
from flowscope.panel_interpreter import interpret_panel, panel_from_channels
from flowscope.pipeline import run_pipeline
from flowscope.reporter import compute_file_checksum, generate_report, render_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "flowscope_uploads")

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

# Evaluation results are pre-computed (scripts/score_eval.py), so this panel needs no key and makes
# no model calls. It sits above the analysis so visitors see it without running anything.
_eval = load_eval_results()
with st.expander("📊 How far can the AI step be trusted? — evaluation results", expanded=False):
    if _eval is None:
        st.info("No evaluation results found (`data/eval_synth_test/results.json`).")
    else:
        st.markdown(
            "The AI step (*technical artifact* vs *real population* for a cluster) was scored on a "
            "held-out **known-answer test**: 63 simulated PBMC-like clusters (37 real populations, "
            "26 planted artifacts: dead cells, debris, T:B and T:monocyte doublets, antibody aggregates). "
            "Original prompt, temperature 0, scored **once**; models and prompt were chosen on a separate "
            "development set."
        )
        _disp = _eval["models"].copy()
        for _c in ("agreement", "artifact called real", "real called artifact", "abstained"):
            _disp[_c] = (_disp[_c] * 100).round(0).astype(int).astype(str) + "%"
        _disp["sec / cluster"] = _disp["sec / cluster"].round(0).astype(int)
        st.dataframe(_disp, hide_index=True, use_container_width=True)
        _d = _eval["detector"]
        if _d:
            st.caption(
                f"For comparison, the statistical anomaly detector alone (no AI) catches "
                f"{_d.get('artifact_recall', float('nan')):.0%} of artifacts and also flags "
                f"{_d.get('false_flag_on_real', float('nan')):.0%} of real clusters."
            )
        st.markdown("**Correct calls by true cluster type**")
        st.dataframe(_eval["by_type"], use_container_width=True)
        st.markdown(
            "**Reading it.** Model size decided the outcome: the same model family abstained on every "
            "cluster at 4B (local) and scored 97% at 31B. The remaining errors sit where they also sit at "
            "the bench: **T:monocyte doublets** and **rare real populations** — confirm those manually.\n\n"
            "**Limits.** Simulated clusters are cleaner than real samples (every cluster was 100% pure), so "
            "these numbers are an upper bound, not real-world accuracy. Test and prompt were written by the same "
            "person (no simulator-specific hints were allowed in the prompt). n = 63. Free-tier cloud models may "
            "use submitted inputs — never send confidential data. Details: `docs/eval-design.md`."
        )


def _discover_fcs_files() -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.fcs"), recursive=True))


def _save_uploaded_fcs(uploaded) -> str:
    """Persist an uploaded FCS to a content-hashed temp path and return it.

    Hashing the bytes keeps the path stable across reruns for the same upload,
    so @st.cache_data-keyed _cached_pipeline(path, ...) hits its cache instead of
    reprocessing on every button click.
    """
    raw = uploaded.getvalue()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{digest}.fcs")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(raw)
    return path


def _sibling_spillover_csv(fcs_path: str) -> str | None:
    """Some fixtures (e.g. the synthetic demo) have no embedded $SPILLOVER and
    ship their ground-truth matrix as a sibling <name>.spillover.csv instead
    (see scripts/generate_synthetic_fcs.py). Auto-detect it so picking that
    file in the UI doesn't crash with "cannot compensate"."""
    candidate = os.path.splitext(fcs_path)[0] + ".spillover.csv"
    return candidate if os.path.exists(candidate) else None


@st.cache_data(show_spinner="Running pipeline (parse → compensate → transform → cluster → UMAP)…")
def _cached_pipeline(path: str, cofactor: float, subsample_n: int, n_metaclusters: int, seed: int):
    spillover_path = _sibling_spillover_csv(path)
    out = run_pipeline(
        path,
        spillover_path=spillover_path,
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
        "channel_names": out.sample.channel_names,  # all channels incl. scatter, for stage ①
        "markers": out.sample.markers,
        "n_metaclusters": out.result.n_metaclusters,
        "n_events_total": out.sample.events.shape[0],
        "n_sub": len(out.result.event_index),
        "spillover_source": "external CSV" if spillover_path else "embedded in FCS file",
    }


with st.sidebar:
    st.header("Input")
    files = _discover_fcs_files()
    if not files:
        # Fresh clone / cloud deploy: data/raw/ only ships a .gitkeep (the real
        # FR-FCM-ZZEB data is git-ignored, ~395MB). Generate the synthetic demo
        # fixture on first run so the app is usable out of the box.
        with st.spinner("First run: generating synthetic demo data…"):
            import subprocess
            import sys

            script = os.path.join(os.path.dirname(__file__), "scripts", "generate_synthetic_fcs.py")
            subprocess.run([sys.executable, script], check=True)
        files = _discover_fcs_files()

    if not files:
        st.error(f"No .fcs files found under {DATA_DIR} and synthetic generation failed.")
        st.stop()

    if not any("ZZEB" in f for f in files):
        st.info(
            "Running on synthetic demo data (no real FR-FCM-ZZEB sample present). "
            "Drop the real data into `data/raw/FR-FCM-ZZEB/` for the real OMIP-024 sample.",
            icon="🧪",
        )

    # Upload widget: an uploaded file becomes a first-class, default-selected
    # option alongside the bundled files. Its panel drives stage ① just the same.
    uploaded = st.file_uploader(
        "Upload an FCS file", type=["fcs"],
        help="Your file is processed in-session; nothing is stored server-side beyond a temp copy.",
    )

    # Map display label → absolute path. Uploaded file is prepended and default.
    options: dict[str, str] = {}
    if uploaded is not None:
        options[f"⬆ {uploaded.name} (uploaded)"] = _save_uploaded_fcs(uploaded)
    for f in files:
        options[os.path.relpath(f, DATA_DIR)] = f

    file_labels = list(options)
    # Default: the uploaded file if present, else the first stained specimen
    # ("Sample*") rather than a single-stain compensation control.
    default_idx = 0 if uploaded is not None else next(
        (i for i, lbl in enumerate(file_labels) if "sample" in os.path.basename(lbl).lower()),
        0,
    )
    choice = st.selectbox("FCS file", file_labels, index=default_idx)
    path = options[choice]

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

try:
    data = _cached_pipeline(*st.session_state["params"])
except ValueError as exc:
    # run_pipeline raises ValueError when an FCS has no embedded compensation
    # matrix (and no bundled spillover CSV) — common for uploaded exports and
    # the synthetic fixture. Show guidance instead of a traceback.
    st.error(
        f"Could not run the pipeline on this file: {exc}\n\n"
        "This usually means the FCS has no embedded compensation ($SPILLOVER) "
        "matrix. FlowScope currently needs one to compensate. Try a file exported "
        "with its spillover matrix, or one of the bundled samples."
    )
    st.stop()

summary = data["summary"]
labels = data["labels"]
umap = data["umap"]
n_meta = data["n_metaclusters"]

c1, c2, c3 = st.columns(3)
c1.metric("Total events", f"{data['n_events_total']:,}")
c2.metric("Analyzed (subsample)", f"{data['n_sub']:,}")
c3.metric("Metaclusters", n_meta)

# --- Stage ①: Panel Interpreter — suggested gating strategy (opt-in AI) ------
st.subheader("① Suggested gating strategy (AI)")
st.caption(
    "Reads this sample's marker panel and proposes a hierarchical gating strategy "
    "an analyst could start from — a suggestion for review, not an applied gate "
    "and not a diagnostic step. Independent of the clustering below."
)
_has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if not _has_key:
    st.info(
        "Set `ANTHROPIC_API_KEY` (git-ignored `.env`, or Streamlit secrets when "
        "deployed) to enable the gating-strategy suggestion. This step calls the "
        "Claude API and uses credits.",
        icon="🔑",
    )
if st.button("Suggest gating strategy", disabled=not _has_key,
             help="Calls the Claude API (uses credits)."):
    try:
        panel = panel_from_channels(data["channel_names"], data["markers"])
        with st.spinner("Proposing a gating strategy from the panel (Claude)…"):
            steps = interpret_panel(panel, context="Sample analyzed in FlowScope.")
        st.session_state["gating_steps"] = steps
    except Exception as exc:  # noqa: BLE001 — surface API/billing errors to the user
        st.error(f"Gating suggestion failed: {exc}")

if st.session_state.get("gating_steps"):
    st.dataframe(
        [
            {
                "step": s.get("step"),
                "marker_pair": " × ".join(s.get("marker_pair", [])),
                "gate_type": s.get("gate_type"),
                "rationale": s.get("rationale"),
            }
            for s in st.session_state["gating_steps"]
        ],
        use_container_width=True,
        hide_index=True,
    )

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

_selected_path, _cofactor, _subsample_n, _n_metaclusters, _seed = st.session_state["params"]
meta = {
    "sample_label": os.path.relpath(_selected_path, DATA_DIR),
    "total_events": data["n_events_total"],
    "analyzed_events": data["n_sub"],
    "n_clusters": n_meta,
    # Analysis Provenance fields (see reporter.render_report) — recorded so the
    # exact settings behind a given report can be reproduced later.
    "file_path": os.path.relpath(_selected_path, DATA_DIR),
    "file_sha256": compute_file_checksum(_selected_path),
    "spillover_source": data["spillover_source"],
    "cofactor": _cofactor,
    "subsample_n": _subsample_n,
    "seed": _seed,
}

# --- Stage 3 + 4: AI interpretation and report (opt-in, needs API key) ------
st.subheader("AI interpretation & report")
st.caption(
    "Interprets the flagged clusters (artifact vs. biological, with marker evidence) "
    "and drafts a LIMS-style report. Reference signals for analyst review — never "
    "diagnostic. See the evaluation panel at the top for how far each model can be trusted."
)
# Offer only the back-ends that have a key configured (git-ignored .env locally, or
# Streamlit secrets). The public deployment ships with no keys, so this stays disabled there.
ai_options: dict[str, tuple[str, str | None]] = {}
if os.environ.get("GEMINI_API_KEY"):
    ai_options["Gemini API free tier — gemma-4-31b-it (best in our evaluation, ~1 min/cluster)"] = ("gemini", "gemma-4-31b-it")
    ai_options["Gemini API free tier — gemini-3.5-flash-lite (faster, ~10 s/cluster; 84% in our evaluation)"] = ("gemini", "gemini-3.5-flash-lite")
if os.environ.get("ANTHROPIC_API_KEY"):
    ai_options["Claude — original design (paid; not evaluated here)"] = ("anthropic", None)
if not ai_options:
    st.info(
        "No AI key configured. Set `GEMINI_API_KEY` (free tier) or `ANTHROPIC_API_KEY` in the "
        "git-ignored `.env` file to enable the AI interpretation. You can still download the "
        "deterministic report below."
    )
    ai_choice = None
else:
    ai_choice = st.radio("AI model", list(ai_options), index=0)
    if ai_options[ai_choice][0] == "gemini":
        st.caption(
            "Only per-cluster summary numbers (marker medians), not raw events, are sent to Google. "
            "Free-tier inputs may be used by Google — do not use with confidential data. "
            "Clusters the service fails on are marked in the report for manual review."
        )

col_ai, col_det = st.columns(2)
gen_ai = col_ai.button("Generate AI report", type="primary", disabled=ai_choice is None)
gen_det = col_det.button("Build report without AI (detection only)")

if gen_ai:
    kind, model_id = ai_options[ai_choice]
    try:
        if kind == "gemini":
            with st.spinner(f"Interpreting {n_flagged} flagged clusters with {model_id} (Gemini free tier)…"):
                result = critique(detected, interpret=True, client=GeminiClient(), model=model_id,
                                  on_error="skip")
                # The narrative step is Claude-only; the per-cluster interpretations are the evaluated part.
                report_md = render_report(
                    result.annotated, result.critiques,
                    {**meta, "ai_model": f"{model_id} via Gemini API (free tier)"}, narrative=None,
                )
        else:
            with st.spinner("Interpreting flagged clusters and drafting report (Claude)…"):
                result = critique(detected, interpret=True)
                report_md = generate_report(
                    result.annotated, result.critiques, meta, with_narrative=True
                )
        _failed = [c.cluster for c in result.critiques if getattr(c, "interpretation_error", None)]
        if _failed:
            st.warning(
                f"The AI service failed for cluster(s) {', '.join(map(str, _failed))} "
                "(provider error, not a model verdict). They are marked in the report for manual "
                "review; you can also retry later."
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
