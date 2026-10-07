"""Load pre-computed Critic evaluation results (written by scripts/score_eval.py) into display tables.

Kept separate from the Streamlit app so it can be unit-tested, and so the public deployment can show
evaluation results without calling any model (no key, no cost).
"""

from __future__ import annotations

import json
import os

import pandas as pd

DEFAULT_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "eval_synth_test", "results.json"
)


def _where(run_name: str) -> str:
    return "Gemini API (free tier)" if run_name.startswith("gemini__") else "local (Ollama)"


def _model(run_name: str) -> str:
    name = run_name.split("__", 1)[1] if run_name.startswith("gemini__") else run_name
    return name.replace("_", ":", 1) if ":" not in name and name.startswith(("gemma3_", "llama", "qwen")) else name


def load_eval_results(path: str = DEFAULT_RESULTS) -> dict | None:
    """Returns {"models": DataFrame, "by_type": DataFrame, "detector": dict} or None if no results file."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        raw = json.load(fh)

    rows = []
    for run, r in raw.get("models", {}).items():
        lo, hi = (r.get("agreement_ci") or [float("nan"), float("nan")])[:2]
        rows.append({
            "model": _model(run),
            "runs on": _where(run),
            "agreement": r["agreement"],
            "95% CI": f"{lo:.0%}–{hi:.0%}" if lo == lo else "n/a",
            "artifact called real": r["dangerous_miss"],
            "real called artifact": r["lost_population"],
            "abstained": r["abstention_rate"],
            "sec / cluster": r.get("median_latency_s"),
            "n": r["n_scored"],
        })
    models = pd.DataFrame(rows).sort_values("agreement", ascending=False).reset_index(drop=True)

    by_type = pd.DataFrame({
        _model(run): {cls: f"{v['sum']}/{v['count']}" for cls, v in classes.items()}
        for run, classes in raw.get("by_true_class", {}).items()
    })
    by_type.index.name = "true cluster type"
    return {"models": models, "by_type": by_type, "detector": raw.get("detector") or {}}
