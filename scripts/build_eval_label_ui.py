"""Build data/eval/label_ui/index.html: the blind expert-labeling page for the eval clusters.

Shows marker values and a per-sample heatmap only. Detector flags and model output are NOT embedded.
Usage: .venv/bin/python scripts/build_eval_label_ui.py
"""
import json
import os
import random
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVAL = os.path.join(ROOT, "data", "eval")
NON_MARKER = {"item_id", "sample", "cluster", "n_events", "frequency_pct", "freq_zscore",
              "iforest_flag", "iforest_score", "anomaly", "flagged_by"}
TITLES = {
    "omip024-s1": "OMIP-024 PBMC, Sample1 (18-colour, FlowRepository FR-FCM-ZZEB)",
    "omip024-s2": "OMIP-024 PBMC, Sample2 (18-colour, FlowRepository FR-FCM-ZZEB)",
    "flowio-13c": "FlowIO demo PBMC (13-colour, BD FACSAria)",
}


def main():
    SUBSET_N = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = label every cluster
    df = pd.read_csv(os.path.join(EVAL, "clusters.csv"))
    samples, items = {}, []
    rng = random.Random(2026 if SUBSET_N else 7)
    for key, g in df.groupby("sample", sort=False):
        markers = [c for c in g.columns if c not in NON_MARKER and g[c].notna().any()]
        samples[key] = {
            "title": TITLES.get(key, key),
            "markers": markers,
            "clusters": [
                {"id": r["item_id"], "cluster": int(r["cluster"]), "n": int(r["n_events"]),
                 "freq": float(r["frequency_pct"]), "v": [float(r[m]) for m in markers]}
                for _, r in g.iterrows()
            ],
        }
        ids = list(g["item_id"])
        rng.shuffle(ids)  # order inside a sample carries no information
        if SUBSET_N:
            # Labeling every cluster takes ~1 h, so label a random subset, split across samples
            # in proportion to their size. The heatmap still shows ALL clusters for context.
            ids = ids[: max(1, round(SUBSET_N * len(ids) / len(df)))]
        items += [{"item_id": x, "sample": key} for x in ids]
    if SUBSET_N:
        with open(os.path.join(EVAL, "label_subset.json"), "w") as fh:
            json.dump(sorted(x["item_id"] for x in items), fh, indent=1)
    html = open(os.path.join(os.path.dirname(__file__), "eval_label_template.html")).read()
    html = html.replace("/*DATA*/{}", json.dumps({"samples": samples, "items": items}))
    out_dir = os.path.join(EVAL, "label_ui")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write(html)
    print(f"{len(items)} items across {len(samples)} samples -> {out_dir}/index.html")

    # Optional practice page: clusters OUTSIDE the labeling subset, with its own storage key,
    # so practising can never mix into the real gold labels.
    practice = [p for p in os.environ.get("PRACTICE_IDS", "").split(",") if p]
    if practice:
        in_subset = {x["item_id"] for x in items}
        assert not (set(practice) & in_subset) or not SUBSET_N, "practice ids must not be in the labeling subset"
        pitems = [{"item_id": p, "sample": p.rsplit("-c", 1)[0]} for p in practice]
        phtml = open(os.path.join(os.path.dirname(__file__), "eval_label_template.html")).read()
        phtml = phtml.replace("/*DATA*/{}", json.dumps({"samples": samples, "items": pitems}))
        phtml = phtml.replace("flowscope_gold_v2", "flowscope_practice")
        phtml = phtml.replace("<title>FlowScope cluster labeling</title>", "<title>PRACTICE - FlowScope labeling</title>")
        phtml = phtml.replace("<h1>FlowScope cluster labeling", "<h1>PRACTICE (not saved to the real labels)")
        with open(os.path.join(out_dir, "practice.html"), "w") as fh:
            fh.write(phtml)
        print(f"practice page ({len(pitems)} items) -> {out_dir}/practice.html")


if __name__ == "__main__":
    main()
