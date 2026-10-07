"""Build the frozen evaluation item set: every cluster on three real samples.

Output: data/eval/clusters.csv  (one row per cluster; marker medians on the arcsinh scale)
Deterministic given the raw FCS files and the fixed seed below.

Usage: .venv/bin/python scripts/make_eval_set.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from flowscope.critic import detect_anomalies  # noqa: E402
from flowscope.pipeline import run_pipeline  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "eval")
SEED = 42

# (sample key, path, number of metaclusters)
SAMPLES = [
    ("omip024-s1", os.path.join(RAW, "FR-FCM-ZZEB", "Sample1.fcs"), 20),
    ("omip024-s2", os.path.join(RAW, "FR-FCM-ZZEB", "Sample2.fcs"), 20),
    ("flowio-13c", os.path.join(RAW, "flowio-demo", "pbmc_13color_facsaria.fcs"), 15),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for key, path, k in SAMPLES:
        print(f"{key}: clustering {os.path.basename(path)} into {k} metaclusters ...", flush=True)
        out = run_pipeline(path, subsample_n=20000, n_metaclusters=k, seed=SEED, with_umap=False)
        ann = detect_anomalies(out.summary, seed=SEED)
        ann.insert(0, "item_id", [f"{key}-c{c}" for c in ann["cluster"]])
        ann.insert(1, "sample", key)
        frames.append(ann)
        print(f"   {len(ann)} clusters, {int(ann['anomaly'].sum())} flagged by the detector", flush=True)
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(OUT_DIR, "clusters.csv")
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} clusters -> {path}")


if __name__ == "__main__":
    main()
