"""Known-answer ("spike-in") evaluation set: synthetic PBMC-like files with artifacts we put in ourselves.

Why: no open database labels flow-cytometry clusters as "technical artifact" vs "real population"
(public benchmarks only carry manually gated cell-type labels). With simulated data the
truth is known by construction, so the Critic can be scored with no human labels.

What is simulated (every event carries a hidden true class):
  REAL      CD4 T, CD8 T, B, NK, monocytes, plus two rare but coherent populations
            (NKT-like, plasmablast-like)
  ARTIFACT  dead cells (viability dye high + broad non-specific staining), debris (everything low),
            T:B doublets, T:monocyte doublets (markers that should not co-occur), antibody aggregates

Truth per cluster (thresholds fixed BEFORE any model is scored):
  ARTIFACT if >= 80% of its events are artifact classes, REAL if >= 80% are real classes,
  otherwise UNSURE (mixed cluster; excluded from every metric).

Outputs
  data/eval_synth/raw/synth-{a..e}.fcs (+ .spillover.csv)   small, committed
  data/eval_synth/clusters.csv     what FlowScope / the Critic sees (no truth columns)
  data/eval_synth/gold_truth.csv   item_id, gold_label, purity, composition  (scoring only)

Usage: .venv/bin/python scripts/make_spikein_eval.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
from flowio import create_fcs

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from generate_synthetic_fcs import build_spillover_matrix  # noqa: E402
from flowscope.critic import detect_anomalies  # noqa: E402
from flowscope.pipeline import run_pipeline  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"   # dev: used to choose the prompt; test: held out, scored once
assert SPLIT in ("dev", "test")
SEED0 = 100 if SPLIT == "dev" else 200
OUT = os.path.join(ROOT, "data", "eval_synth" if SPLIT == "dev" else "eval_synth_test")
RAW = os.path.join(OUT, "raw")
FLUOR = ["CD3-FITC", "CD4-PE", "CD8-APC", "CD19-PerCP", "CD14-PE-Cy7", "CD56-APC-Cy7",
         "CD16-BV421", "HLA-DR-BV510", "CD38-BV605", "CD45RA-BV711", "Viability-AmCyan"]
CHANNELS = ["FSC-A", "SSC-A"] + FLUOR + ["Time"]
N_EVENTS = 20000
NOISE_CV = 0.25
N_METACLUSTERS = 20
ARTIFACT_THRESHOLD = 0.80  # fixed in advance

#                    CD3   CD4   CD8  CD19  CD14  CD56  CD16 HLADR CD38 CD45RA Viab
def _m(*v): return list(v)

# name: (class, base fraction, (FSC mean, SSC mean), linear true means in FLUOR order)
POPS = {
    "CD4_T":        ("REAL", 0.28, (55000, 20000), _m(8000, 6000, 200, 100, 100, 100, 100, 300, 500, 4000, 150)),
    "CD8_T":        ("REAL", 0.17, (52000, 21000), _m(7500, 150, 6500, 120, 100, 150, 150, 300, 800, 2500, 150)),
    "B_cells":      ("REAL", 0.09, (58000, 30000), _m(150, 100, 100, 7000, 120, 100, 100, 5000, 300, 200, 150)),
    "NK_cells":     ("REAL", 0.08, (60000, 45000), _m(130, 100, 400, 100, 150, 6800, 5500, 200, 200, 200, 150)),
    "Monocytes":    ("REAL", 0.12, (95000, 85000), _m(200, 250, 150, 150, 7200, 100, 800, 6000, 300, 150, 150)),
    "NKT_like":     ("REAL", 0.02, (56000, 25000), _m(7000, 200, 3000, 100, 100, 5500, 1500, 400, 600, 800, 150)),
    "Plasmablast":  ("REAL", 0.015, (70000, 40000), _m(120, 100, 100, 3500, 100, 100, 100, 7000, 8000, 200, 150)),
    "Dead_cells":   ("ARTIFACT", 0.09, (30000, 45000), _m(1200, 1000, 900, 1100, 1300, 1000, 1000, 1500, 1800, 1200, 9000)),
    "Debris":       ("ARTIFACT", 0.05, (8000, 20000), _m(150, 150, 150, 150, 150, 150, 150, 150, 200, 150, 200)),
    "Doublet_T_B":  ("ARTIFACT", 0.035, (105000, 45000), _m(7500, 3000, 100, 6500, 100, 100, 100, 4500, 500, 2000, 200)),
    "Doublet_T_Mono": ("ARTIFACT", 0.03, (120000, 110000), _m(7000, 2500, 150, 100, 6800, 100, 700, 5500, 400, 1500, 200)),
    "Ab_aggregates": ("ARTIFACT", 0.02, (65000, 35000), _m(5200, 4800, 5000, 5100, 4900, 5200, 5000, 5100, 4800, 5000, 300)),
}


def simulate(seed: int):
    rng = np.random.default_rng(seed)
    donor = rng.lognormal(0.0, 0.15, size=len(FLUOR))          # file-level staining-intensity shift
    frac = {}
    for name, (cls, f, _, _) in POPS.items():
        frac[name] = f * (rng.uniform(0.6, 1.5) if cls == "ARTIFACT" else rng.uniform(0.85, 1.15))
    tot = sum(frac.values())
    counts = {n: int(round(N_EVENTS * f / tot)) for n, f in frac.items()}

    fl, sc, label = [], [], []
    for name, (cls, _, (fsc_m, ssc_m), means) in POPS.items():
        n = counts[name]
        mu = np.array(means, dtype=float) * donor
        fl.append(mu[None, :] * rng.lognormal(0.0, NOISE_CV, size=(n, len(FLUOR))))
        sc.append(np.column_stack([rng.normal(fsc_m, fsc_m * 0.10, n), rng.normal(ssc_m, ssc_m * 0.12, n)]))
        label += [name] * n
    true_expr, scatter, label = np.vstack(fl), np.vstack(sc), np.array(label)
    order = rng.permutation(len(label))                          # events are not sorted by population
    true_expr, scatter, label = true_expr[order], scatter[order], label[order]

    spill = build_spillover_matrix(len(FLUOR))
    measured = np.clip(true_expr @ spill.T, 1.0, None)
    time_ch = np.sort(rng.uniform(0, 600, size=len(label)))
    data = np.column_stack([scatter, measured, time_ch]).astype(np.float32)
    return data, label, spill


def write_fcs(key, data, spill):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{key}.fcs")
    with open(path, "wb") as fh:
        create_fcs(fh, data.flatten().tolist(), channel_names=CHANNELS,
                   metadata_dict={"cyt": "Synthetic", "experiment name": "FlowScope spike-in evaluation (simulated)"})
    sp = os.path.join(RAW, f"{key}.spillover.csv")
    with open(sp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + FLUOR)
        for name, row in zip(FLUOR, spill):
            w.writerow([name] + list(row))
    return path, sp


def main():
    os.makedirs(OUT, exist_ok=True)
    clusters, truth = [], []
    for i, key in enumerate(["synth-a", "synth-b", "synth-c", "synth-d", "synth-e"]):
        data, label, spill = simulate(seed=SEED0 + i)
        path, sp = write_fcs(key, data, spill)
        out = run_pipeline(path, spillover_path=sp, subsample_n=N_EVENTS, n_metaclusters=N_METACLUSTERS,
                           seed=42, with_umap=False)
        ann = detect_anomalies(out.summary, seed=42)
        ann.insert(0, "item_id", [f"{key}-c{c}" for c in ann["cluster"]])
        ann.insert(1, "sample", key)
        clusters.append(ann)

        ev_label = label[out.result.event_index]
        is_art = np.array([POPS[l][0] == "ARTIFACT" for l in ev_label])
        for c in ann["cluster"]:
            m = out.result.labels == c
            art = float(is_art[m].mean())
            comp = pd.Series(ev_label[m]).value_counts(normalize=True)
            gold = "ARTIFACT" if art >= ARTIFACT_THRESHOLD else "REAL" if (1 - art) >= ARTIFACT_THRESHOLD else "UNSURE"
            truth.append({"item_id": f"{key}-c{c}", "gold_label": gold,
                          "purity": round(max(art, 1 - art), 3),
                          "composition": ";".join(f"{k}:{v:.2f}" for k, v in comp.head(4).items())})
        print(f"{key}: {len(ann)} clusters", flush=True)

    pd.concat(clusters, ignore_index=True).to_csv(os.path.join(OUT, "clusters.csv"), index=False)
    t = pd.DataFrame(truth)
    t.to_csv(os.path.join(OUT, "gold_truth.csv"), index=False)
    print(t["gold_label"].value_counts().to_dict(), "->", OUT)


if __name__ == "__main__":
    main()
