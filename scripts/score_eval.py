"""Score every model run in <eval-dir>/runs/ against a gold file, and write results.json.

Usage:
    .venv/bin/python scripts/score_eval.py --eval-dir data/eval_synth --gold data/eval_synth/gold_truth.csv
    .venv/bin/python scripts/score_eval.py --eval-dir data/eval --gold data/eval/gold_labels.csv   # expert labels
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from flowscope.evaluation import confusion, score_critic, score_detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    gold = pd.read_csv(args.gold)
    clusters = pd.read_csv(os.path.join(eval_dir, "clusters.csv"))

    out = {"gold_file": os.path.relpath(args.gold, ROOT), "models": {}, "detector": None, "by_true_class": {}}
    out["detector"] = score_detector(clusters, gold)

    for f in sorted(glob.glob(os.path.join(eval_dir, "runs", "*.csv"))):
        model = os.path.basename(f)[:-4]
        preds = pd.read_csv(f)
        preds["ok"] = preds["ok"].astype(bool)
        r = score_critic(preds, gold)
        r["confusion"] = confusion(preds, gold).to_dict()
        out["models"][model] = r

        # which kinds of cluster does this model get wrong? (only if the gold file lists composition)
        if "composition" in gold.columns:
            d = preds.merge(gold, on="item_id")
            d = d[d["gold_label"].isin(["ARTIFACT", "REAL"])].copy()
            d["true_class"] = d["composition"].str.split(";").str[0].str.split(":").str[0]
            d["agree"] = d.apply(
                lambda x: (x["ok"] and {"likely_artifact": "ARTIFACT", "likely_biological": "REAL"}.get(x["verdict"]) == x["gold_label"]), axis=1)
            out["by_true_class"][model] = d.groupby("true_class")["agree"].agg(["sum", "count"]).astype(int).to_dict("index")

    path = os.path.join(eval_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if o != o else str(o))
    print("wrote", path)

    pct = lambda x: "n/a" if x != x else f"{100 * x:.0f}%"
    ci = lambda r, k: f"{pct(r[k + '_ci'][0])}-{pct(r[k + '_ci'][1])}"
    print(f"\nGold: {out['gold_file']}")
    d = out["detector"]
    print(f"Detector (no AI): flags {d['n_flagged']} clusters | catches {pct(d['artifact_recall'])} of true artifacts | "
          f"wrongly flags {pct(d['false_flag_on_real'])} of real | {pct(d['flagged_precision'])} of its flags are true artifacts\n")
    hdr = f"{'model':14}{'n':>4}{'agree':>7}{'(95% CI)':>11}{'danger-miss':>13}{'lost-real':>11}{'abstain':>9}{'format-fail':>12}{'sec/item':>9}"
    print(hdr)
    for m, r in out["models"].items():
        print(f"{m:14}{r['n_scored']:>4}{pct(r['agreement']):>7}{ci(r, 'agreement'):>11}{pct(r['dangerous_miss']):>13}"
              f"{pct(r['lost_population']):>11}{pct(r['abstention_rate']):>9}{pct(r['format_failure_rate']):>12}{r.get('median_latency_s', float('nan')):>9.0f}")
    if out["by_true_class"]:
        print("\nCorrect calls by true cluster type (correct/total):")
        classes = sorted({c for v in out["by_true_class"].values() for c in v})
        print(f"{'':16}" + "".join(f"{m:>14}" for m in out["by_true_class"]))
        for c in classes:
            print(f"{c:16}" + "".join(f"{str(v.get(c, {}).get('sum', '-')) + '/' + str(v.get(c, {}).get('count', '-')):>14}" for v in out["by_true_class"].values()))


if __name__ == "__main__":
    main()
