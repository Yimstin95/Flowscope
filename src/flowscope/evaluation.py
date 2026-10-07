"""Score the Critic (and the anomaly detector) against an expert's gold labels.

Gold labels : ARTIFACT | REAL | UNSURE   (UNSURE items are excluded from every accuracy metric)
Model verdicts (critic.INTERPRET_TOOL): likely_artifact | likely_biological | needs_confirmation

Reported (see docs/eval-design.md for the plain-language reading):
  agreement             verdict == gold, over scored items (abstentions and format failures count as "no match")
  agreement_confident   verdict == gold, over the items where the model committed to a call
  abstention_rate       share answered `needs_confirmation`
  format_failure_rate   share where no valid structured answer came back
  dangerous_miss        gold ARTIFACT called likely_biological, / all gold ARTIFACT
  lost_population       gold REAL called likely_artifact, / all gold REAL
  abstain_on_unsure     share of gold-UNSURE items where the model said needs_confirmation (informational)
Detector (no LLM): artifact_recall, false_flag_on_real, flagged_precision.
Bootstrap 95% intervals are attached to the headline metrics.
"""

from __future__ import annotations

import random

import pandas as pd

VERDICT_TO_LABEL = {"likely_artifact": "ARTIFACT", "likely_biological": "REAL"}
HEADLINE = ("agreement", "dangerous_miss", "lost_population")


def _call(row) -> str:
    """Model's committed label: ARTIFACT / REAL, or ABSTAIN / FAIL."""
    if not row.get("ok", True):
        return "FAIL"
    return VERDICT_TO_LABEL.get(row["verdict"], "ABSTAIN")


def _metrics(df: pd.DataFrame) -> dict:
    scored = df[df["gold_label"].isin(["ARTIFACT", "REAL"])]
    n = len(scored)
    art, real = scored[scored["gold_label"] == "ARTIFACT"], scored[scored["gold_label"] == "REAL"]
    committed = scored[scored["call"].isin(["ARTIFACT", "REAL"])]
    unsure = df[df["gold_label"] == "UNSURE"]

    def rate(num, den):
        return float(num) / den if den else float("nan")

    return {
        "n_scored": n,
        "n_gold_artifact": len(art),
        "n_gold_real": len(real),
        "n_gold_unsure": len(unsure),
        "agreement": rate((scored["call"] == scored["gold_label"]).sum(), n),
        "agreement_confident": rate((committed["call"] == committed["gold_label"]).sum(), len(committed)),
        "abstention_rate": rate((scored["call"] == "ABSTAIN").sum(), n),
        "format_failure_rate": rate((scored["call"] == "FAIL").sum(), n),
        "dangerous_miss": rate((art["call"] == "REAL").sum(), len(art)),
        "lost_population": rate((real["call"] == "ARTIFACT").sum(), len(real)),
        "abstain_on_unsure": rate((unsure["call"] == "ABSTAIN").sum(), len(unsure)),
    }


def score_critic(preds: pd.DataFrame, gold: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """preds: item_id, verdict, ok(optional), latency_s(optional). gold: item_id, gold_label."""
    df = preds.merge(gold[["item_id", "gold_label"]], on="item_id", how="inner").copy()
    df["call"] = df.apply(_call, axis=1)
    out = _metrics(df)
    if "latency_s" in df:
        out["median_latency_s"] = float(df["latency_s"].median())
    rng = random.Random(seed)
    idx = list(range(len(df)))
    boots = {k: [] for k in HEADLINE}
    for _ in range(n_boot):
        m = _metrics(df.iloc[[rng.choice(idx) for _ in idx]])
        for k in HEADLINE:
            boots[k].append(m[k])
    for k, v in boots.items():
        v = sorted(x for x in v if x == x)
        out[f"{k}_ci"] = (v[int(0.025 * len(v))], v[max(int(0.975 * len(v)) - 1, 0)]) if v else (float("nan"),) * 2
    return out


def score_detector(clusters: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """clusters: item_id, anomaly (bool). Does the no-LLM detector flag the expert's artifacts?"""
    df = clusters[["item_id", "anomaly"]].merge(gold[["item_id", "gold_label"]], on="item_id")
    art, real = df[df["gold_label"] == "ARTIFACT"], df[df["gold_label"] == "REAL"]
    flagged = df[df["anomaly"]]
    scored_flagged = flagged[flagged["gold_label"].isin(["ARTIFACT", "REAL"])]

    def rate(num, den):
        return float(num) / den if den else float("nan")

    return {
        "n_flagged": int(len(flagged)),
        "artifact_recall": rate(art["anomaly"].sum(), len(art)),
        "false_flag_on_real": rate(real["anomaly"].sum(), len(real)),
        "flagged_precision": rate((scored_flagged["gold_label"] == "ARTIFACT").sum(), len(scored_flagged)),
        "n_gold_artifact": len(art),
    }


def confusion(preds: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Gold (rows) x model call (columns) counts, UNSURE gold included for transparency."""
    df = preds.merge(gold[["item_id", "gold_label"]], on="item_id").copy()
    df["call"] = df.apply(_call, axis=1)
    return pd.crosstab(df["gold_label"], df["call"]).reindex(
        index=["ARTIFACT", "REAL", "UNSURE"], columns=["ARTIFACT", "REAL", "ABSTAIN", "FAIL"], fill_value=0
    )
