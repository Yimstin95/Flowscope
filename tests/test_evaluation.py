"""Hand-computable checks of the evaluation definitions."""

import pandas as pd

from flowscope.evaluation import confusion, score_critic, score_detector

GOLD = pd.DataFrame({
    "item_id": list("abcdefgh"),
    "gold_label": ["ARTIFACT", "ARTIFACT", "ARTIFACT", "REAL", "REAL", "REAL", "UNSURE", "UNSURE"],
})
PREDS = pd.DataFrame({
    "item_id": list("abcdefgh"),
    "verdict": ["likely_artifact", "likely_biological", "needs_confirmation",
                "likely_biological", "likely_artifact", "likely_biological",
                "needs_confirmation", "likely_artifact"],
    "ok": [True] * 8,
    "latency_s": [1.0] * 8,
})


def test_critic_metrics_match_hand_counts():
    r = score_critic(PREDS, GOLD, n_boot=50)
    assert r["n_scored"] == 6 and r["n_gold_artifact"] == 3 and r["n_gold_real"] == 3
    assert abs(r["agreement"] - 3 / 6) < 1e-9              # a, d, f match
    assert abs(r["agreement_confident"] - 3 / 5) < 1e-9    # c abstained -> 5 committed
    assert abs(r["abstention_rate"] - 1 / 6) < 1e-9
    assert abs(r["dangerous_miss"] - 1 / 3) < 1e-9         # b: artifact called biological
    assert abs(r["lost_population"] - 1 / 3) < 1e-9        # e: real called artifact
    assert abs(r["abstain_on_unsure"] - 1 / 2) < 1e-9      # g abstained, h did not
    lo, hi = r["agreement_ci"]
    assert lo <= r["agreement"] <= hi


def test_format_failures_are_not_silently_dropped():
    preds = PREDS.copy()
    preds.loc[0, "ok"] = False                              # item a failed
    r = score_critic(preds, GOLD, n_boot=20)
    assert abs(r["format_failure_rate"] - 1 / 6) < 1e-9
    assert abs(r["agreement"] - 2 / 6) < 1e-9               # a no longer counts as a match


def test_detector_metrics():
    clusters = pd.DataFrame({"item_id": list("abcdef"),
                             "anomaly": [True, True, False, True, False, False]})
    r = score_detector(clusters, GOLD)
    assert abs(r["artifact_recall"] - 2 / 3) < 1e-9
    assert abs(r["false_flag_on_real"] - 1 / 3) < 1e-9      # d flagged but REAL
    assert abs(r["flagged_precision"] - 2 / 3) < 1e-9


def test_confusion_shape():
    c = confusion(PREDS, GOLD)
    assert c.loc["ARTIFACT", "ARTIFACT"] == 1 and c.loc["ARTIFACT", "REAL"] == 1
    assert c.loc["UNSURE"].sum() == 2
