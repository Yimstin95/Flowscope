"""Run the Critic prompt on every evaluation cluster with a free local model (Ollama).

Resumable: results are appended to data/eval/runs/<model>.csv and finished items are skipped.
The prompt/schema are those in src/flowscope/critic.py — frozen before scoring.

Usage:
    ollama serve   # or: brew services start ollama
    .venv/bin/python scripts/run_eval.py --model llama3.1:8b [--limit 3]
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from flowscope.critic import PROMPTS, interpret_anomaly  # noqa: E402
from flowscope.llm_providers import (  # noqa: E402
    GeminiClient, LLMFormatError, OllamaClient, ProviderUnavailable, QuotaExhausted)

try:  # keys live only in the git-ignored .env next to the project root
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

EVAL = os.path.join(ROOT, "data", "eval")
DROP = ["item_id", "sample"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (dry run)")
    ap.add_argument("--eval-dir", default=EVAL,
                    help="folder with clusters.csv (default data/eval; use data/eval_synth for the spike-in set)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v1",
                    help="Critic system prompt variant (v1 = original, v2 = commit-to-a-call policy)")
    ap.add_argument("--provider", choices=["ollama", "gemini"], default="ollama",
                    help="ollama = free local models; gemini = Google AI Studio free tier (GEMINI_API_KEY in .env)")
    ap.add_argument("--list-models", action="store_true", help="print the provider's available models and exit")
    args = ap.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    clusters = pd.read_csv(os.path.join(eval_dir, "clusters.csv"))
    runs_dir = os.path.join(eval_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    suffix = "" if args.prompt == "v1" else "__" + args.prompt
    prefix = "gemini__" if args.provider == "gemini" else ""
    out_path = os.path.join(runs_dir, prefix + args.model.replace(":", "_").replace("/", "_") + suffix + ".csv")
    done = set(pd.read_csv(out_path)["item_id"]) if os.path.exists(out_path) else set()

    client = GeminiClient() if args.provider == "gemini" else OllamaClient()
    if args.list_models:
        print("\n".join(client.available_models()))
        return
    if args.model not in client.available_models():
        sys.exit(f"Model {args.model!r} is not pulled. Run: ollama pull {args.model}")

    rows_todo = clusters[~clusters["item_id"].isin(done)]
    if args.limit:
        rows_todo = rows_todo.head(args.limit)
    print(f"{len(done)} done, {len(rows_todo)} to run with {args.model}", flush=True)

    summaries = {}  # per-sample frame the Critic compares a cluster against
    for key, g in clusters.groupby("sample"):
        s = g.drop(columns=DROP)
        summaries[key] = s.dropna(axis=1, how="all")

    skipped = 0
    for n, (_, row) in enumerate(rows_todo.iterrows(), 1):
        rec = {"item_id": row["item_id"], "model": args.model, "prompt": args.prompt, "flagged": bool(row["anomaly"]),
               "verdict": "", "confidence": "", "key_markers": "", "reasoning": "",
               "ok": True, "error": "", "latency_s": float("nan")}
        t0 = time.perf_counter()
        try:
            res = interpret_anomaly(summaries[row["sample"]], row["cluster"], client=client, model=args.model,
                                    system_prompt=PROMPTS[args.prompt])
            rec.update(verdict=res["verdict"], confidence=res["confidence"],
                       key_markers=json.dumps(res["key_markers"]), reasoning=res["reasoning"])
        except LLMFormatError as e:
            rec.update(ok=False, error=str(e)[:200])
        except ProviderUnavailable as e:
            skipped += 1
            print(f"  [{n}/{len(rows_todo)}] {row['item_id']}: SKIPPED, server error ({e}); not saved, will retry on re-run", flush=True)
            continue
        except QuotaExhausted as e:
            print(f"STOPPED: {e}  ({n - 1} new results saved; re-run the same command to resume)")
            return
        rec["latency_s"] = round(time.perf_counter() - t0, 2)
        pd.DataFrame([rec]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"  [{n}/{len(rows_todo)}] {row['item_id']}: {rec['verdict'] or 'FORMAT-FAIL'} ({rec['latency_s']}s)", flush=True)
    print(f"results -> {out_path}" + (f"  ({skipped} skipped for server errors; re-run to retry)" if skipped else ""))


if __name__ == "__main__":
    main()
