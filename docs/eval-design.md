# FlowScope-Eval — how much should an analyst trust the AI Critic?

Date: 2026-10-06 · Status: in progress

## Question

FlowScope's Critic (an LLM) reads a cluster's marker profile and says whether it looks like a
**technical artifact** (dead cells / debris, doublets, compensation spill-over) or a **biologically
real** population, or that it **needs confirmation**. Until now this was only eyeballed on one sample.
This evaluation measures it against an expert, and checks whether it can run **for free**.

## Design

- **Items:** every cluster FlowScope produces on three real samples (OMIP-024 `Sample1`, `Sample2`,
  and the 13-colour FlowIO demo file), fixed seeds, committed as `data/eval/clusters.csv`
  so the evaluation reproduces without the 395 MB raw files.
- **Gold labels:** one analyst (the author, 3.5 years at a cell-therapy CRO) labels a **random
  subset of 30 of the 55 clusters** (11 / 11 / 8 per sample, fixed seed, listed in
  `data/eval/label_subset.json`) as `ARTIFACT` / `REAL` / `UNSURE` *before* seeing any model output
  or detector flag. The subset was chosen because labeling all 55 carefully takes about an hour; the
  cost is wider confidence intervals, which are reported. Labels are committed as
  `data/eval/gold_labels.csv`, together with the **seconds spent per item** (`dwell_s`) so the
  labeling effort can be audited. (A first, unusably fast labeling pass — 55 labels in 15 s — was
  discarded and redone; see the project log.)
- **Systems compared:** the same Critic prompt and schema, run through different models:
  free local models (Ollama) and, if available, the paid Claude model the project was built on.
- **Two layers are scored separately:**
  1. *Detector* (pure Python, no LLM): of the gold-`ARTIFACT` clusters, how many does it flag?
     how many gold-`REAL` clusters does it flag by mistake?
  2. *Interpreter* (LLM): on **all** clusters (not only flagged ones, otherwise there are too few
     items), verdict vs gold.

## Metrics (plain-language names in the report)

| Metric | Meaning |
|---|---|
| Agreement | share of clusters where the verdict equals the gold label (excluding gold `UNSURE`) |
| **Dangerous miss** | gold `ARTIFACT` called `likely_biological` — an artifact would be reported as biology |
| Lost population | gold `REAL` called `likely_artifact` — a real population would be discarded |
| Abstention rate | share answered `needs_confirmation` (not an error; reported with agreement-when-confident) |
| Format failure | the model did not return valid structured output |
| Speed / cost | seconds per cluster; $0 for local models |

All with bootstrap 95% intervals. n is small (~55 clusters), so results are indicative, not definitive.

## Second ground-truth track: known-answer ("spike-in") test, no human labels

**Why:** a literature/database check found no open dataset that labels clusters as *technical artifact*
vs *real population*. Public flow/mass-cytometry benchmarks (e.g. the HDCytoData collection: Levine,
Samusik, Nilsson_rare, Mosmann_rare, FlowCAP) provide **manually gated cell-type labels only**; debris,
dead cells and doublets are not labelled there. They could test "does a cluster match an expert-gated
population", which is a different question from the one the Critic answers.

**What we do instead:** `scripts/make_spikein_eval.py` simulates 5 PBMC-like files (20,000 events each) in
which every event carries a hidden true class: 7 real populations (incl. two rare coherent ones) and 5 artifact
types (dead cells, debris, T:B doublets, T:monocyte doublets, antibody aggregates). After FlowScope clusters a
file, each cluster's truth is fixed by a rule set **before any model is scored**: ARTIFACT if >= 80% of its
events are artifact classes, REAL if >= 80% are real classes, else UNSURE (excluded).
Truth lives in `data/eval_synth/gold_truth.csv`, never in the file the model sees.

**Reading the result honestly:** the simulation is cleaner than real data (all 60 clusters came out 100% pure,
one cluster per simulated population), so this track measures whether a model can *read* an obvious profile,
not how it copes with messy real clusters. It is reported as a controlled sanity test, next to (not instead of)
the expert-labelled real-data track.

### Finding 1 (before any tuning): the original prompt makes small local models abstain on everything

With the original Critic prompt (v1), `gemma3:4b`, `llama3.1:8b` and `qwen2.5:7b` answered
`needs_confirmation` on **all** clusters of both the simulated set (60/60) and the real-data set (55/55):
0% agreement, 100% abstention, no format failures. This is not a scoring bug (raw outputs inspected). The
v1 text tells the model that abstaining is "the correct answer under uncertainty"; the small models take it
literally. The prompt was written for Claude.

### Protocol for fixing it without cheating

* Prompt **v2** changes only the decision-policy sentences ("commit to the more likely reading; express doubt
  through the confidence field; abstain only when two readings are about equally likely"). The artifact/biology
  descriptions are word-for-word identical to v1 (checked by diff), so no new domain knowledge is added.
* **dev** set (`data/eval_synth`, seeds 100-104) is used to look at v2's behaviour and, if needed, allow one
  further policy tweak. **test** set (`data/eval_synth_test`, seeds 200-204) was generated before v2 was run,
  and no model has been run on it; it is scored once, with the frozen prompt, and is the headline number.
* The simulator and the prompt were written by the same person; the prompt therefore deliberately contains
  no hints derived from the simulator's numbers.

## Results — held-out spike-in TEST set (scored once, 2026-10-08)

63 simulated clusters (37 real / 26 artifact, all 100% pure), original Critic prompt **v1**, temperature 0.
Model and prompt choices were made on the separate dev set; the test set had never been run before.

| Model (where it ran) | Agreement (95% CI) | Artifact called real ("dangerous miss") | Real called artifact | Abstained | s / cluster |
|---|---|---|---|---|---|
| gemma-4-31b-it (Gemini API, free tier) | **97% (92–100%)** | 4% (1/26: a T:monocyte doublet) | 3% (1/37: a B-cell cluster) | 0% | 42 |
| gemini-3.5-flash-lite (Gemini API, free tier) | 84% (75–92%) | 4% (1/26) | 16% (6/37: mostly the two rare populations) | 5% | 7 |
| gemma3:4b (local, Ollama) | 0% | 0% | 0% | 100% | 16 |
| *Anomaly detector alone (no AI)* | *catches 38% of artifacts, flags 51% of real clusters* | | | | |

Reading it:
* Model size decided the outcome: the same model family went from "abstains on everything" (4B, local)
  to 97% (31B). 4–8B local models never discriminated under any prompt variant tried on dev.
* The hardest cases are the ones that are hard at the bench too: **T:monocyte doublets** (CD3+CD14+) and
  **rare real populations** (plasmablast-like, NKT-like), which were absent from the prompt's examples.
* Practical rule this supports: AI calls on doublet-like and rare clusters must be confirmed by an analyst.

Limits (must accompany any number above): simulated data, cleaner than real samples; dev/test from the
same simulator written by the same person; n = 63 (see CIs); free-tier cloud models may use submitted data,
so they must not receive proprietary data; 31B cannot run on a 16 GB laptop. Real-data, expert-labelled
evaluation remains open (`data/eval/`).

### Change made after scoring (disclosed)

The per-cluster message sent to the model used to say "deviation from the cross-cluster **median**",
while `cluster_marker_profile` computes deviations from the cross-cluster **mean**. On 2026-10-08 the
wording was corrected to "mean". All numbers above were produced with the old wording; they were not
re-run. The numbers themselves (values, signs, order) sent to the model did not change.

## Out of scope

Clinical use; claims beyond these 3 samples; tuning the Critic prompt on the gold labels
(the prompt is frozen before scoring; any later prompt change is reported as a separate, labelled run).

## Free-to-run constraint

Local Ollama models (no key, no cost). The public Streamlit deployment runs the deterministic
pipeline live and shows the **pre-computed** evaluation results, so visitors need no key
and the host pays nothing for model calls.
