"""Validates the second real dataset (data/raw/flowio-demo/), unlike the real
FR-FCM-ZZEB tests in test_executor.py this file is NOT git-ignored -- it's
small (4MB, BSD-3 licensed) and committed directly to the repo -- so these
tests always run, in CI and on a fresh clone alike, with no skip condition.
"""

import os

from flowscope.pipeline import run_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FCS_PATH = os.path.join(DATA_DIR, "flowio-demo", "pbmc_13color_facsaria.fcs")

EXPECTED_MARKERS = {
    "KI67", "CD3", "CD28", "CD45RO", "CD8", "CD4", "CD57",
    "CCR5", "CD19", "CD27", "CCR7", "CD127",
}


def test_fixture_is_present():
    assert os.path.exists(FCS_PATH), (
        f"{FCS_PATH} is missing -- it should be committed to the repo, not "
        "generated or downloaded at test time."
    )


def test_pipeline_runs_end_to_end_on_the_flowio_demo_sample():
    out = run_pipeline(FCS_PATH, subsample_n=10000, n_metaclusters=10, seed=1, with_umap=False)

    assert out.sample.events.shape[0] == 65016
    assert len(out.result.event_index) == 10000
    assert out.summary["n_events"].sum() == 10000

    markers_present = set(out.sample.markers.values())
    # Strip stray whitespace some real-world FCS files carry on marker names.
    markers_present = {m.strip() for m in markers_present if m and m.strip()}
    missing = EXPECTED_MARKERS - markers_present
    assert not missing, f"expected markers missing from parsed panel: {missing}"


def test_pipeline_has_embedded_spillover_no_external_csv_needed():
    out = run_pipeline(FCS_PATH, subsample_n=5000, n_metaclusters=8, seed=1, with_umap=False)
    assert out.sample.spillover is not None
    n = out.sample.spillover.shape[0]
    assert out.sample.spillover.shape == (n, n)
