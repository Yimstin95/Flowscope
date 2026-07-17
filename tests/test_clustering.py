"""Validates FlowSOM-style clustering, summary stats, and the pipeline
composition against the synthetic fixture (known 6-population structure).

Uses the synthetic data because it's fast and has ground-truth populations;
UMAP is exercised in a single small-N test since it's the slow step.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from flowscope.clustering import cluster_summary, flowsom_cluster
from flowscope.executor import apply_compensation, arcsinh_transform, load_fcs
from flowscope.pipeline import run_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FCS_PATH = os.path.join(DATA_DIR, "synthetic_pbmc_demo.fcs")
SPILL_PATH = os.path.join(DATA_DIR, "synthetic_pbmc_demo.spillover.csv")


@pytest.fixture(scope="module", autouse=True)
def ensure_synthetic_data():
    if not os.path.exists(FCS_PATH):
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_synthetic_fcs.py")
        subprocess.run([sys.executable, script], check=True)


@pytest.fixture(scope="module")
def transformed_sample():
    sample = load_fcs(FCS_PATH, spillover_path=SPILL_PATH)
    comp = apply_compensation(sample)
    fluor = sample.fluor_channels()
    return arcsinh_transform(comp, channels=fluor, cofactor=150.0)


def test_flowsom_cluster_shapes_and_determinism(transformed_sample):
    r1 = flowsom_cluster(transformed_sample, subsample_n=8000, n_metaclusters=8, seed=7)
    assert len(r1.labels) == len(r1.event_index) == 8000
    assert r1.labels.max() < r1.n_metaclusters
    # Deterministic given the seed.
    r2 = flowsom_cluster(transformed_sample, subsample_n=8000, n_metaclusters=8, seed=7)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    np.testing.assert_array_equal(r1.event_index, r2.event_index)


def test_flowsom_recovers_multiple_populations(transformed_sample):
    """The synthetic data has 6 designed populations; clustering into 8
    metaclusters should recover several well-separated groups, not collapse
    everything into one."""
    r = flowsom_cluster(transformed_sample, subsample_n=10000, n_metaclusters=8, seed=42)
    unique, counts = np.unique(r.labels, return_counts=True)
    assert len(unique) >= 5
    # No single cluster should swallow essentially everything.
    assert counts.max() < 0.9 * len(r.labels)


def test_cluster_summary_frequencies_sum_to_100(transformed_sample):
    r = flowsom_cluster(transformed_sample, subsample_n=10000, n_metaclusters=8, seed=42)
    summary = cluster_summary(transformed_sample, r)
    assert {"cluster", "n_events", "frequency_pct"}.issubset(summary.columns)
    assert summary["n_events"].sum() == 10000
    np.testing.assert_allclose(summary["frequency_pct"].sum(), 100.0, atol=1e-6)


def test_cluster_summary_columns_use_channel_names(transformed_sample):
    r = flowsom_cluster(transformed_sample, subsample_n=6000, n_metaclusters=6, seed=1)
    summary = cluster_summary(transformed_sample, r)
    # Synthetic file has no PnS markers, so columns fall back to detector names.
    assert any("CD3-FITC" in c for c in summary.columns)


def test_run_pipeline_without_umap(transformed_sample):
    out = run_pipeline(
        FCS_PATH,
        spillover_path=SPILL_PATH,
        subsample_n=6000,
        n_metaclusters=6,
        seed=3,
        with_umap=False,
    )
    assert out.umap is None
    assert out.summary["n_events"].sum() == 6000


def test_run_pipeline_with_umap_small(transformed_sample):
    out = run_pipeline(
        FCS_PATH,
        spillover_path=SPILL_PATH,
        subsample_n=2000,
        n_metaclusters=6,
        seed=3,
        with_umap=True,
    )
    assert out.umap is not None
    assert out.umap.shape == (2000, 2)
    assert np.isfinite(out.umap).all()
