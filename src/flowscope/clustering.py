"""Stage 2 (continued): FlowSOM-style clustering, UMAP, and summary stats.

Pure Python, no LLM calls — deterministic given a fixed random seed. This is
deliberately *not* delegated to a language model; the LLM stages (Panel
Interpreter, Critic, Reporter) reason about the output of this module, they
don't produce the clustering itself.

Why FlowSOM-style rather than scanpy+Leiden: the target data (OMIP-024 PBMC)
has hundreds of thousands of events per sample. A Leiden partition on a KNN
graph of that many cells is slow and memory-hungry, whereas a self-organizing
map scales to large event counts cheaply and is the clustering approach flow
cytometrists actually use (FlowSOM). The pipeline here mirrors FlowSOM:

    1. standardize the compensated+transformed fluorescence channels
    2. train a SOM (grid of nodes) on the events
    3. assign each event to its best-matching SOM node
    4. metacluster the SOM nodes into a smaller number of populations via
       agglomerative (consensus-style) hierarchical clustering
    5. map each event's node -> metacluster

For tractability and to keep UMAP interactive, the analysis runs on a random
subsample of events (frequencies from a uniform subsample are unbiased
estimates of the whole-sample frequencies).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering

from .executor import FCSSample


@dataclass
class ClusterResult:
    """Aligned outputs of the stage-2 clustering, all indexed to the same
    subsampled events (in `event_index`, referring to rows of the input
    sample's events DataFrame)."""

    event_index: np.ndarray          # (n_sub,) indices into the original events
    labels: np.ndarray               # (n_sub,) metacluster id per event
    som_nodes: np.ndarray            # (n_sub,) flat SOM node id per event
    channels: list[str]              # fluorescence channels used, in matrix order
    n_metaclusters: int
    seed: int


def _subsample_indices(n_total: int, subsample_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if subsample_n >= n_total:
        return np.arange(n_total)
    return np.sort(rng.choice(n_total, size=subsample_n, replace=False))


def _standardize(matrix: np.ndarray) -> np.ndarray:
    """Per-channel z-score so no single bright channel dominates the SOM's
    Euclidean distance. Guards against zero-variance channels."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def flowsom_cluster(
    sample: FCSSample,
    channels: list[str] | None = None,
    subsample_n: int = 50000,
    som_grid: tuple[int, int] = (10, 10),
    n_metaclusters: int = 15,
    som_iterations: int = 10000,
    seed: int = 42,
) -> ClusterResult:
    """Run the FlowSOM-style pipeline on `sample` (expected to already be
    compensated and arcsinh-transformed).

    `channels` defaults to the fluorescence channels named in the spillover
    matrix. `n_metaclusters` is the number of populations the SOM nodes are
    collapsed into — a starting point, not a biological ground truth.
    """
    channels = channels or sample.fluor_channels()
    if not channels:
        raise ValueError("No fluorescence channels to cluster on (no spillover/channels given)")

    n_total = sample.events.shape[0]
    idx = _subsample_indices(n_total, subsample_n, seed)
    matrix = sample.events.loc[idx, channels].to_numpy(dtype=float)
    scaled = _standardize(matrix)

    grid_x, grid_y = som_grid
    som = MiniSom(
        grid_x,
        grid_y,
        input_len=scaled.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        random_seed=seed,
    )
    som.random_weights_init(scaled)
    som.train_random(scaled, som_iterations)

    # Vectorized best-matching-unit assignment: distance from each event to
    # every SOM node, then argmin. weights shape (grid_x, grid_y, n_features).
    weights = som.get_weights().reshape(grid_x * grid_y, scaled.shape[1])
    # (n_events, n_nodes) squared distances via the (a-b)^2 expansion
    dists = (
        (scaled**2).sum(axis=1, keepdims=True)
        - 2 * scaled @ weights.T
        + (weights**2).sum(axis=1)
    )
    som_nodes = dists.argmin(axis=1)

    # Metacluster the SOM node prototypes (not the events) — this is what makes
    # FlowSOM fast: hierarchical clustering runs on ~100 nodes, not 50k events.
    n_meta = min(n_metaclusters, weights.shape[0])
    node_meta = AgglomerativeClustering(n_clusters=n_meta).fit_predict(weights)
    labels = node_meta[som_nodes]

    return ClusterResult(
        event_index=idx,
        labels=labels,
        som_nodes=som_nodes,
        channels=channels,
        n_metaclusters=n_meta,
        seed=seed,
    )


def compute_umap(
    sample: FCSSample,
    result: ClusterResult,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """2D UMAP embedding of the *same* subsampled, standardized events the
    clustering used, so the embedding and cluster labels line up row-for-row.
    Returns an (n_sub, 2) array. Imported lazily because umap-learn is heavy.
    """
    import umap  # noqa: PLC0415 — lazy import keeps module import cheap

    matrix = sample.events.loc[result.event_index, result.channels].to_numpy(dtype=float)
    scaled = _standardize(matrix)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
    )
    return reducer.fit_transform(scaled)


def cluster_summary(sample: FCSSample, result: ClusterResult) -> pd.DataFrame:
    """Per-metacluster summary: event count, frequency (% of the subsample),
    and median expression of each fluorescence channel (on the transformed
    scale). Median is used rather than mean because flow marker distributions
    are typically skewed/bimodal. Columns are labeled with marker names
    (PnS) where known.
    """
    matrix = sample.events.loc[result.event_index, result.channels]
    df = matrix.copy()
    df["__cluster__"] = result.labels

    grouped = df.groupby("__cluster__")
    medians = grouped[result.channels].median()
    counts = grouped.size().rename("n_events")
    freq = (counts / counts.sum() * 100).rename("frequency_pct")

    summary = pd.concat([counts, freq, medians], axis=1)
    summary.index.name = "cluster"
    # Label expression columns with marker names for readability downstream.
    summary = summary.rename(columns={ch: sample.label(ch) for ch in result.channels})
    return summary.reset_index()
