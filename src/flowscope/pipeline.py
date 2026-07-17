"""Convenience composition of the deterministic stage-2 pipeline.

Keeps the Streamlit app (and future FastAPI routes) thin: one call runs
parse -> compensate -> transform -> cluster -> UMAP -> summary and returns the
aligned results. No LLM calls happen here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clustering import ClusterResult, cluster_summary, compute_umap, flowsom_cluster
from .executor import FCSSample, apply_compensation, arcsinh_transform, load_fcs


@dataclass
class PipelineOutput:
    sample: FCSSample            # compensated + transformed sample
    result: ClusterResult
    summary: pd.DataFrame        # per-cluster frequency + median marker expression
    umap: np.ndarray | None      # (n_sub, 2) embedding, or None if skipped


def run_pipeline(
    fcs_path: str,
    spillover_path: str | None = None,
    cofactor: float = 150.0,
    subsample_n: int = 20000,
    n_metaclusters: int = 15,
    seed: int = 42,
    with_umap: bool = True,
) -> PipelineOutput:
    sample = load_fcs(fcs_path, spillover_path=spillover_path)
    if sample.spillover is None:
        raise ValueError(
            f"{fcs_path} has no embedded spillover and no spillover_path was given — "
            "cannot compensate. Supply a spillover CSV."
        )
    fluor = sample.fluor_channels()
    compensated = apply_compensation(sample)
    transformed = arcsinh_transform(compensated, channels=fluor, cofactor=cofactor)

    result = flowsom_cluster(
        transformed,
        subsample_n=subsample_n,
        n_metaclusters=n_metaclusters,
        seed=seed,
    )
    summary = cluster_summary(transformed, result)
    umap = compute_umap(transformed, result, seed=seed) if with_umap else None

    return PipelineOutput(sample=transformed, result=result, summary=summary, umap=umap)
