"""Generate a synthetic PBMC-like FCS file for pipeline development and testing.

This is placeholder data only. It exists because the real target dataset,
FR-FCM-ZZEB (OMIP-024) on flowrepository.org, currently requires solving a
CAPTCHA (anonymous bulk download) or logging in (individual file download) --
neither of which Claude Code performs. Swap this out for the real accession
once it's manually downloaded into data/raw/ (see README.md).

The synthetic data simulates five roughly PBMC-proportioned populations
(CD4 T cells, CD8 T cells, B cells, NK cells, monocytes) plus a dim
debris/dead-cell population, across a 10-color fluorescence panel loosely
modeled on OMIP-024's lineage markers. A known spillover matrix is applied
to the "true" marker expression to produce realistic cross-channel bleed,
and saved alongside the FCS file so the compensation step has something
real to undo and be checked against.
"""

import csv
import os

import numpy as np
from flowio import create_fcs

RNG = np.random.default_rng(seed=42)

FLUOR_CHANNELS = [
    "CD3-FITC",
    "CD4-PE",
    "CD8-APC",
    "CD19-PerCP",
    "CD14-PE-Cy7",
    "CD56-APC-Cy7",
    "CD16-BV421",
    "HLA-DR-BV510",
    "CD38-BV605",
    "CD45RA-BV711",
]
SCATTER_CHANNELS = ["FSC-A", "SSC-A"]
ALL_CHANNELS = SCATTER_CHANNELS + FLUOR_CHANNELS + ["Time"]

# Mean "true" expression per population, in linear fluorescence units.
# Index order matches FLUOR_CHANNELS.
POPULATIONS = {
    "CD4_T": {
        "fraction": 0.35,
        "scatter": (55000, 20000),
        "means": [8000, 6000, 200, 100, 100, 100, 100, 300, 500, 4000],
    },
    "CD8_T": {
        "fraction": 0.20,
        "scatter": (52000, 21000),
        "means": [7500, 150, 6500, 120, 100, 150, 150, 300, 800, 2500],
    },
    "B_cells": {
        "fraction": 0.10,
        "scatter": (58000, 30000),
        "means": [150, 100, 100, 7000, 120, 100, 100, 5000, 300, 200],
    },
    "NK_cells": {
        "fraction": 0.10,
        "scatter": (60000, 45000),
        "means": [130, 100, 400, 100, 150, 6800, 5500, 200, 200, 200],
    },
    "Monocytes": {
        "fraction": 0.15,
        "scatter": (95000, 85000),
        "means": [200, 250, 150, 150, 7200, 100, 800, 6000, 300, 150],
    },
    "Debris_dead": {
        "fraction": 0.10,
        "scatter": (15000, 60000),
        "means": [300, 300, 300, 300, 300, 300, 300, 300, 1500, 300],
    },
}

N_EVENTS = 20000
NOISE_CV = 0.25  # per-event multiplicative log-normal noise on top of population mean


def build_spillover_matrix(n_channels: int) -> np.ndarray:
    """Mostly-diagonal matrix with modest bleed into neighboring channels,
    mimicking adjacent-fluorochrome spectral overlap."""
    spill = np.eye(n_channels)
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue
            distance = abs(i - j)
            if distance == 1:
                spill[i, j] = 0.12
            elif distance == 2:
                spill[i, j] = 0.04
    return spill


def simulate_true_expression() -> np.ndarray:
    rows = []
    for pop in POPULATIONS.values():
        n = int(round(N_EVENTS * pop["fraction"]))
        means = np.array(pop["means"], dtype=float)
        # log-normal noise per event per channel, centered on the population mean
        noise = RNG.lognormal(mean=0.0, sigma=NOISE_CV, size=(n, len(means)))
        rows.append(means[np.newaxis, :] * noise)
    return np.vstack(rows)


def simulate_scatter(n_per_pop):
    rows = []
    for pop in POPULATIONS.values():
        n = int(round(N_EVENTS * pop["fraction"]))
        fsc_mean, ssc_mean = pop["scatter"]
        fsc = RNG.normal(fsc_mean, fsc_mean * 0.08, size=n)
        ssc = RNG.normal(ssc_mean, ssc_mean * 0.10, size=n)
        rows.append(np.column_stack([fsc, ssc]))
    return np.vstack(rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    true_expression = simulate_true_expression()
    n_total = true_expression.shape[0]

    spill = build_spillover_matrix(len(FLUOR_CHANNELS))
    measured_fluor = true_expression @ spill.T
    measured_fluor = np.clip(measured_fluor, a_min=1.0, a_max=None)

    scatter = simulate_scatter(N_EVENTS)
    time_channel = np.sort(RNG.uniform(0, 600, size=n_total))

    event_data = np.column_stack([scatter, measured_fluor, time_channel]).astype(np.float32)

    fcs_path = os.path.join(out_dir, "synthetic_pbmc_demo.fcs")
    with open(fcs_path, "wb") as fh:
        create_fcs(
            fh,
            event_data.flatten().tolist(),
            channel_names=ALL_CHANNELS,
            metadata_dict={
                "cyt": "Synthetic",
                "experiment name": "FlowScope synthetic placeholder (not real acquisition data)",
            },
        )

    spill_path = os.path.join(out_dir, "synthetic_pbmc_demo.spillover.csv")
    with open(spill_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + FLUOR_CHANNELS)
        for name, row in zip(FLUOR_CHANNELS, spill):
            writer.writerow([name] + list(row))

    print(f"Wrote {n_total} synthetic events to {fcs_path}")
    print(f"Wrote ground-truth spillover matrix to {spill_path}")


if __name__ == "__main__":
    main()
