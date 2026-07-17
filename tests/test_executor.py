"""Validates FCS parsing, compensation, and arcsinh transform.

Most tests run against a synthetic placeholder dataset (see
scripts/generate_synthetic_fcs.py) with a known ground-truth spillover
matrix, since that can be regenerated on any machine.

The real FR-FCM-ZZEB (OMIP-024) data was manually downloaded (FlowRepository
gates automated downloads behind a CAPTCHA/login) into
data/raw/FR-FCM-ZZEB/ and is git-ignored due to its size (~395MB). The
real-data tests below are skipped if that folder isn't present.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from flowscope.executor import apply_compensation, arcsinh_transform, load_fcs

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FCS_PATH = os.path.join(DATA_DIR, "synthetic_pbmc_demo.fcs")
SPILL_PATH = os.path.join(DATA_DIR, "synthetic_pbmc_demo.spillover.csv")

REAL_SAMPLE_PATH = os.path.join(DATA_DIR, "FR-FCM-ZZEB", "Sample1.fcs")
REAL_MARKERS = [
    "CD3", "CD4", "CD8", "CD19", "CD56", "CD14", "CD16", "CD25", "CD38",
    "CD57", "CD45RA", "CCR7", "HLADR", "NKG2C", "gdTCR", "Vdelta2", "C127", "AViD",
]

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


@pytest.fixture(scope="module", autouse=True)
def ensure_synthetic_data():
    if not os.path.exists(FCS_PATH):
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_synthetic_fcs.py")
        subprocess.run([sys.executable, script], check=True)


def test_load_fcs_parses_expected_shape_and_channels():
    sample = load_fcs(FCS_PATH, spillover_path=SPILL_PATH)
    assert sample.events.shape[0] == 20000
    assert "FSC-A" in sample.channel_names
    assert "Time" in sample.channel_names
    for ch in FLUOR_CHANNELS:
        assert ch in sample.channel_names
    assert (sample.events[FLUOR_CHANNELS].to_numpy() > 0).all()


def test_load_fcs_reads_external_spillover():
    sample = load_fcs(FCS_PATH, spillover_path=SPILL_PATH)
    assert sample.spillover is not None
    assert list(sample.spillover.columns) == FLUOR_CHANNELS
    np.testing.assert_allclose(np.diag(sample.spillover.to_numpy()), 1.0)


def test_compensation_reduces_cross_channel_correlation():
    """Neighboring channels in the synthetic panel were given deliberate
    spillover (see build_spillover_matrix). Compensation should reduce the
    correlation between a channel and its neighbor's dedicated population,
    compared to the raw (uncompensated) data."""
    sample = load_fcs(FCS_PATH, spillover_path=SPILL_PATH)
    compensated = apply_compensation(sample)

    # CD4-PE (channel 1) neighbors CD3-FITC (channel 0) and CD8-APC (channel 2)
    # in the synthetic spillover matrix. The debris/dead population has flat,
    # low expression everywhere, so use the CD8_T-dominant tail (high CD8-APC)
    # to check that CD4-PE spillover from CD8-APC is removed.
    raw_cd8 = sample.events["CD8-APC"].to_numpy()
    raw_cd4 = sample.events["CD4-PE"].to_numpy()
    comp_cd8 = compensated.events["CD8-APC"].to_numpy()
    comp_cd4 = compensated.events["CD4-PE"].to_numpy()

    high_cd8_mask = raw_cd8 > np.percentile(raw_cd8, 90)
    raw_corr = np.corrcoef(raw_cd8[high_cd8_mask], raw_cd4[high_cd8_mask])[0, 1]
    comp_corr = np.corrcoef(comp_cd8[high_cd8_mask], comp_cd4[high_cd8_mask])[0, 1]

    assert comp_corr < raw_corr


def test_compensation_requires_spillover():
    sample = load_fcs(FCS_PATH)  # no spillover_path, none embedded
    assert sample.spillover is None
    with pytest.raises(ValueError):
        apply_compensation(sample)


def test_arcsinh_transform_compresses_dynamic_range():
    sample = load_fcs(FCS_PATH, spillover_path=SPILL_PATH)
    compensated = apply_compensation(sample)
    transformed = arcsinh_transform(compensated, channels=FLUOR_CHANNELS, cofactor=150.0)

    raw_range = compensated.events[FLUOR_CHANNELS].to_numpy().ptp()
    transformed_range = transformed.events[FLUOR_CHANNELS].to_numpy().ptp()
    assert transformed_range < raw_range

    # Scatter/time channels must be untouched by the fluorescence-only transform
    np.testing.assert_array_equal(
        transformed.events["FSC-A"].to_numpy(), compensated.events["FSC-A"].to_numpy()
    )


requires_real_data = pytest.mark.skipif(
    not os.path.exists(REAL_SAMPLE_PATH),
    reason="Real FR-FCM-ZZEB data not present at data/raw/FR-FCM-ZZEB/Sample1.fcs "
    "(manually downloaded, git-ignored) — skipping real-data validation",
)


@requires_real_data
def test_load_real_zzeb_sample_parses_expected_markers():
    sample = load_fcs(REAL_SAMPLE_PATH)
    assert sample.events.shape[0] > 0
    for marker in REAL_MARKERS:
        assert marker in sample.metadata.values() or any(
            marker in str(v) for v in sample.metadata.values()
        ), f"expected marker {marker} in FCS metadata (PnS fields)"


@requires_real_data
def test_real_zzeb_sample_has_embedded_spillover():
    sample = load_fcs(REAL_SAMPLE_PATH)
    assert sample.spillover is not None
    n = sample.spillover.shape[0]
    assert sample.spillover.shape == (n, n)
    np.testing.assert_allclose(np.diag(sample.spillover.to_numpy()), 1.0)


@requires_real_data
def test_real_zzeb_sample_full_pipeline_runs():
    """End-to-end smoke test: parse -> compensate -> transform on a real
    18-color acquisition, not just synthetic data."""
    sample = load_fcs(REAL_SAMPLE_PATH)
    compensated = apply_compensation(sample)
    fluor_channels = list(sample.spillover.columns)
    transformed = arcsinh_transform(compensated, channels=fluor_channels, cofactor=150.0)

    assert transformed.events.shape == sample.events.shape
    raw_range = compensated.events[fluor_channels].to_numpy().ptp()
    transformed_range = transformed.events[fluor_channels].to_numpy().ptp()
    assert transformed_range < raw_range
    assert np.isfinite(transformed.events[fluor_channels].to_numpy()).all()
