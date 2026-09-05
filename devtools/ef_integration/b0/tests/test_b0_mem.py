"""B1a-memory tests: the observed-RSS sampler (runnable) + the isolated measurement
(psi4-gated). The sampler test pins that we catch NumPy *native* allocations."""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))


def test_sampler_catches_native_numpy_allocation():
    pytest.importorskip("psutil")
    from devtools.ef_integration.b0.bench_mem import sampled_peak_rss

    def region():
        a = np.ones((10_000_000,), dtype=np.float64)  # ~80 MB, native (tracemalloc-invisible)
        return float(a.sum())

    r = sampled_peak_rss(region)
    assert set(r) == {"baseline_bytes", "peak_bytes", "incremental_peak_bytes"}
    assert r["incremental_peak_bytes"] > 40e6  # RSS sampler sees the native allocation


def test_sampler_fails_closed_without_psutil(monkeypatch):
    import devtools.ef_integration.b0.bench_mem as m
    monkeypatch.setattr(m, "_HAVE_PSUTIL", False)
    with pytest.raises(RuntimeError):
        m.sampled_peak_rss(lambda: None)


def test_measure_isolated_regions():
    pytest.importorskip("psutil")
    pytest.importorskip("psi4")
    pytest.importorskip("pycc")
    pytest.importorskip("ehrenfest")
    from devtools.ef_integration.b0.bench_mem import measure

    rep = measure()
    for side in ("pycc_region", "ef_slice"):
        assert rep[side]["incremental_peak_bytes"] >= 0
