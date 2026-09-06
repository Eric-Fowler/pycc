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


def test_run_isolated_raises_on_child_crash():
    """A child that exits without sending a result must make the parent RAISE, not
    hang forever (the whole point for an OOM-prone memory benchmark)."""
    from devtools.ef_integration.b0.bench_mem import run_isolated, _crash_target
    with pytest.raises(RuntimeError):
        run_isolated(_crash_target, timeout=30.0)


def test_run_isolated_returns_child_result():
    from devtools.ef_integration.b0 import bench_mem as m
    assert m.run_isolated(m._echo_target, value=41) == 42


def test_ef_planned_is_deferred_not_budget():
    from devtools.ef_integration.b0.bench_mem import ef_planned_bytes

    class FakeRun:
        def budget(self):
            return 4.0e9   # remaining capacity — must NOT be reported as planned memory

    out = ef_planned_bytes(FakeRun())
    assert out["planned_bytes"] is None   # deferred, never runner.budget()


def test_measure_isolated_regions():
    pytest.importorskip("psutil")
    pytest.importorskip("psi4")
    pytest.importorskip("pycc")
    pytest.importorskip("ehrenfest")
    from devtools.ef_integration.b0.bench_mem import measure

    rep = measure(threads=1)
    for side in ("pycc_region", "ef_slice"):
        assert rep[side]["incremental_peak_bytes"] >= 0
        assert rep[side]["thread_policy"] == "controlled"
    assert rep["ef_planned"]["planned_bytes"] is None
