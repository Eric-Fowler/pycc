"""Smoke test for the B1a T2 benchmark harness — skipped without psi4.

Verifies the harness runs end-to-end and reports the required lifecycle fields; it
does NOT assert or interpret timings. The harness's own gate (production
correctness first) is exercised via bench_t2.main; here we call benchmark() after a
correctness check so a green run proves the wiring.
"""

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))


def test_bench_reports_lifecycle_fields():
    pytest.importorskip("psi4")
    pytest.importorskip("pycc")
    pytest.importorskip("ehrenfest")
    from devtools.ef_integration.b0.against_pycc import build_pycc_state
    from devtools.ef_integration.b0.against_pycc_t2 import compare_against_pycc
    from devtools.ef_integration.b0.bench_t2 import benchmark

    cc = build_pycc_state(maxiter=3)
    gate = compare_against_pycc(cc)      # production gate before any timing
    assert gate["residual_ok"] and gate["update_ok"]

    rep = benchmark(cc, warm=3, pycc_warm=3)
    for key in ("ef_build_s", "ef_runner_ctor_s", "ef_precompile_s",
                "ef_first_exec_s", "ef_warm_s", "pycc_t2_region_s",
                "warm_ratio_ef_over_pycc"):
        assert key in rep
