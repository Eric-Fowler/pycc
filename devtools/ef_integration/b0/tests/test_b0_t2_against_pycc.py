"""Live B0 T2 production comparison — skipped where psi4 is unavailable.

The authority for the T2 slice: EF intermediates vs PyCC's own build_*, and the
composed r2 / t2_trial vs the production cc.residuals path. Requires psi4.
"""

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))


def test_ef_t2_slice_matches_production_pycc():
    pytest.importorskip("psi4")
    pytest.importorskip("pycc")
    pytest.importorskip("ehrenfest")
    from devtools.ef_integration.b0.against_pycc import build_pycc_state
    from devtools.ef_integration.b0.against_pycc_t2 import compare_against_pycc

    cc = build_pycc_state(maxiter=3)
    res = compare_against_pycc(cc)
    bad = {k: v for k, v in res["intermediates"].items() if not v["ok"]}
    assert not bad, f"intermediate mismatch: {bad}"
    assert res["residual_ok"], f"T2 residual mismatch: {res}"
    assert res["update_ok"], f"post-Jacobi t2 mismatch: {res}"
