"""Live B0 production-PyCC comparison — skipped where psi4 is unavailable.

This is the test that catches a transcription error common to ``reference.py`` and
``residual_t1.py``: it compares the ef slice against PyCC's *production* residual +
update, not the numpy transcription. Requires psi4 + the full pycc stack, so it
skips cleanly in the lightweight environment.
"""

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))


def test_ef_t1_slice_matches_production_pycc():
    pytest.importorskip("psi4")
    pytest.importorskip("pycc")
    pytest.importorskip("ehrenfest")
    from devtools.ef_integration.b0.against_pycc import build_pycc_state, compare_against_pycc

    cc = build_pycc_state(maxiter=3)
    res = compare_against_pycc(cc)
    assert res["residual_ok"], f"T1 residual mismatch: {res}"
    assert res["update_ok"], f"post-Jacobi t1 mismatch: {res}"
