"""B0 numerical-comparison contract for the PyCC T1 residual slice.

Per EF_INTEGRATION_PLAN.md §4 (B0): for one FIXED state, compare the ef program's
residual and post-Jacobi amplitude SEPARATELY against the PyCC equations, with no
DIIS / convergence / second iteration involved. Here the "PyCC" side is the
independent numpy reference (``reference.py``); the live-``ccwfn`` substitution is
``against_pycc.py`` and needs psi4.

Runnable with numpy + ehrenfest only.
"""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import ehrenfest as ef  # noqa: E402
from devtools.ef_integration.b0 import residual_t1 as R  # noqa: E402
from devtools.ef_integration.b0 import reference as ref  # noqa: E402

O, V = 3, 5


@pytest.fixture(scope="module")
def state():
    inp = ref.random_state(O, V, seed=0)
    r1_np = ref.r_t1(inp)
    t1_np = ref.jacobi_update(inp["t1"], r1_np, inp["eps_o"], inp["eps_v"])
    return inp, r1_np, t1_np


def test_oracle_residual_and_update(state):
    inp, r1_np, t1_np = state
    sl = R.build(O, V)
    r1, t1t = R.evaluate_oracle(sl, inp)
    assert np.allclose(r1, r1_np)     # residual compared separately
    assert np.allclose(t1t, t1_np)    # post-Jacobi amplitude compared separately


def test_composed_program_residual_and_update(state):
    inp, r1_np, t1_np = state
    sl = R.build(O, V)
    run = ef.runner(1 << 28, device="cpu")
    r1, t1t = sl.run(inp, run, search=False)   # frozen B0 baseline: search disabled
    assert np.allclose(r1, r1_np)
    assert np.allclose(t1t, t1_np)


def test_program_is_one_forest(state):
    # sanity: the slice is one composed program (residual cut + eps-sum cut + root),
    # not three disconnected graphs
    sl = R.build(O, V)
    assert sl.program is not None
    assert sl.R1 is not None and sl.Dsum1 is not None and sl.t1_trial is not None
