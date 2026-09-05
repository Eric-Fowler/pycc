"""B0 T2 slice — staged local checks against the faithful numpy mirror.

Staged per the plan so a failure localizes:
  1. each EF intermediate vs the mirror's builder (localized diagnosis);
  2. the composed-program r2 vs the mirror (oracle + program, search=False);
  3. t2_trial vs the mirror update.

The mirror (``reference_t2.py``) shares a hand-translation with the EF slice, so it
is the LOCAL oracle, not the authority; production-PyCC equivalence is
``test_b0_t2_against_pycc.py`` (needs psi4). Runnable with numpy + ehrenfest.
"""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import ehrenfest as ef  # noqa: E402
from devtools.ef_integration.b0 import residual_t2 as R2mod  # noqa: E402
from devtools.ef_integration.b0 import reference_t2 as ref  # noqa: E402

O, V = 3, 5

_INTERMEDIATES = [
    ("Fae", "ae", ref.build_Fae),
    ("Fmi", "mi", ref.build_Fmi),
    ("Fme", "me", ref.build_Fme),
    ("Wmnij", "mnij", ref.build_Wmnij),
    ("Wmbej", "mbej", ref.build_Wmbej),
    ("Wmbje", "mbje", ref.build_Wmbje),
    ("Zmbij", "mbij", ref.build_Zmbij),
]


@pytest.fixture(scope="module")
def fixture():
    s = ref.random_state(O, V, seed=0)
    r2 = ref.r_t2(s)
    t2t = ref.jacobi_update(s["t2"], r2, s["eps_o"], s["eps_v"])
    return s, r2, t2t, R2mod.build(O, V)


@pytest.mark.parametrize("name,letters,builder", _INTERMEDIATES)
def test_intermediate_matches_mirror(fixture, name, letters, builder):
    s, _r2, _t2t, sl = fixture
    got = sl.eval_intermediate(name, s, letters)
    assert np.allclose(got, builder(s)), f"{name} mismatch"


def test_composed_residual_and_update_oracle(fixture):
    s, r2, t2t, sl = fixture
    r2_o, t2t_o = sl.evaluate_oracle(s)
    assert np.allclose(r2_o, r2)
    assert np.allclose(t2t_o, t2t)


def test_composed_residual_and_update_program(fixture):
    s, r2, t2t, sl = fixture
    r2_p, t2t_p = sl.run(s, ef.runner(1 << 30, device="cpu"), search=False)
    assert np.allclose(r2_p, r2)
    assert np.allclose(t2t_p, t2t)


def test_residual_is_p_symmetric(fixture):
    _s, r2, _t2t, _sl = fixture
    assert np.allclose(r2, r2.swapaxes(0, 1).swapaxes(2, 3))  # P(ij)(ab)


def test_dijab_mode_matches_mirror(fixture):
    """The B1a fairness build (denom='dijab', precomputed Dijab leaf) must produce the
    same residual and update as the eps-mode correctness build / the mirror."""
    s, r2, t2t, _sl = fixture
    sl_d = R2mod.build(O, V, denom="dijab")
    r2_o, t2t_o = sl_d.evaluate_oracle(s)          # s already carries "Dijab"
    assert np.allclose(r2_o, r2)
    assert np.allclose(t2t_o, t2t)
    r2_p, t2t_p = sl_d.run(s, ef.runner(1 << 30, device="cpu"), search=False)
    assert np.allclose(r2_p, r2)
    assert np.allclose(t2t_p, t2t)
