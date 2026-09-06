"""Independent numpy reference for PyCC's CCSD T1 residual + Jacobi update.

Hand-transcribed from ``pycc/ccwfn.py`` (``_r_T1`` CCSD path, lines ~754-760, and
the update ``self.t1 += r1/Dia``), NOT from ehrenfest's DF-CCSD example — this is
the "existing PyCC equations" side of the B0 numerical-comparison contract. In a
psi4-equipped environment ``against_pycc.py`` replaces this with a live ``ccwfn``;
here it is the oracle the ef program is checked against.

PyCC conventions used:
  * t1 indexed [i, a] with shape (o, v); t2 indexed [i, j, a, b].
  * f_ai is F[v, o] transposed to [i, a] (the constant term).
  * Fae (v, v), Fmi (o, o), Fme (o, v) are the CCSD intermediates, taken as inputs
    (this slice isolates the residual assembly + update boundary, per the plan).
  * L[o,v,v,o], ERI[o,v,v,v], L[o,o,v,o] are the required integral blocks.
  * Dia[i, a] = eps_o[i] - eps_v[a]  (orbital-energy denominator).

The ``2*t2 - t2.swapaxes(2,3)`` factors are expanded into two einsum terms with
swapped labels rather than a materialized transpose.
"""

from __future__ import annotations

import numpy as np


def r_t1(inp: dict) -> np.ndarray:
    """PyCC CCSD singles residual r1[i, a] from the input tensors in ``inp``."""
    t1, t2 = inp["t1"], inp["t2"]
    Fae, Fmi, Fme = inp["Fae"], inp["Fmi"], inp["Fme"]
    L_ovvo, ERI_ovvv, L_oovo = inp["L_ovvo"], inp["ERI_ovvv"], inp["L_oovo"]
    r = inp["f_ai"].copy()
    r += np.einsum("ie,ae->ia", t1, Fae)
    r -= np.einsum("mi,ma->ia", Fmi, t1)
    # (2 t2 - t2.swapaxes(2,3)) contracted with Fme
    r += 2.0 * np.einsum("imae,me->ia", t2, Fme) - np.einsum("imea,me->ia", t2, Fme)
    r += np.einsum("nf,nafi->ia", t1, L_ovvo)
    # (2 t2 - t2.swapaxes(2,3)) contracted with ERI[o,v,v,v]
    r += 2.0 * np.einsum("mief,maef->ia", t2, ERI_ovvv) - np.einsum("mife,maef->ia", t2, ERI_ovvv)
    r -= np.einsum("mnae,nmei->ia", t2, L_oovo)
    return r


def jacobi_update(t1: np.ndarray, r1: np.ndarray, eps_o: np.ndarray, eps_v: np.ndarray) -> np.ndarray:
    """t1_new[i, a] = t1[i, a] + r1[i, a] / (eps_o[i] - eps_v[a])."""
    Dia = eps_o[:, None] - eps_v[None, :]
    return t1 + r1 / Dia


def random_state(o: int, v: int, seed: int = 0) -> dict:
    """A fixed, self-consistent random PyCC-shaped state (well-separated eps so the
    denominator is safely non-singular)."""
    rng = np.random.default_rng(seed)
    eps_o = np.sort(rng.uniform(-1.0, -0.3, size=o))
    eps_v = np.sort(rng.uniform(0.3, 1.5, size=v))
    return {
        "t1": rng.standard_normal((o, v)) * 1e-2,
        "t2": rng.standard_normal((o, o, v, v)) * 1e-2,
        "f_ai": rng.standard_normal((o, v)) * 1e-3,
        "Fae": rng.standard_normal((v, v)),
        "Fmi": rng.standard_normal((o, o)),
        "Fme": rng.standard_normal((o, v)),
        "L_ovvo": rng.standard_normal((o, v, v, o)),
        "ERI_ovvv": rng.standard_normal((o, v, v, v)),
        "L_oovo": rng.standard_normal((o, o, v, o)),
        "eps_o": eps_o,
        "eps_v": eps_v,
    }
