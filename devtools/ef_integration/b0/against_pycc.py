"""Live B0 check: the ef T1-residual program vs a real PyCC ``ccwfn`` state.

Completes the B0 numerical-comparison contract against *production* PyCC (not the
numpy transcription in ``reference.py``, which shares its hand-translation with the
ef slice and so cannot catch a mistake common to both). Requires the full PyCC
stack (psi4); it is API-complete and importable, and the pytest test skips cleanly
where psi4 is unavailable (as in the lightweight CI/dev environment).

Contract enforced here, per EF_INTEGRATION_PLAN.md §4 (B0):
  * one deterministic PyCC CCSD state (fixed geometry/basis, fixed iteration count);
  * intermediates ``Fae``/``Fmi``/``Fme`` from PyCC's own builders, exact signatures;
  * PyCC's actual ``cc.Dia`` used for the reference update, and the eps-derived
    denominator the ef slice builds is ASSERTED equal to it;
  * ``r1`` from the production residual path (``cc.residuals``);
  * the SAME tensors fed to the ef slice;
  * ``r1`` and ``t1_trial`` compared SEPARATELY, no DIIS / convergence.
"""

from __future__ import annotations

import numpy as np


def build_pycc_state(geometry: str | None = None, basis: str = "cc-pVDZ", maxiter: int = 3):
    """A deterministic finite-Jacobi-iteration CCSD state; convergence and DIIS
    deliberately disabled/irrelevant.

    B0 is about one fixed *algebraic* state, not solver acceleration, so DIIS is
    turned off (``max_diis=0``) and the loop is stopped after a few plain Jacobi
    steps (``maxiter`` small) — the point is a reproducible ``(t1, t2)`` decoupled
    from DIIS history, easy to reason about when a residual mismatch is being
    debugged, and reusable verbatim by the T2 comparison. Canonical RHF has
    F[o,v]=0 (so the MP2 seed gives t1=0); a few Jacobi steps make t1 non-trivial.

    ``solve_cc`` mutates ``cc.t1``/``cc.t2`` in place; its return value is
    deliberately ignored (for the intentionally non-converged finite-iteration case
    it may fall off the end without returning an energy). The mutated amplitudes are
    the state we need. With ``max_diis=0`` PyCC's DIIS returns the input amplitudes
    unchanged, so the state is a pure finite-Jacobi one.
    """
    import psi4
    import pycc

    psi4.core.clean()
    psi4.set_memory("2 GB")
    psi4.core.set_output_file("pycc_b0.out", False)
    geometry = geometry or "O\nH 1 0.96\nH 1 0.96 2 104.5"
    psi4.geometry(geometry)
    psi4.set_options({"basis": basis, "scf_type": "pk", "e_convergence": 1e-10,
                      "d_convergence": 1e-10})
    _, wfn = psi4.energy("SCF", return_wfn=True)
    cc = pycc.CCwfn(wfn, model="CCSD")
    cc.solve_cc(e_conv=1e-8, r_conv=1e-7, maxiter=maxiter, max_diis=0)  # DIIS off: fixed Jacobi state
    return cc


def extract_inputs(cc) -> dict:
    """The ef slice's input leaves, pulled from a live ``ccwfn`` (all numpy)."""
    o, v = cc.o, cc.v
    F, ERI, L = cc.H.F, cc.H.ERI, cc.H.L
    t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
    Fae = np.asarray(cc.build_Fae(o, v, F, L, t1, t2))
    Fmi = np.asarray(cc.build_Fmi(o, v, F, L, t1, t2))
    Fme = np.asarray(cc.build_Fme(o, v, F, L, t1))
    return {
        "t1": t1,
        "t2": t2,
        "f_ai": np.asarray(F[v, o]).swapaxes(0, 1),   # the r_T1 constant term, [i, a]
        "Fae": Fae,
        "Fmi": Fmi,
        "Fme": Fme,
        "L_ovvo": np.asarray(L[o, v, v, o]),
        "ERI_ovvv": np.asarray(ERI[o, v, v, v]),
        "L_oovo": np.asarray(L[o, o, v, o]),
        "eps_o": np.asarray(cc.H.eps[o]),
        "eps_v": np.asarray(cc.H.eps[v]),
    }


def compare_against_pycc(cc=None, capacity: int = 1 << 32, search: bool = False,
                         rtol: float = 1e-9, atol: float = 1e-11) -> dict:
    """Run the ef T1 slice on a live PyCC state and compare to production PyCC.

    Returns a dict of the separate residual/update max-abs diffs and pass flags.
    Raises AssertionError if the eps-derived denominator disagrees with ``cc.Dia``.
    """
    import ehrenfest as ef
    from . import residual_t1 as R

    if cc is None:
        cc = build_pycc_state()

    inp = extract_inputs(cc)

    # the eps-derived denominator the ef slice builds must equal PyCC's own Dia
    Dia = inp["eps_o"][:, None] - inp["eps_v"][None, :]
    np.testing.assert_allclose(Dia, np.asarray(cc.Dia), rtol=1e-12, atol=0,
                               err_msg="eps-derived denominator != cc.Dia")

    # production PyCC residual + update (the ground truth)
    r1_pycc, _r2 = cc.residuals(cc.H.F, cc.t1, cc.t2)
    r1_pycc = np.asarray(r1_pycc)
    t1_pycc = np.asarray(cc.t1) + r1_pycc / np.asarray(cc.Dia)

    # the ef composed program on the same inputs
    ef_slice = R.build(int(cc.no), int(cc.nv))
    r1_ef, t1_ef = ef_slice.run(inp, ef.runner(capacity, device="cpu"), search=search)

    res_diff = float(np.max(np.abs(r1_ef - r1_pycc)))
    upd_diff = float(np.max(np.abs(t1_ef - t1_pycc)))
    return {
        "residual_max_absdiff": res_diff,
        "update_max_absdiff": upd_diff,
        "residual_ok": bool(np.allclose(r1_ef, r1_pycc, rtol=rtol, atol=atol)),
        "update_ok": bool(np.allclose(t1_ef, t1_pycc, rtol=rtol, atol=atol)),
    }


def main() -> int:
    res = compare_against_pycc()
    print(res)
    return 0 if (res["residual_ok"] and res["update_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
