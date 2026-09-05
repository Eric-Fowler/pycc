"""Live B0 T2 check: the ef T2 slice vs a real PyCC ``ccwfn`` state.

The production authority for the T2 slice (``reference_t2.py`` shares a translation
with ``residual_t2.py`` and so cannot catch a common error). Requires psi4; the
pytest test skips cleanly where it is unavailable.

Staged, per the plan:
  1. each EF intermediate vs PyCC's own ``build_*`` (localized diagnosis);
  2. composed r2 vs the production residual ``cc.residuals(...)[1]``;
  3. t2_trial vs ``cc.t2 + r2_pycc / cc.Dijab``.

Reuses the DIIS-off fixed-state fixture from ``against_pycc.build_pycc_state``.
"""

from __future__ import annotations

import numpy as np


def extract_inputs(cc) -> dict:
    """The T2 slice's leaves pulled from a live ``ccwfn`` (all numpy)."""
    o, v = cc.o, cc.v
    F, ERI, L = cc.H.F, cc.H.ERI, cc.H.L
    blk = {
        "t1": np.asarray(cc.t1), "t2": np.asarray(cc.t2),
        "F_vv": np.asarray(F[v, v]), "F_oo": np.asarray(F[o, o]), "F_ov": np.asarray(F[o, v]),
        "L_oovv": np.asarray(L[o, o, v, v]), "L_ovvv": np.asarray(L[o, v, v, v]),
        "L_ooov": np.asarray(L[o, o, o, v]),
        "ERI_oooo": np.asarray(ERI[o, o, o, o]), "ERI_ooov": np.asarray(ERI[o, o, o, v]),
        "ERI_oovo": np.asarray(ERI[o, o, v, o]), "ERI_ovvv": np.asarray(ERI[o, v, v, v]),
        "ERI_ovvo": np.asarray(ERI[o, v, v, o]), "ERI_ovov": np.asarray(ERI[o, v, o, v]),
        "ERI_vvvv": np.asarray(ERI[v, v, v, v]), "ERI_vvvo": np.asarray(ERI[v, v, v, o]),
        "ERI_ovoo": np.asarray(ERI[o, v, o, o]), "ERI_vvoo": np.asarray(ERI[v, v, o, o]),
        "ERI_oovv": np.asarray(ERI[o, o, v, v]),
        "eps_o": np.asarray(cc.H.eps[o]), "eps_v": np.asarray(cc.H.eps[v]),
    }
    return blk


def _pycc_intermediates(cc):
    """PyCC's own intermediate builders, aligned to the ef out-letter order."""
    o, v, F, ERI, L = cc.o, cc.v, cc.H.F, cc.H.ERI, cc.H.L
    t1, t2 = cc.t1, cc.t2
    return {
        "Fae": (np.asarray(cc.build_Fae(o, v, F, L, t1, t2)), "ae"),
        "Fmi": (np.asarray(cc.build_Fmi(o, v, F, L, t1, t2)), "mi"),
        "Fme": (np.asarray(cc.build_Fme(o, v, F, L, t1)), "me"),
        "Wmnij": (np.asarray(cc.build_Wmnij(o, v, ERI, t1, t2)), "mnij"),
        "Wmbej": (np.asarray(cc.build_Wmbej(o, v, ERI, L, t1, t2)), "mbej"),
        "Wmbje": (np.asarray(cc.build_Wmbje(o, v, ERI, t1, t2)), "mbje"),
        "Zmbij": (np.asarray(cc.build_Zmbij(o, v, ERI, t1, t2)), "mbij"),
    }


def compare_against_pycc(cc=None, capacity: int = 1 << 32, search: bool = False,
                         rtol: float = 1e-9, atol: float = 1e-11) -> dict:
    import ehrenfest as ef
    from . import residual_t2 as R2mod
    from .against_pycc import build_pycc_state

    if cc is None:
        cc = build_pycc_state()

    inp = extract_inputs(cc)
    sl = R2mod.build(int(cc.no), int(cc.nv))

    # stage 1: intermediates vs PyCC's own builders
    inter = {}
    for name, (ref_val, letters) in _pycc_intermediates(cc).items():
        got = sl.eval_intermediate(name, inp, letters)
        inter[name] = {"max_absdiff": float(np.max(np.abs(got - ref_val))),
                       "ok": bool(np.allclose(got, ref_val, rtol=rtol, atol=atol))}

    # stages 2-3: composed residual + update vs production PyCC
    _r1, r2_pycc = cc.residuals(cc.H.F, cc.t1, cc.t2)
    r2_pycc = np.asarray(r2_pycc)
    t2_pycc = np.asarray(cc.t2) + r2_pycc / np.asarray(cc.Dijab)

    r2_ef, t2_ef = sl.run(inp, ef.runner(capacity, device="cpu"), search=search)
    return {
        "intermediates": inter,
        "residual_max_absdiff": float(np.max(np.abs(r2_ef - r2_pycc))),
        "update_max_absdiff": float(np.max(np.abs(t2_ef - t2_pycc))),
        "residual_ok": bool(np.allclose(r2_ef, r2_pycc, rtol=rtol, atol=atol)),
        "update_ok": bool(np.allclose(t2_ef, t2_pycc, rtol=rtol, atol=atol)),
    }


def main() -> int:
    res = compare_against_pycc()
    print(res)
    inter_ok = all(v["ok"] for v in res["intermediates"].values())
    return 0 if (inter_ok and res["residual_ok"] and res["update_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
