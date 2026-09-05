"""Faithful numpy MIRROR of PyCC's CCSD T2 residual + intermediates + update.

This deliberately mirrors ``pycc/ccwfn.py`` term-for-term — the same einsum
subscripts, the same ``.T`` / ``.swapaxes`` gymnastics — so it is a low-risk copy
of the source used as the LOCAL diagnosis oracle in the psi4-less environment. It
is NOT the final authority: because it and ``residual_t2.py`` are both translations
of the same equations, only the production comparison in ``against_pycc_t2.py``
(``cc.residuals(...)[1]``) can catch an error common to both. Its value here is
that ``residual_t2.py`` uses clean relabelled einsums (no ``.T``/``.swapaxes``), so
a mismatch against this faithful mirror localizes a relabelling mistake.

All integral blocks are passed as independent arrays (see ``random_state``): the CC
equations are algebraic identities in the blocks, so EF-vs-mirror equivalence holds
for independent random blocks; physical inter-block symmetry only matters for the
production comparison, which uses real integrals.

Blocks / conventions (PyCC): t1 [i,a] (o,v); t2 [i,j,a,b]; ERI is <pq|rs>.
"""

from __future__ import annotations

import numpy as np


def build_tau(t1, t2, f1=1.0, f2=1.0):
    return f1 * t2 + f2 * np.einsum("ia,jb->ijab", t1, t1)


def build_Fae(s):
    return (s["F_vv"]
            - 0.5 * np.einsum("me,ma->ae", s["F_ov"], s["t1"])
            + np.einsum("mf,mafe->ae", s["t1"], s["L_ovvv"])
            - np.einsum("mnaf,mnef->ae", build_tau(s["t1"], s["t2"], 1.0, 0.5), s["L_oovv"]))


def build_Fmi(s):
    return (s["F_oo"]
            + 0.5 * np.einsum("ie,me->mi", s["t1"], s["F_ov"])
            + np.einsum("ne,mnie->mi", s["t1"], s["L_ooov"])
            + np.einsum("inef,mnef->mi", build_tau(s["t1"], s["t2"], 1.0, 0.5), s["L_oovv"]))


def build_Fme(s):
    return s["F_ov"] + np.einsum("nf,mnef->me", s["t1"], s["L_oovv"])


def build_Wmnij(s):
    return (s["ERI_oooo"]
            + np.einsum("je,mnie->mnij", s["t1"], s["ERI_ooov"])
            + np.einsum("ie,mnej->mnij", s["t1"], s["ERI_oovo"])
            + np.einsum("ijef,mnef->mnij", build_tau(s["t1"], s["t2"]), s["ERI_oovv"]))


def build_Wmbej(s):
    return (s["ERI_ovvo"]
            + np.einsum("jf,mbef->mbej", s["t1"], s["ERI_ovvv"])
            - np.einsum("nb,mnej->mbej", s["t1"], s["ERI_oovo"])
            - np.einsum("jnfb,mnef->mbej", build_tau(s["t1"], s["t2"], 0.5, 1.0), s["ERI_oovv"])
            + 0.5 * np.einsum("njfb,mnef->mbej", s["t2"], s["L_oovv"]))


def build_Wmbje(s):
    return (-1.0 * s["ERI_ovov"]
            - np.einsum("jf,mbfe->mbje", s["t1"], s["ERI_ovvv"])
            + np.einsum("nb,mnje->mbje", s["t1"], s["ERI_ooov"])
            + np.einsum("jnfb,mnfe->mbje", build_tau(s["t1"], s["t2"], 0.5, 1.0), s["ERI_oovv"]))


def build_Zmbij(s):
    return np.einsum("mbef,ijef->mbij", s["ERI_ovvv"], build_tau(s["t1"], s["t2"]))


def r_t2(s):
    """The full CCSD T2 residual r2[i,j,a,b], symmetrized (r_ijab + r_jiba)."""
    t1, t2 = s["t1"], s["t2"]
    Fae, Fmi, Fme = build_Fae(s), build_Fmi(s), build_Fme(s)
    Wmnij, Wmbej, Wmbje, Zmbij = build_Wmnij(s), build_Wmbej(s), build_Wmbje(s), build_Zmbij(s)
    tau = build_tau(t1, t2)

    r = 0.5 * s["ERI_vvoo"].swapaxes(0, 2).swapaxes(1, 3)          # <ab|ij> -> [i,j,a,b]
    r = r + np.einsum("ijae,eb->ijab", t2, Fae.T)
    tmp = np.einsum("bm,me->be", t1.T, Fme)
    r = r - 0.5 * np.einsum("ijae,eb->ijab", t2, tmp.T)
    r = r - np.einsum("imab,mj->ijab", t2, Fmi)
    tmp = np.einsum("je,em->jm", t1, Fme.T)
    r = r - 0.5 * np.einsum("imab,mj->ijab", t2, tmp.T)
    r = r + 0.5 * np.einsum("mnab,mnij->ijab", tau, Wmnij)
    r = r + 0.5 * np.einsum("ijef,abef->ijab", tau, s["ERI_vvvv"])
    r = r - np.einsum("ma,mbij->ijab", t1, Zmbij)
    r = r + np.einsum("imae,mbej->ijab", (t2 - t2.swapaxes(2, 3)), Wmbej)
    r = r + np.einsum("imae,mbej->ijab", t2, (Wmbej + Wmbje.swapaxes(2, 3)))
    r = r + np.einsum("mjae,mbie->ijab", t2, Wmbje)
    tmp = np.einsum("ei,am->aemi", t1.T, t1.T)
    r = r - np.einsum("imea,mbej->ijab", tmp.swapaxes(0, 3).swapaxes(1, 2), s["ERI_ovvo"])
    r = r - np.einsum("imeb,maje->ijab", tmp.swapaxes(0, 3).swapaxes(1, 2), s["ERI_ovov"])
    r = r + np.einsum("ie,abej->ijab", t1, s["ERI_vvvo"])
    r = r - np.einsum("ma,mbij->ijab", t1, s["ERI_ovoo"])

    return r + r.swapaxes(0, 1).swapaxes(2, 3)                     # P(ij)(ab) symmetrization


def jacobi_update(t2, r2, eps_o, eps_v):
    Dijab = (eps_o[:, None, None, None] + eps_o[None, :, None, None]
             - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    return t2 + r2 / Dijab


def random_state(o: int, v: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    eps_o = np.sort(rng.uniform(-1.0, -0.3, size=o))
    eps_v = np.sort(rng.uniform(0.3, 1.5, size=v))
    R = rng.standard_normal
    return {
        "t1": R((o, v)) * 1e-2, "t2": R((o, o, v, v)) * 1e-2,
        "F_vv": R((v, v)), "F_oo": R((o, o)), "F_ov": R((o, v)) * 1e-3,
        "L_oovv": R((o, o, v, v)), "L_ovvv": R((o, v, v, v)), "L_ooov": R((o, o, o, v)),
        "ERI_oooo": R((o, o, o, o)), "ERI_ooov": R((o, o, o, v)), "ERI_oovo": R((o, o, v, o)),
        "ERI_ovvv": R((o, v, v, v)), "ERI_ovvo": R((o, v, v, o)), "ERI_ovov": R((o, v, o, v)),
        "ERI_vvvv": R((v, v, v, v)), "ERI_vvvo": R((v, v, v, o)), "ERI_ovoo": R((o, v, o, o)),
        "ERI_vvoo": R((v, v, o, o)), "ERI_oovv": R((o, o, v, v)),
        "eps_o": eps_o, "eps_v": eps_v,
    }
