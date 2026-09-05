"""PyCC CCSD T2 residual + intermediates + fused Jacobi update as ONE ef.program.

Track B (B0), the meaningful slice: unlike the T1 plumbing proof, the seven
T2-dependent intermediates (``Fae``/``Fmi``/``Fme``/``Wmnij``/``Wmbej``/``Wmbje``/
``Zmbij``) are built INSIDE the single program as cuts (not received pre-
materialized), so ehrenfest actually sees the intermediate-production region and
its reuse/materialization choices. Only ``R2`` (the residual) and ``t2_trial`` are
host-read; the intermediates stay internal program values.

Each PyCC term (``pycc/ccwfn.py`` ``_r_T2_ccsd`` and the ``build_*`` intermediates)
is transcribed with CLEAN relabelled einsums — no ``.T`` / ``.swapaxes`` ops, and
the ``tau`` factors and the ``(2 t2 - t2^T)`` / ``Wmbje^T`` combinations expanded
into explicit terms — so a relabelling mistake shows up as a mismatch against the
faithful mirror (``reference_t2.py``) and, authoritatively, against production PyCC
(``against_pycc_t2.py``).

Planner search is DISABLED (frozen B0 baseline); this is the first slice suitable
for B1a performance work (it includes intermediate production).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ehrenfest as ef

LEAF_SHAPES = {
    "t1": ("o", "v"), "t2": ("o", "o", "v", "v"),
    "F_vv": ("v", "v"), "F_oo": ("o", "o"), "F_ov": ("o", "v"),
    "L_oovv": ("o", "o", "v", "v"), "L_ovvv": ("o", "v", "v", "v"), "L_ooov": ("o", "o", "o", "v"),
    "ERI_oooo": ("o", "o", "o", "o"), "ERI_ooov": ("o", "o", "o", "v"), "ERI_oovo": ("o", "o", "v", "o"),
    "ERI_ovvv": ("o", "v", "v", "v"), "ERI_ovvo": ("o", "v", "v", "o"), "ERI_ovov": ("o", "v", "o", "v"),
    "ERI_vvvv": ("v", "v", "v", "v"), "ERI_vvvo": ("v", "v", "v", "o"), "ERI_ovoo": ("o", "v", "o", "o"),
    "ERI_vvoo": ("v", "v", "o", "o"), "ERI_oovv": ("o", "o", "v", "v"),
    "eps_o": ("o",), "eps_v": ("v",),
}


def _transposed(value, node, letters):
    names = [ix.name for ix in node.free]
    return np.asarray(value).transpose(tuple(names.index(L) for L in letters))


@dataclass
class Slice:
    leaves: dict
    R2: object
    t2_trial: object
    program: object
    inter: dict            # name -> intermediate cut (Fae/Fmi/Fme/Wmnij/Wmbej/Wmbje/Zmbij)

    def arrays(self, inp: dict) -> dict:
        return {leaf.base.uid: np.asarray(inp[name]) for name, leaf in self.leaves.items()}

    def eval_intermediate(self, name: str, inp: dict, out_letters: str):
        """Evaluate one internal intermediate cut on ``inp`` (for staged unit tests)."""
        cut = self.inter[name]
        return _transposed(ef.evaluate(cut.node, self.arrays(inp)), cut.node, out_letters)

    def precompile(self, run, search: bool = False):
        """Plan the program once (search=False is the frozen B0 baseline). Separated
        from execute() so B1a can time build/precompile/first/warm phases apart and
        never re-plan in the warm loop."""
        run.precompile(self.program, search=search)

    def execute(self, run, inp: dict):
        """One pass on an already-precompiled program: begin_pass + execute, no
        planning. This is the per-residual-evaluation cost B1a's warm loop measures."""
        arrays = self.arrays(inp)
        run.begin_pass(arrays)
        got = run.execute(through=self.t2_trial, arrays=arrays)
        r2 = _transposed(got[self.R2], self.R2.node, "ijab")
        t2t = _transposed(got[self.t2_trial], self.t2_trial, "ijab")
        return r2, t2t

    def run(self, inp: dict, run, search: bool = False):
        self.precompile(run, search=search)
        return self.execute(run, inp)

    def evaluate_oracle(self, inp: dict):
        arrays = self.arrays(inp)
        r2 = _transposed(ef.evaluate(self.R2.node, arrays), self.R2.node, "ijab")
        t2t = _transposed(ef.evaluate(self.t2_trial, arrays), self.t2_trial, "ijab")
        return r2, t2t


def build(o: int, v: int) -> Slice:
    ext = {"o": o, "v": v}
    L = {name: ef.array(tuple(ext[s] for s in shp), name) for name, shp in LEAF_SHAPES.items()}
    t1, t2 = L["t1"], L["t2"]

    def tau_cut(f1, f2, name):
        # tau_ijab = f1 t2 + f2 t1_ia t1_jb, materialized once and read positionally
        return ef.cut(f1 * t2.at("ijab") + f2 * ef.einsum("ia,jb->ijab", t1, t1), "ijab", name)

    tau11 = tau_cut(1.0, 1.0, "tau11")
    tau1h = tau_cut(1.0, 0.5, "tau1h")     # tau(1, 0.5): Fae, Fmi
    tauh1 = tau_cut(0.5, 1.0, "tauh1")     # tau(0.5, 1): Wmbej, Wmbje

    # --- one-body intermediates ---
    Fae = ef.cut(L["F_vv"].at("ae")
                 - 0.5 * ef.einsum("me,ma->ae", L["F_ov"], t1)
                 + ef.einsum("mf,mafe->ae", t1, L["L_ovvv"])
                 - ef.einsum("mnaf,mnef->ae", tau1h, L["L_oovv"]), "ae", "Fae")
    Fmi = ef.cut(L["F_oo"].at("mi")
                 + 0.5 * ef.einsum("ie,me->mi", t1, L["F_ov"])
                 + ef.einsum("ne,mnie->mi", t1, L["L_ooov"])
                 + ef.einsum("inef,mnef->mi", tau1h, L["L_oovv"]), "mi", "Fmi")
    Fme = ef.cut(L["F_ov"].at("me") + ef.einsum("nf,mnef->me", t1, L["L_oovv"]), "me", "Fme")

    # --- two-body intermediates ---
    Wmnij = ef.cut(L["ERI_oooo"].at("mnij")
                   + ef.einsum("je,mnie->mnij", t1, L["ERI_ooov"])
                   + ef.einsum("ie,mnej->mnij", t1, L["ERI_oovo"])
                   + ef.einsum("ijef,mnef->mnij", tau11, L["ERI_oovv"]), "mnij", "Wmnij")
    Wmbej = ef.cut(L["ERI_ovvo"].at("mbej")
                   + ef.einsum("jf,mbef->mbej", t1, L["ERI_ovvv"])
                   - ef.einsum("nb,mnej->mbej", t1, L["ERI_oovo"])
                   - ef.einsum("jnfb,mnef->mbej", tauh1, L["ERI_oovv"])
                   + 0.5 * ef.einsum("njfb,mnef->mbej", t2, L["L_oovv"]), "mbej", "Wmbej")
    Wmbje = ef.cut(-1.0 * L["ERI_ovov"].at("mbje")
                   - ef.einsum("jf,mbfe->mbje", t1, L["ERI_ovvv"])
                   + ef.einsum("nb,mnje->mbje", t1, L["ERI_ooov"])
                   + ef.einsum("jnfb,mnfe->mbje", tauh1, L["ERI_oovv"]), "mbje", "Wmbje")
    Zmbij = ef.cut(ef.einsum("mbef,ijef->mbij", L["ERI_ovvv"], tau11), "mbij", "Zmbij")

    # --- raw T2 residual (pre-symmetrization), PyCC terms with clean relabels ---
    tmp_be = ef.einsum("mb,me->be", t1, Fme)     # = (t1^T Fme) used in the -0.5 Fme term
    tmp_jm = ef.einsum("je,me->jm", t1, Fme)
    r2raw = (
        0.5 * ef.einsum("abij->ijab", L["ERI_vvoo"])                       # <ab|ij> -> [i,j,a,b]
        + ef.einsum("ijae,be->ijab", t2, Fae)                             # t2 . Fae^T
        - 0.5 * ef.einsum("ijae,be->ijab", t2, tmp_be)
        - ef.einsum("imab,mj->ijab", t2, Fmi)
        - 0.5 * ef.einsum("imab,jm->ijab", t2, tmp_jm)
        + 0.5 * ef.einsum("mnab,mnij->ijab", tau11, Wmnij)
        + 0.5 * ef.einsum("ijef,abef->ijab", tau11, L["ERI_vvvv"])
        - ef.einsum("ma,mbij->ijab", t1, Zmbij)
        + ef.einsum("imae,mbej->ijab", t2, Wmbej)                         # (2 t2 - t2^T) . Wmbej, expanded:
        - ef.einsum("imea,mbej->ijab", t2, Wmbej)
        + ef.einsum("imae,mbej->ijab", t2, Wmbej)                         # t2 . (Wmbej + Wmbje^T), expanded:
        + ef.einsum("imae,mbje->ijab", t2, Wmbje)
        + ef.einsum("mjae,mbie->ijab", t2, Wmbje)
        - ef.einsum("ie,ma,mbej->ijab", t1, t1, L["ERI_ovvo"])            # the two t1 t1 ERI terms
        - ef.einsum("ie,mb,maje->ijab", t1, t1, L["ERI_ovov"])
        + ef.einsum("ie,abej->ijab", t1, L["ERI_vvvo"])
        - ef.einsum("ma,mbij->ijab", t1, L["ERI_ovoo"])
    )
    R2raw = ef.cut(r2raw, "ijab", "R2raw")

    # --- P(ij)(ab) symmetrization: r_ijab + r_jiba (read the raw-residual cut twice) ---
    R2 = ef.cut(ef.einsum("ijab->ijab", R2raw) + ef.einsum("jiba->ijab", R2raw), "ijab", "R2")

    # --- fused Jacobi update: t2_trial = t2 + R2 * 1/Dijab ---
    e2 = {"a": v, "b": v, "i": o, "j": o}

    def bc(e, own):
        return ef.broadcast(e, {ll: n for ll, n in e2.items() if ll != own})

    Dsum2 = ef.cut(bc(L["eps_o"].at("i"), "i") + bc(L["eps_o"].at("j"), "j")
                   - bc(L["eps_v"].at("a"), "a") - bc(L["eps_v"].at("b"), "b"), "ijab", "Dsum2")
    Dinv2 = ef.map(Dsum2, "reciprocal")
    t2_trial = (t2.at("ijab") + ef.einsum("ijab,ijab->ijab", R2, Dinv2)).node("ijab")

    items = [tau11, tau1h, tauh1, Fae, Fmi, Fme, Wmnij, Wmbej, Wmbje, Zmbij,
             Dsum2, R2raw, R2, t2_trial]
    program = ef.program(items, host=["R2"])
    inter = {"Fae": Fae, "Fmi": Fmi, "Fme": Fme, "Wmnij": Wmnij,
             "Wmbej": Wmbej, "Wmbje": Wmbje, "Zmbij": Zmbij}
    return Slice(leaves=L, R2=R2, t2_trial=t2_trial, program=program, inter=inter)
