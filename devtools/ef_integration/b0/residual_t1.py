"""PyCC CCSD T1 residual + fused Jacobi update as ONE composed ``ef.program``.

Track B (B0), plumbing milestone. This re-expresses PyCC's own ``r_T1`` equations
(see ``reference.py``) as an ehrenfest composed program: a residual cut ``R1``, the
reciprocal-denominator ``Map`` of an eps-sum cut, and the fused update root
``t1 + R1 * Dinv1`` — the same update shape ehrenfest's CCSD ``amplitude_update``
uses, driven with planner search DISABLED (the frozen B0 baseline).

The CCSD intermediates (``Fae``/``Fmi``/``Fme``) and the integral blocks are input
leaves, so this slice isolates the residual assembly + update boundary (the T1
scope the plan allows as an explicit plumbing milestone; T2 is the next step).

Public API:
  * ``build(o, v)`` -> a ``Slice`` describing the leaves, cuts, roots and program;
  * ``Slice.arrays(inp)`` -> the ``{base.uid: ndarray}`` feed for ``inp`` (name->array);
  * ``Slice.run(inp, run, search=False)`` -> ``(r1, t1_trial)`` numpy arrays, both
    aligned to PyCC's ``[i, a]`` order, via the composed program.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ehrenfest as ef


def _transposed(value, node, letters):
    # a Sum/cut free order is a realization detail; recover the caller's order by name
    names = [ix.name for ix in node.free]
    return np.asarray(value).transpose(tuple(names.index(L) for L in letters))


@dataclass
class Slice:
    leaves: dict          # name -> ef.array (the input leaves)
    R1: object            # residual cut
    Dsum1: object         # eps-sum cut
    t1_trial: object      # update root node
    program: object       # ef.program

    def arrays(self, inp: dict) -> dict:
        return {leaf.base.uid: np.asarray(inp[name]) for name, leaf in self.leaves.items()}

    def run(self, inp: dict, run, search: bool = False):
        arrays = self.arrays(inp)
        run.precompile(self.program, search=search)
        run.begin_pass(arrays)
        got = run.execute(through=self.t1_trial, arrays=arrays)
        r1 = _transposed(got[self.R1], self.R1.node, "ia")
        t1_trial = _transposed(got[self.t1_trial], self.t1_trial, "ia")
        return r1, t1_trial


def build(o: int, v: int) -> Slice:
    # --- input leaves (PyCC tensors / intermediates / integral blocks) ---
    t1 = ef.array((o, v), "t1")
    t2 = ef.array((o, o, v, v), "t2")
    f_ai = ef.array((o, v), "f_ai")
    Fae = ef.array((v, v), "Fae")
    Fmi = ef.array((o, o), "Fmi")
    Fme = ef.array((o, v), "Fme")
    L_ovvo = ef.array((o, v, v, o), "L_ovvo")
    ERI_ovvv = ef.array((o, v, v, v), "ERI_ovvv")
    L_oovo = ef.array((o, o, v, o), "L_oovo")
    eps_o = ef.array((o,), "eps_o")
    eps_v = ef.array((v,), "eps_v")

    leaves = {"t1": t1, "t2": t2, "f_ai": f_ai, "Fae": Fae, "Fmi": Fmi, "Fme": Fme,
              "L_ovvo": L_ovvo, "ERI_ovvv": ERI_ovvv, "L_oovo": L_oovo,
              "eps_o": eps_o, "eps_v": eps_v}

    # --- residual r1[i,a], PyCC's own equation, expanded so no swapaxes op is needed ---
    r1_expr = (
        f_ai.at("ia")
        + ef.einsum("ie,ae->ia", t1, Fae)
        - ef.einsum("mi,ma->ia", Fmi, t1)
        + 2.0 * ef.einsum("imae,me->ia", t2, Fme) - ef.einsum("imea,me->ia", t2, Fme)
        + ef.einsum("nf,nafi->ia", t1, L_ovvo)
        + 2.0 * ef.einsum("mief,maef->ia", t2, ERI_ovvv) - ef.einsum("mife,maef->ia", t2, ERI_ovvv)
        - ef.einsum("mnae,nmei->ia", t2, L_oovo)
    )
    R1 = ef.cut(r1_expr, "ia", "R1")

    # --- Jacobi denominator Dia = eps_o[i] - eps_v[a]; reciprocal fused as a Map leaf ---
    e1 = {"a": v, "i": o}

    def bc(e, own):
        return ef.broadcast(e, {L: n for L, n in e1.items() if L != own})

    Dsum1 = ef.cut(bc(eps_o.at("i"), "i") - bc(eps_v.at("a"), "a"), "ia", "Dsum1")
    Dinv1 = ef.map(Dsum1, "reciprocal")

    # --- fused update root: t1_trial = t1 + R1 * Dinv1 (elementwise) ---
    t1_trial = (t1.at("ia") + ef.einsum("ia,ia->ia", R1, Dinv1)).node("ia")

    program = ef.program([Dsum1, R1, t1_trial], host=["R1"])
    return Slice(leaves=leaves, R1=R1, Dsum1=Dsum1, t1_trial=t1_trial, program=program)


# --- oracle-only path (no runner/program), for the equation-correctness check ---
def evaluate_oracle(slice_: Slice, inp: dict):
    arrays = slice_.arrays(inp)
    r1 = _transposed(ef.evaluate(slice_.R1.node, arrays), slice_.R1.node, "ia")
    t1_trial = _transposed(ef.evaluate(slice_.t1_trial, arrays), slice_.t1_trial, "ia")
    return r1, t1_trial
