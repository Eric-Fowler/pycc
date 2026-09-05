"""B1a benchmark harness for the T2 slice — authored ahead, DO NOT interpret its
numbers until the psi4 production-equivalence gate (against_pycc_t2.py) is green.

Two things this gets right, per the review:

1. **Apples-to-apples PyCC comparator.** NOT ``cc.residuals()`` (which also computes
   r_T1 and would over-charge PyCC). The comparator is a T2-ONLY production region
   built from PyCC's own methods with the same seven intermediate productions, the
   same symmetrized r_T2, and the same Jacobi update, run on ``cc.contract`` /
   opt_einsum:

       Fae..Zmbij = cc.build_*(...)
       r2 = cc.r_T2(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)
       t2_trial = t2 + r2 / cc.Dijab

   (``cc.r_T2`` already applies the P(ij)(ab) symmetrization, so this equals
   ``cc.residuals(...)[1]`` — the correctness authority — while doing only T2 work.)

2. **Lifecycle split, not one opaque number.** ``Slice.run`` calls ``precompile``
   internally; timing repeated ``run`` calls would fold planning into every sample.
   This times, separately: EF IR/program build, runner construction,
   precompile(search=False), first begin_pass+execute, and a warm loop that reuses
   the same Slice / program / runner / already-precompiled plan (no rebuild, no
   precompile). Only the warm distribution is compared to the PyCC T2 region.

``search=False`` throughout: this measures the declared-cut materialization baseline
(the default value of the materialize-vs-fuse DoF), not the planner's search. The
search=True experiment is B1b, reported separately.
"""

from __future__ import annotations

import statistics
import time


def pycc_t2_region(cc):
    """A zero-arg callable computing the T2-only production result (r2, t2_trial),
    the apples-to-apples performance comparator for the ef T2 slice."""
    o, v, F, ERI, L = cc.o, cc.v, cc.H.F, cc.H.ERI, cc.H.L
    t1, t2 = cc.t1, cc.t2
    Dijab = cc.Dijab

    def run_once():
        Fae = cc.build_Fae(o, v, F, L, t1, t2)
        Fmi = cc.build_Fmi(o, v, F, L, t1, t2)
        Fme = cc.build_Fme(o, v, F, L, t1)
        Wmnij = cc.build_Wmnij(o, v, ERI, t1, t2)
        Wmbej = cc.build_Wmbej(o, v, ERI, L, t1, t2)
        Wmbje = cc.build_Wmbje(o, v, ERI, t1, t2)
        Zmbij = cc.build_Zmbij(o, v, ERI, t1, t2)
        r2 = cc.r_T2(o, v, F, ERI, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)
        return r2, t2 + r2 / Dijab

    return run_once


def _time(fn, repeats: int):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def benchmark(cc, capacity: int = 1 << 32, warm: int = 25, pycc_warm: int = 25) -> dict:
    """Run the B1a lifecycle-split benchmark against a live PyCC state.

    Returns a report with each EF lifecycle phase, the PyCC T2-region distribution,
    and the warm speed ratio. Correctness is NOT re-asserted here — that is the
    production gate's job; call it first.
    """
    import ehrenfest as ef
    from . import residual_t2 as R2mod
    from .against_pycc_t2 import extract_inputs

    inp = extract_inputs(cc)
    o, v = int(cc.no), int(cc.nv)

    # --- EF lifecycle, timed in phases ---
    t0 = time.perf_counter()
    sl = R2mod.build(o, v)                       # IR / program build
    ef_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    run = ef.runner(capacity, device="cpu")      # runner construction
    ef_runner_ctor = time.perf_counter() - t0

    t0 = time.perf_counter()
    sl.precompile(run, search=False)             # planning (declared-cut baseline)
    ef_precompile = time.perf_counter() - t0

    t0 = time.perf_counter()
    sl.execute(run, inp)                         # first (cold) execution
    ef_first_exec = time.perf_counter() - t0

    ef_warm = _time(lambda: sl.execute(run, inp), warm)   # reuses plan; no precompile

    # --- PyCC T2-only production region ---
    region = pycc_t2_region(cc)
    region()                                     # warm caches (opt_einsum path/threads)
    pycc_warm_ts = _time(region, pycc_warm)

    ef_med = statistics.median(ef_warm)
    pycc_med = statistics.median(pycc_warm_ts)
    return {
        "shape": {"o": o, "v": v},
        "ef_build_s": ef_build,
        "ef_runner_ctor_s": ef_runner_ctor,
        "ef_precompile_s": ef_precompile,
        "ef_first_exec_s": ef_first_exec,
        "ef_warm_s": {"median": ef_med, "min": min(ef_warm), "max": max(ef_warm), "n": warm},
        "pycc_t2_region_s": {"median": pycc_med, "min": min(pycc_warm_ts),
                             "max": max(pycc_warm_ts), "n": pycc_warm},
        "warm_ratio_ef_over_pycc": (ef_med / pycc_med) if pycc_med else None,
        "note": "search=False (declared-cut materialization baseline); warm reuses one "
                "precompiled program. Interpret only after the psi4 production gate is green.",
    }


def main() -> int:
    from .against_pycc import build_pycc_state
    from .against_pycc_t2 import compare_against_pycc
    cc = build_pycc_state()
    gate = compare_against_pycc(cc)     # correctness authority FIRST
    inter_ok = all(v["ok"] for v in gate["intermediates"].values())
    if not (inter_ok and gate["residual_ok"] and gate["update_ok"]):
        print("PRODUCTION GATE FAILED — refusing to report timings:", gate)
        return 1
    print(benchmark(cc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
