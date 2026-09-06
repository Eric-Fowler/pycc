"""B1a benchmark harness for the T2 slice — authored ahead, DO NOT interpret its
numbers until the psi4 production-equivalence gate (against_pycc_t2.py) is green.

Contract points, per review:

1. **Apples-to-apples comparator.** NOT ``cc.residuals()`` (which also computes
   r_T1 and would over-charge PyCC). The comparator is a T2-ONLY production region
   (``build_* -> cc.r_T2 -> t2 + r2/cc.Dijab`` on ``cc.contract``/opt_einsum). Before
   timing, both the comparator AND the EF slice are asserted equal to the production
   authority ``cc.residuals(...)[1]`` — the benchmark pins that in code rather than
   relying on review-time knowledge that ``cc.r_T2`` symmetrizes.

2. **Denominator fairness.** The EF benchmark build uses ``denom="dijab"``: it
   consumes PyCC's precomputed ``cc.Dijab`` (built once at wavefunction construction)
   instead of rebuilding ``Dijab`` from orbital energies every pass. So the per-pass
   work is the same on both sides: ``r2 + division/reciprocal-multiply by a
   precomputed denominator``. (The ``denom="eps"`` graph stays the B0 correctness
   evidence.)

3. **Lifecycle split, not one opaque number.** ``Slice.precompile`` / ``Slice.execute``
   are separate; the warm loop reuses the same Slice/program/runner/precompiled plan
   (no rebuild, no precompile). Only the warm distribution is compared.

4. **Thread policy + fair sampling.** The whole timed section runs under an explicit
   ``threadpool_limits(threads)`` when available, and the environment (threads,
   threadpool_info, platform, numpy/opt_einsum versions, EF pin, capacity) is recorded
   so a ratio isn't secretly a thread-runtime artifact. EF and PyCC are both warmed,
   then their timed samples alternate order to avoid fixed-order thermal/frequency bias.

``search=False`` throughout (declared-cut materialization baseline, not
materialize-vs-fuse search — that is B1b).
"""

from __future__ import annotations

import platform
import statistics
import time

import numpy as np

EF_PIN = "da4d2d9"  # ehrenfest claude/tiling-from-scratch pin (see EF_INTEGRATION_PLAN.md §8)

try:
    from threadpoolctl import threadpool_limits, threadpool_info
    _HAVE_TPC = True
except Exception:  # threadpoolctl not installed in this env
    _HAVE_TPC = False

    class _NullLimits:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def threadpool_limits(limits=None):  # noqa: D401 - shim
        return _NullLimits()

    def threadpool_info():
        return None


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


def _authority(cc):
    """Production PyCC residual/update — the correctness authority both timed sides
    are pinned to before any timing."""
    _r1, r2 = cc.residuals(cc.H.F, cc.t1, cc.t2)
    r2 = np.asarray(r2)
    return r2, np.asarray(cc.t2) + r2 / np.asarray(cc.Dijab)


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _env(threads, thread_policy, controlled_info, capacity):
    info = {
        "thread_policy": thread_policy,      # "controlled" or "uncontrolled"
        "threads": threads,
        "threadpool_info": controlled_info,  # captured INSIDE the limits context (in force during timing)
        "platform": platform.platform(),
        "numpy": np.__version__,
        "ef_pin": EF_PIN,
        "capacity": capacity,
    }
    try:
        import opt_einsum
        info["opt_einsum"] = opt_einsum.__version__
    except Exception:
        info["opt_einsum"] = None
    return info


def benchmark(cc, capacity: int = 1 << 32, warm: int = 25, warmup: int = 3,
              threads: int | None = 1, rtol: float = 1e-9, atol: float = 1e-11) -> dict:
    """Run the B1a lifecycle-split benchmark against a live PyCC state.

    Pins both timed sides to the production authority, then times EVERY numerical
    phase (precompile, first execution, warm) under one explicit thread policy, with
    alternating EF/PyCC samples. Correctness is asserted (raises on mismatch); the
    psi4 gate remains the authority.

    ``threads`` fixes the BLAS/OpenMP thread count for the timed section. It requires
    ``threadpoolctl``: with ``threads`` set but threadpoolctl absent the harness fails
    closed (a benchmark must not silently run uncontrolled while reporting a thread
    count). Pass ``threads=None`` to request an explicit *uncontrolled* diagnostic run.
    """
    if threads is not None and not _HAVE_TPC:
        raise RuntimeError(
            "threadpoolctl is required for a controlled B1a benchmark; "
            "pass threads=None explicitly for an uncontrolled diagnostic run")
    thread_policy = "controlled" if (threads is not None and _HAVE_TPC) else "uncontrolled"

    import ehrenfest as ef
    from . import residual_t2 as R2mod
    from .against_pycc_t2 import extract_inputs

    inp = extract_inputs(cc)
    o, v = int(cc.no), int(cc.nv)

    r2_auth, t2_auth = _authority(cc)

    # --- comparator pinned to the authority (untimed): pycc_t2_region == cc.residuals ---
    region = pycc_t2_region(cc)
    r2_reg, t2_reg = region()
    np.testing.assert_allclose(np.asarray(r2_reg), r2_auth, rtol=rtol, atol=atol,
                               err_msg="pycc_t2_region residual != cc.residuals(...)[1]")
    np.testing.assert_allclose(np.asarray(t2_reg), t2_auth, rtol=rtol, atol=atol,
                               err_msg="pycc_t2_region update != authority")

    ef_samples: list = []
    pycc_samples: list = []
    # EVERY numerical-kernel phase runs under ONE thread policy; capture the in-force
    # thread info INSIDE the context (threadpoolctl restores defaults on exit).
    with threadpool_limits(limits=threads):
        controlled_info = threadpool_info() if _HAVE_TPC else None

        t0 = time.perf_counter()
        sl = R2mod.build(o, v, denom="dijab")            # fairness build (fix 1)
        ef_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        run = ef.runner(capacity, device="cpu")
        ef_runner_ctor = time.perf_counter() - t0

        t0 = time.perf_counter()
        sl.precompile(run, search=False)
        ef_precompile = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2_ef, t2_ef = sl.execute(run, inp)              # first (cold) execution
        ef_first_exec = time.perf_counter() - t0

        # the exact EF graph being benchmarked, pinned to the authority
        np.testing.assert_allclose(np.asarray(r2_ef), r2_auth, rtol=rtol, atol=atol,
                                   err_msg="ef dijab-mode residual != authority")
        np.testing.assert_allclose(np.asarray(t2_ef), t2_auth, rtol=rtol, atol=atol,
                                   err_msg="ef dijab-mode update != authority")

        # warm both, then alternate timed samples (fix 4)
        for _ in range(warmup):
            sl.execute(run, inp)
            region()
        for k in range(warm):
            if k % 2 == 0:
                ef_samples.append(_timed(lambda: sl.execute(run, inp)))
                pycc_samples.append(_timed(region))
            else:
                pycc_samples.append(_timed(region))
                ef_samples.append(_timed(lambda: sl.execute(run, inp)))

    def dist(xs):
        q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0], xs[0], xs[0]]
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs),
                "q1": q[0], "q3": q[2], "n": len(xs), "samples": xs}

    ef_d, pycc_d = dist(ef_samples), dist(pycc_samples)
    return {
        "shape": {"o": o, "v": v},
        "environment": _env(threads, thread_policy, controlled_info, capacity),
        "ef_build_s": ef_build,
        "ef_runner_ctor_s": ef_runner_ctor,
        "ef_precompile_s": ef_precompile,
        "ef_first_exec_s": ef_first_exec,
        "ef_warm_s": ef_d,
        "pycc_t2_region_s": pycc_d,
        "warm_ratio_ef_over_pycc": ef_d["median"] / pycc_d["median"] if pycc_d["median"] else None,
        "memory": "not measured here — see B1a-memory (bench_mem.py); timing and memory "
                  "are separate deliverables",
        "note": "search=False declared-cut baseline; denom=dijab (precomputed cc.Dijab); "
                "all numerical phases under one thread policy; warm reuses one precompiled "
                "program; both sides pinned to cc.residuals(...)[1] before timing. Interpret "
                "only after the psi4 production gate is green.",
    }


def main() -> int:
    from .against_pycc import build_pycc_state
    from .against_pycc_t2 import compare_against_pycc
    cc = build_pycc_state()
    gate = compare_against_pycc(cc)     # correctness authority FIRST (staged, eps-mode)
    inter_ok = all(v["ok"] for v in gate["intermediates"].values())
    if not (inter_ok and gate["residual_ok"] and gate["update_ok"]):
        print("PRODUCTION GATE FAILED — refusing to report timings:", gate)
        return 1
    print(benchmark(cc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
