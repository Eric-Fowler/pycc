"""B1a report driver — ORCHESTRATION ONLY, no harness logic of its own.

This is the runbook made executable, for a psi4-capable machine. It changes nothing
about the measurement: it calls the frozen gate / ``bench_t2.benchmark`` /
``bench_mem.measure`` in the required order and stamps the three provenance fields the
harness does not itself emit (the two live git SHAs and the molecule/basis fixture),
so a single run produces one complete B1a report.

Discipline enforced here (all already the harness's own rules; restated at the driver
level so a report is never assembled out of order):
  * the production-equivalence gate (T1 + all 7 T2 intermediates + composed r2 vs
    ``cc.residuals(...)[1]`` + t2_trial vs ``cc.t2 + r2/cc.Dijab``) runs FIRST; timings
    are refused unless it is green;
  * the timing comparator is the T2-ONLY production region (``bench_t2`` owns that —
    never full ``cc.residuals()``);
  * ``search=False`` declared-cut baseline (B1b ``search=True`` is a SEPARATE run, not
    driven here);
  * memory is a separate section (observed RSS via ``bench_mem``); EF planned working-set
    stays deferred (never ``runner.budget()``).

Run (on a psi4 machine)::

    PYTHONPATH=/path/to/ehrenfest python -m devtools.ef_integration.b0.run_b1a

Prints one JSON document with: provenance (SHAs, platform, versions), fixture
(molecule/basis/(o,v)), correctness (max-abs diffs from the gate), timing (lifecycle +
warm distributions + ratio), and memory (observed RSS both sides + EF planned deferred).
Exit code is nonzero if the gate fails (and no timings are reported).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _git_sha(anchor_file: str) -> str:
    """Short git SHA of the repo containing ``anchor_file`` (a package __file__), or a
    marker string if it cannot be resolved — provenance must never be silently blank."""
    try:
        d = os.path.dirname(os.path.abspath(anchor_file))
        out = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or f"unknown (git: {out.stderr.strip()})"
    except Exception as e:  # not a checkout, git absent, etc.
        return f"unknown ({type(e).__name__}: {e})"


def _provenance() -> dict:
    import numpy as np
    import ehrenfest as ef
    import pycc
    from .bench_t2 import EF_PIN
    prov = {
        "pycc_sha": _git_sha(pycc.__file__),
        "ehrenfest_sha": _git_sha(ef.__file__),
        "ehrenfest_pin_expected": EF_PIN,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import opt_einsum
        prov["opt_einsum"] = opt_einsum.__version__
    except Exception:
        prov["opt_einsum"] = None
    return prov


def _fixture(cc) -> dict:
    """The fixed algebraic state, echoing ``against_pycc.build_pycc_state`` defaults
    (single source of truth for the actual geometry/basis/iteration policy)."""
    return {
        "molecule": "H2O (O; H 1 0.96; H 1 0.96 2 104.5)",
        "basis": "cc-pVDZ",
        "maxiter": 3,
        "max_diis": 0,
        "note": "build_pycc_state defaults — DIIS-off finite-Jacobi state",
        "o": int(cc.no),
        "v": int(cc.nv),
    }


def run(capacity: int = 1 << 32, threads: int | None = 1,
        warm: int = 25, with_memory: bool = True) -> dict:
    from .against_pycc import build_pycc_state, compare_against_pycc as gate_t1
    from .against_pycc_t2 import compare_against_pycc as gate_t2
    from .bench_t2 import benchmark

    cc = build_pycc_state()

    # --- gate FIRST (both slices); timings refused unless green ---
    g1 = gate_t1(cc)
    g2 = gate_t2(cc)
    inter_ok = all(v["ok"] for v in g2["intermediates"].values())
    gate_green = bool(g1["residual_ok"] and g1["update_ok"]
                      and inter_ok and g2["residual_ok"] and g2["update_ok"])

    report = {
        "provenance": _provenance(),
        "fixture": _fixture(cc),
        "correctness": {
            "gate_green": gate_green,
            "t1": {k: g1[k] for k in ("residual_max_absdiff", "update_max_absdiff",
                                      "residual_ok", "update_ok")},
            "t2_intermediates": g2["intermediates"],
            "t2_residual_max_absdiff": g2["residual_max_absdiff"],
            "t2_update_max_absdiff": g2["update_max_absdiff"],
            "t2_residual_ok": g2["residual_ok"],
            "t2_update_ok": g2["update_ok"],
        },
    }
    if not gate_green:
        report["timing"] = None
        report["memory"] = None
        report["note"] = "PRODUCTION GATE NOT GREEN — timings and memory deliberately omitted."
        return report

    # --- timing (T2-only comparator, search=False, controlled threads) ---
    report["timing"] = benchmark(cc, capacity=capacity, warm=warm, threads=threads)

    # --- memory (separate deliverable; fresh subprocess per side) ---
    if with_memory:
        from .bench_mem import measure
        report["memory"] = measure(capacity=capacity, threads=threads)
    else:
        report["memory"] = "skipped (with_memory=False)"

    report["note"] = ("B1a captured: gate green -> T2-only search=False denom=dijab timing "
                      "+ observed RSS. EF planned working-set deferred. B1b (search=True) is a "
                      "SEPARATE run, not included here.")
    return report


def main() -> int:
    rep = run()
    print(json.dumps(rep, indent=2))
    return 0 if rep["correctness"]["gate_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
