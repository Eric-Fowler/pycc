"""B1a-memory: observed peak-RSS measurement for the T2 slice vs the PyCC region.

Kept SEPARATE from the timing ratio (`bench_t2.py`) on purpose — timing and memory
are distinct B1a deliverables. Two quantities, never conflated:

* **Observed incremental peak RSS** — a process-RSS sampler (``psutil``), which
  catches NumPy's *native* allocations (``tracemalloc`` alone does not). Contract:
  ``stabilized baseline RSS before region`` → ``peak sampled RSS during region`` →
  ``incremental = peak - baseline``. PyCC and EF are each measured in a **fresh
  subprocess** (spawn) so one implementation's retained allocator/cache state is not
  charged to the other.
* **EF planned memory** — what the compiled ``search=False`` plan says its CPU
  resident/working-set requirement is, from EF's *own* plan/schedule accounting (not
  a manual tensor-size estimate). The exact accessor is resolved on a live machine
  (see ``ef_planned_bytes``); until then this field is reported as deferred, never as
  equal to observed RSS.

Requires ``psutil`` (and, for the region measurements, psi4). The core sampler is
provider-agnostic and unit-tested on a numpy workload.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import threading
import time

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False


def _rss() -> int:
    return psutil.Process().memory_info().rss


def sampled_peak_rss(fn, *, warmup: int = 1, interval: float = 5e-4) -> dict:
    """Run ``fn`` once and return its observed incremental peak RSS (bytes).

    A background thread samples process RSS every ``interval`` s. ``warmup`` calls
    (default 1) run first so import/allocator growth is in the baseline, isolating the
    region's own incremental footprint. Requires psutil (fails closed).
    """
    if not _HAVE_PSUTIL:
        raise RuntimeError("psutil is required for observed-RSS measurement")
    for _ in range(warmup):
        fn()
    gc.collect()
    baseline = _rss()
    peak = baseline
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _rss())
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        fn()
        peak = max(peak, _rss())
    finally:
        stop.set()
        t.join()
    return {"baseline_bytes": baseline, "peak_bytes": peak,
            "incremental_peak_bytes": max(0, peak - baseline)}


# --- region builders (need psi4); each is measured in its own fresh subprocess ---

def _pycc_region_target():
    from .against_pycc import build_pycc_state
    from .bench_t2 import pycc_t2_region
    cc = build_pycc_state()
    region = pycc_t2_region(cc)
    return sampled_peak_rss(region)


def _ef_slice_target(capacity: int):
    import ehrenfest as ef
    from .against_pycc import build_pycc_state
    from .against_pycc_t2 import extract_inputs
    from . import residual_t2 as R2mod
    cc = build_pycc_state()
    inp = extract_inputs(cc)
    sl = R2mod.build(int(cc.no), int(cc.nv), denom="dijab")
    run = ef.runner(capacity, device="cpu")
    sl.precompile(run, search=False)
    return sampled_peak_rss(lambda: sl.execute(run, inp))


def _child(entry, kw, q):
    try:
        q.put({"ok": True, "result": entry(**kw)})
    except Exception as e:  # surface the failure to the parent
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def run_isolated(entry, **kw) -> dict:
    """Run ``entry(**kw)`` in a fresh spawned subprocess and return its result dict."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(entry, kw, q))
    p.start()
    out = q.get()
    p.join()
    if not out["ok"]:
        raise RuntimeError(f"isolated measurement failed: {out['error']}")
    return out["result"]


def ef_planned_bytes(run) -> dict:
    """Best-effort EF planned working-set from the runner's own accounting.

    The exact accessor is confirmed on a live machine; this tries a known candidate
    and otherwise reports the quantity as deferred (never guessed, never conflated
    with observed RSS)."""
    budget = getattr(run, "budget", None)
    if callable(budget):
        try:
            return {"planned_bytes": float(budget()), "source": "run.budget()"}
        except Exception as e:
            return {"planned_bytes": None, "note": f"run.budget() raised: {type(e).__name__}: {e}"}
    return {"planned_bytes": None,
            "note": "EF plan-accounting accessor not resolved; fill in on a live machine"}


def measure(capacity: int = 1 << 32) -> dict:
    """Observed incremental peak RSS for the PyCC region and the EF slice, each in a
    fresh subprocess. Gated on psi4 in the children; the timing gate/authority is
    bench_t2/against_pycc_t2, not this module."""
    return {
        "pycc_region": run_isolated(_pycc_region_target),
        "ef_slice": run_isolated(_ef_slice_target, capacity=capacity),
        "ef_planned": "deferred — see ef_planned_bytes (resolve accessor on a live machine)",
        "note": "observed incremental peak RSS (psutil), PyCC and EF isolated in fresh "
                "subprocesses; separate from timing and from EF planned memory.",
    }


if __name__ == "__main__":
    print(measure())
