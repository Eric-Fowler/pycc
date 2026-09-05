"""B1a-memory: observed peak-RSS measurement for the T2 slice vs the PyCC region.

Kept SEPARATE from the timing ratio (`bench_t2.py`) on purpose — timing and memory
are distinct B1a deliverables. Two quantities, never conflated:

* **Observed incremental peak RSS** — a process-RSS *sampler* (``psutil``), which
  catches NumPy's *native* allocations (``tracemalloc`` alone does not). Contract:
  ``stabilized baseline RSS before region`` -> ``sampled peak RSS during region`` ->
  ``incremental = peak - baseline``. It is a *sampled* peak (a polling sampler can
  miss a very short-lived allocation between samples), not the exact OS high-water
  mark. PyCC and EF are each measured in a **fresh subprocess** (spawn) so one
  implementation's retained allocator/cache state is not charged to the other, and
  each runs under the same explicit CPU-thread policy as the timing benchmark.
* **EF planned memory** — what the compiled ``search=False`` plan says its CPU
  resident/working-set requirement is, from EF's *own* plan/schedule accounting. This
  is reported as **deferred** until the correct accessor is confirmed on a live
  machine (see ``ef_planned_bytes``); it is never approximated from ``runner.budget()``
  (that is remaining fit *capacity*, not the plan's requirement) or from a manual
  tensor-size estimate.

Requires ``psutil`` (and, for the region measurements, psi4). The core sampler and
the subprocess-failure handling are provider-agnostic and unit-tested.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import threading
import time

from .bench_t2 import threadpool_limits, threadpool_info, _HAVE_TPC  # single-source shim

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False


def _rss() -> int:
    return psutil.Process().memory_info().rss


def sampled_peak_rss(fn, *, warmup: int = 1, interval: float = 5e-4) -> dict:
    """Run ``fn`` once and return its observed incremental peak RSS (bytes).

    A background thread samples process RSS every ``interval`` s. The sampler is
    started and has taken its first reading BEFORE the baseline, so its own thread
    stack/startup is in the baseline, not the region's incremental footprint.
    ``warmup`` calls (default 1) run first so import/allocator growth is baselined.
    Requires psutil (fails closed).
    """
    if not _HAVE_PSUTIL:
        raise RuntimeError("psutil is required for observed-RSS measurement")
    for _ in range(warmup):
        fn()

    ready = threading.Event()
    armed = threading.Event()
    stop = threading.Event()
    box = {"peak": 0}

    def sampler():
        _rss()               # first read: pay the sampler's own startup before baseline
        ready.set()
        armed.wait()
        while not stop.is_set():
            box["peak"] = max(box["peak"], _rss())
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    ready.wait()
    gc.collect()
    baseline = _rss()
    box["peak"] = baseline
    armed.set()
    try:
        fn()
        box["peak"] = max(box["peak"], _rss())
    finally:
        stop.set()
        t.join()
    return {"baseline_bytes": baseline, "peak_bytes": box["peak"],
            "incremental_peak_bytes": max(0, box["peak"] - baseline)}


def _measured_under_threads(fn, threads) -> dict:
    """``sampled_peak_rss(fn)`` under an explicit CPU-thread policy; the in-force
    ``threadpool_info`` is captured inside the context and returned. Fails closed if a
    thread count is requested without threadpoolctl (same rule as bench_t2)."""
    if threads is not None and not _HAVE_TPC:
        raise RuntimeError(
            "threadpoolctl is required for a controlled memory measurement; "
            "pass threads=None for an uncontrolled diagnostic")
    with threadpool_limits(limits=threads):
        info = threadpool_info() if _HAVE_TPC else None
        rss = sampled_peak_rss(fn)
    rss["thread_policy"] = "controlled" if (threads is not None and _HAVE_TPC) else "uncontrolled"
    rss["threads"] = threads
    rss["threadpool_info"] = info
    return rss


# --- region builders (need psi4); each is measured in its own fresh subprocess ---

def _pycc_region_target(threads=1):
    from .against_pycc import build_pycc_state
    from .bench_t2 import pycc_t2_region
    cc = build_pycc_state()
    return _measured_under_threads(pycc_t2_region(cc), threads)


def _ef_slice_target(capacity=1 << 32, threads=1):
    import ehrenfest as ef
    from .against_pycc import build_pycc_state
    from .against_pycc_t2 import extract_inputs
    from . import residual_t2 as R2mod
    cc = build_pycc_state()
    inp = extract_inputs(cc)
    sl = R2mod.build(int(cc.no), int(cc.nv), denom="dijab")
    run = ef.runner(capacity, device="cpu")
    sl.precompile(run, search=False)
    return _measured_under_threads(lambda: sl.execute(run, inp), threads)


def _crash_target():  # test helper: die without returning a result
    os._exit(1)


def _echo_target(value=0):  # test helper: a normal child that returns a value
    return value + 1


def _child(entry, kw, conn):
    try:
        conn.send({"ok": True, "result": entry(**kw)})
    except Exception as e:  # surface the failure to the parent
        try:
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
    finally:
        conn.close()


def run_isolated(entry, timeout: float = 180.0, **kw) -> dict:
    """Run ``entry(**kw)`` in a fresh spawned subprocess and return its result dict.

    Bounded and loud: a child that is OOM-killed / segfaults / exits without sending
    is detected via pipe EOF (not a hang), and a wedged child is terminated after
    ``timeout``. Never blocks forever in ``recv``."""
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(entry, kw, send_conn))
    p.start()
    send_conn.close()   # parent is not a writer -> child death reaches us as EOF
    try:
        if not recv_conn.poll(timeout):
            p.join(0)
            if p.is_alive():
                p.terminate()
                p.join()
                raise RuntimeError(f"memory measurement timed out after {timeout}s")
            raise RuntimeError(
                f"memory-measurement child exited {p.exitcode} without returning a result")
        try:
            out = recv_conn.recv()
        except EOFError:
            p.join()
            raise RuntimeError(
                f"memory-measurement child exited {p.exitcode} without returning a result")
    finally:
        recv_conn.close()
    p.join()
    if not out["ok"]:
        raise RuntimeError(f"isolated measurement failed: {out['error']}")
    return out["result"]


def ef_planned_bytes(run) -> dict:
    """EF planned working-set — DEFERRED.

    Not derived from ``runner.budget()`` (that is remaining fit *capacity*, not the
    plan's requirement) nor from manual tensor sizes. EF does expose real schedule
    accounting (e.g. ``ChainSchedule.resident_bytes()`` — the peak device-node bytes
    ``fits`` checks), but the composed-program-level accessor that also captures the
    retained cut/deposit terms for this CPU program must be confirmed on a live
    machine before it is reported. Until then this returns deferred."""
    return {"planned_bytes": None,
            "note": "EF plan/schedule accounting accessor not yet resolved on the pinned "
                    "commit; resolve on a live machine (ChainSchedule.resident_bytes is the "
                    "starting point). Do NOT use runner.budget() — that is remaining capacity."}


def measure(capacity: int = 1 << 32, threads: int | None = 1, timeout: float = 180.0) -> dict:
    """Observed incremental peak RSS for the PyCC region and the EF slice, each in a
    fresh subprocess under the same thread policy. Gated on psi4 in the children."""
    return {
        "pycc_region": run_isolated(_pycc_region_target, timeout=timeout, threads=threads),
        "ef_slice": run_isolated(_ef_slice_target, timeout=timeout, capacity=capacity, threads=threads),
        "ef_planned": ef_planned_bytes(None),
        "note": "observed incremental peak RSS (psutil, sampled), PyCC and EF isolated in "
                "fresh subprocesses under one thread policy; separate from timing and from EF "
                "planned memory (deferred).",
    }


if __name__ == "__main__":
    print(measure())
