"""A0 contraction-signature capture (ehrenfest integration, Track A).

A dependency-light recorder for the *signatures* of the ``contract(subscripts,
*operands)`` calls PyCC makes, so the real corpus can be replayed offline against
``ehrenfest`` (``ef``) without ever serializing production tensor values. It is
imported lazily by :class:`pycc.device.ContractionBackend` only when capture is
switched on (env var ``PYCC_EF_CAPTURE`` = the output JSONL path), so it adds
nothing to a normal run and pulls in nothing beyond the standard library + numpy.

A "signature" is everything that identifies the contraction *invocation* for the
EF integration, and nothing about the values:

* ``clean_spec`` — the subscripts with all whitespace removed and case preserved
  (``ef.einsum`` is whitespace-sensitive and case-sensitive; numpy is neither);
* ``explicit_output`` — whether the spec spells ``->`` (the adapter refuses
  implicit-output specs loudly rather than raising ``IndexError``; A0 reports
  whether any occur in the real corpus);
* per-operand ``shapes``, ``dtypes`` (per operand, *not* ``np.result_type`` — two
  mixed-width operand combinations can share a result type yet be different
  invocations), and ``layouts`` (``C`` / ``F`` / ``non-contiguous`` — the view
  class PyCC actually passes; captured here, deliberately NOT part of the EF node
  cache key, which is contraction identity only).

This module owns these definitions; the offline tools under
``devtools/ef_integration/`` consume them rather than re-deriving them.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading

import numpy as np


def layout_class(a) -> str:
    """The contiguity class of an operand: ``'C'``, ``'F'`` or ``'non-contiguous'``.

    This is the coverage-relevant fact about a view (a PyCC operand is often a
    ``swapaxes`` / slice / transpose view); the exact strides are not needed to
    reconstruct a representative operand for replay.
    """
    a = np.asarray(a)
    if a.flags["C_CONTIGUOUS"]:
        return "C"
    if a.flags["F_CONTIGUOUS"]:
        return "F"
    return "non-contiguous"


def signature(subscripts: str, operands) -> dict:
    """The value-free signature of one ``contract(subscripts, *operands)`` call."""
    spec = "".join(subscripts.split())  # normalize ALL whitespace; keep case
    ops = [np.asarray(o) for o in operands]
    return {
        "clean_spec": spec,
        "explicit_output": "->" in spec,
        "n_operands": len(ops),
        "shapes": [list(o.shape) for o in ops],
        "dtypes": [o.dtype.str for o in ops],
        "layouts": [layout_class(o) for o in ops],
    }


def _dedup_key(sig: dict):
    return (
        sig["clean_spec"],
        tuple(tuple(s) for s in sig["shapes"]),
        tuple(sig["dtypes"]),
        tuple(sig["layouts"]),
    )


class SignatureRecorder:
    """Aggregate unique contraction signatures with correct occurrence counts.

    One JSONL line per unique ``(clean_spec, shapes, dtypes, layouts)`` carrying a
    running ``count``. Counts are held in memory and written by :meth:`flush`; the
    file is (re)written atomically on a periodic batch threshold and, crucially, on
    :meth:`close` — which :func:`maybe_record` registers with :mod:`atexit`, so the
    final counts always land regardless of when the last new signature appeared
    (the earlier flush-only-on-new-signature bug left duplicate occurrences that
    arrived after the last unique one unpersisted).

    Best-effort: recording never raises into the caller's contraction, but capture
    failures are collected (``.errors``), surfaced once on stderr, and written to a
    ``<path>.errors`` sidecar on flush, so an unwritable path or a bad operand can
    never masquerade as a clean empty/partial capture.
    """

    def __init__(self, path: str, flush_every: int = 2000):
        self.path = path
        self.flush_every = flush_every
        self._seen: dict = {}
        self._lock = threading.Lock()
        self._since_flush = 0
        self.errors: list = []

    def record(self, subscripts: str, operands) -> None:
        try:
            sig = signature(subscripts, operands)
            key = _dedup_key(sig)
            with self._lock:
                entry = self._seen.get(key)
                if entry is None:
                    sig["count"] = 1
                    self._seen[key] = sig
                else:
                    entry["count"] += 1
                self._since_flush += 1
                do_flush = self._since_flush >= self.flush_every
            if do_flush:
                self.flush()  # crash-safety only; correctness comes from close()/atexit
        except Exception as exc:
            self._note_error(subscripts, exc)

    def _note_error(self, subscripts, exc) -> None:
        with self._lock:
            first = not self.errors
            self.errors.append({"subscripts": repr(subscripts)[:120],
                                "error": f"{type(exc).__name__}: {exc}"})
        if first:
            print(f"[pycc.ef_capture] WARNING: signature capture error (further errors "
                  f"collected in {self.path}.errors): {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    def flush(self) -> None:
        with self._lock:
            self._since_flush = 0
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                for sig in self._seen.values():
                    fh.write(json.dumps(sig) + "\n")
            os.replace(tmp, self.path)  # atomic; a crash mid-write can't truncate the file
            if self.errors:
                with open(self.path + ".errors", "w") as fh:
                    for e in self.errors:
                        fh.write(json.dumps(e) + "\n")

    def close(self) -> None:
        self.flush()


_recorder = None
_recorder_lock = threading.Lock()


def maybe_record(subscripts: str, operands) -> None:
    """Record a signature iff ``PYCC_EF_CAPTURE`` names an output path.

    Called from the hot contraction path; the env lookup + lazy singleton keep the
    disabled case to one dict/attr check. The singleton's :meth:`SignatureRecorder.close`
    is registered with :mod:`atexit` so final counts persist at interpreter exit.
    """
    path = os.environ.get("PYCC_EF_CAPTURE")
    if not path:
        return
    global _recorder
    if _recorder is None or _recorder.path != path:
        with _recorder_lock:
            if _recorder is None or _recorder.path != path:
                _recorder = SignatureRecorder(path)
                atexit.register(_recorder.close)
    _recorder.record(subscripts, operands)
