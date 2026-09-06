"""Load the canonical capture definitions (``pycc/ef_capture.py``).

``signature``/``layout_class`` are owned by ``pycc.ef_capture`` (one
implementation of the concept). In a normal environment ``import pycc.ef_capture``
works; in a lightweight dev environment where importing the ``pycc`` package fails
(no psi4/scipy), we load the module directly by file path — it only needs stdlib +
numpy — so the offline tools keep using the same definitions rather than a copy.
"""

from __future__ import annotations

import importlib.util
import pathlib


def _load():
    try:  # normal path: the pycc package imports cleanly
        from pycc.ef_capture import layout_class, signature  # type: ignore
        return signature, layout_class
    except Exception:
        pass
    # dev path: load the single-file module without importing the package
    p = pathlib.Path(__file__).resolve().parents[2] / "pycc" / "ef_capture.py"
    spec = importlib.util.spec_from_file_location("pycc_ef_capture_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.signature, mod.layout_class


signature, layout_class = _load()
