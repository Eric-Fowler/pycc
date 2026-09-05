"""A0 self-tests — runnable with numpy + ehrenfest only (no psi4/pycc-package).

Covers the pieces the plan's A0 must get right:
  * signature capture records per-operand layout / dtype;
  * the ef node cache is layout-INDEPENDENT (same cached node, contiguous and
    non-contiguous operands -> same result);
  * whitespace and uppercase specs are handled; implicit-output is rejected loudly;
  * the reference adapter matches numpy on representative PyCC patterns.

Run:  python -m pytest devtools/ef_integration/tests/test_a0.py
"""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.ef_integration._capture_api import signature, layout_class  # noqa: E402
from devtools.ef_integration.adapter import EfContractionAdapter  # noqa: E402
from devtools.ef_integration.replay import make_operand  # noqa: E402


def test_layout_class():
    c = np.ascontiguousarray(np.zeros((3, 4)))
    f = np.asfortranarray(np.zeros((3, 4)))
    nc = np.zeros((6, 4))[::2]
    assert layout_class(c) == "C"
    assert layout_class(f) == "F"
    assert layout_class(nc) == "non-contiguous"


def test_signature_records_metadata():
    a = np.zeros((4, 3), dtype="f8")
    b = np.zeros((6, 3))[::2]  # non-contiguous, shape (3, 3)
    sig = signature("mnaf, mnef -> ae", [a, b][0:1] + [b])
    assert sig["clean_spec"] == "mnaf,mnef->ae"  # whitespace normalized
    assert sig["explicit_output"] is True
    assert sig["layouts"][1] == "non-contiguous"
    assert sig["dtypes"][0] == "<f8"


def test_make_operand_honors_layout():
    rng = np.random.default_rng(0)
    assert layout_class(make_operand((3, 4), "<f8", "C", rng)) == "C"
    assert layout_class(make_operand((3, 4), "<f8", "F", rng)) == "F"
    assert layout_class(make_operand((3, 4), "<f8", "non-contiguous", rng)) == "non-contiguous"


def test_node_cache_is_layout_independent():
    """The SAME cached ef node must give the same result for contiguous and
    non-contiguous operands — layout is runtime storage, not contraction identity."""
    rng = np.random.default_rng(1)
    ad = EfContractionAdapter(oracle=True)
    spec = "mnaf,mnef->ae"
    shp = [(4, 4, 3, 3), (4, 4, 3, 3)]
    c_ops = [make_operand(s, "<f8", "C", rng) for s in shp]
    r1 = ad(spec, *c_ops)
    key = ("mnaf,mnef->ae", tuple(shp), ("<f8", "<f8"))
    node_after_first = ad._cache[key][0]
    # non-contiguous operands with identical values
    nc_ops = [np.zeros((s[0] * 2,) + s[1:])[::2] for s in shp]
    for nc, c in zip(nc_ops, c_ops):
        nc[...] = c
        assert layout_class(nc) == "non-contiguous"
    r2 = ad(spec, *nc_ops)
    assert ad._cache[key][0] is node_after_first  # same node object reused
    assert np.allclose(r1, r2)
    assert np.allclose(r1, np.einsum(spec, *c_ops))


def test_implicit_output_rejected():
    ad = EfContractionAdapter(oracle=True)
    with pytest.raises(ValueError):
        ad("ab,bc", np.zeros((2, 2)), np.zeros((2, 2)))  # no '->'


def test_uppercase_and_whitespace_specs():
    rng = np.random.default_rng(2)
    ad = EfContractionAdapter(oracle=True)
    ops = [rng.standard_normal((3,)), rng.standard_normal((3,)), rng.standard_normal((6, 3, 3))]
    got = ad("E,F,mEF-> m", *ops)  # uppercase indices + whitespace
    assert np.allclose(got, np.einsum("E,F,mEF->m", *ops))


def _load_capture_module():
    import importlib.util
    p = pathlib.Path(__file__).resolve().parents[3] / "pycc" / "ef_capture.py"
    spec = importlib.util.spec_from_file_location("pycc_ef_capture_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_recorder_persists_final_counts(tmp_path):
    """Regression: occurrences after the last NEW signature must still be counted."""
    m = _load_capture_module()
    import json
    out = str(tmp_path / "sigs.jsonl")
    rec = m.SignatureRecorder(out, flush_every=10_000)  # no batch flush; rely on close()
    a = np.zeros((3, 4))
    b = np.zeros((5,))
    rec.record("ij,jk->ik", [a, np.zeros((4, 2))])   # signature A, first seen
    rec.record("i->", [b])                            # signature B, first seen (last new one)
    for _ in range(100):
        rec.record("ij,jk->ik", [a, np.zeros((4, 2))])  # A repeated after B discovered
    rec.close()
    counts = {}
    with open(out) as fh:
        for line in fh:
            s = json.loads(line)
            counts[s["clean_spec"]] = s["count"]
    assert counts["ij,jk->ik"] == 101   # would be 1 under the old flush-on-new-signature bug
    assert counts["i->"] == 1


def test_recorder_surfaces_errors(tmp_path):
    m = _load_capture_module()
    import json
    out = str(tmp_path / "sigs.jsonl")
    rec = m.SignatureRecorder(out)
    rec.record(123, [np.zeros((2, 2))])   # non-str subscripts -> signature() raises; must be caught
    rec.close()
    assert rec.errors                      # error captured, not silently swallowed
    with open(out + ".errors") as fh:
        assert fh.read().strip()           # written to the sidecar


@pytest.mark.parametrize("spec,shapes", [
    ("ia,ia->", [(4, 3), (4, 3)]),
    ("ijab,ijab->", [(4, 4, 3, 3), (4, 4, 3, 3)]),
    ("ijcd,acQ,bdQ->abij", [(4, 4, 3, 3), (3, 3, 5), (3, 3, 5)]),
    ("ia,jb->ijab", [(4, 3), (4, 3)]),
])
def test_adapter_matches_numpy(spec, shapes):
    rng = np.random.default_rng(3)
    ops = [rng.standard_normal(s) for s in shapes]
    ad = EfContractionAdapter(oracle=True)
    assert np.allclose(ad(spec, *ops), np.einsum(spec, *ops))
