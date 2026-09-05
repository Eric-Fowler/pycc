"""Replay PyCC contraction signatures against ``ef`` and report coverage.

Two input modes:
  * captured JSONL (real shapes/dtypes/layouts from a live run under
    PYCC_EF_CAPTURE) — the faithful A0 replay;
  * a spec list with a uniform synthetic extent (``--extent``) — a dry run that
    exercises the real *index patterns* of the corpus without a live capture
    (useful where psi4 is unavailable). Layouts default to C plus one
    non-contiguous pass so the §7 non-contiguous target is still covered.

For each signature it builds representative operands (values random, layout
matching the record), computes the numpy reference, and runs the ef adapter in
oracle (``ef.evaluate``) and, optionally, runner (``ef.runner``) mode, comparing
with ``np.allclose``. Emits a per-signature status and a summary gap list.

Run examples:
  python -m devtools.ef_integration.replay --from-corpus --extent 3
  python -m devtools.ef_integration.replay --jsonl capture.jsonl --runner
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ._capture_api import layout_class


def make_operand(shape, dtype, layout, rng):
    """A random operand of the given shape/dtype whose contiguity class == layout."""
    shape = tuple(int(s) for s in shape)
    dt = np.dtype(dtype)

    def _rand(shp):
        a = rng.standard_normal(shp)
        if dt.kind == "c":
            a = a + 1j * rng.standard_normal(shp)
        return a.astype(dt)

    if layout == "F" and len(shape) >= 2:
        return np.asfortranarray(_rand(shape))
    if layout == "non-contiguous" and len(shape) >= 1 and all(s > 0 for s in shape):
        big = list(shape)
        big[0] = shape[0] * 2
        view = _rand(tuple(big))[::2]
        assert view.shape == shape and layout_class(view) == "non-contiguous"
        return view
    return np.ascontiguousarray(_rand(shape))  # C (also the fallback for degenerate shapes)


def _signatures_from_corpus(extent: int, layouts):
    from .static_corpus import inventory
    import pathlib
    inv = inventory(pathlib.Path(__file__).resolve().parents[2] / "pycc")
    for spec in inv["distinct_specs"]:
        if "->" not in spec:
            continue
        ins = spec.split("->")[0].split(",")
        shapes = [[extent] * len(term) for term in ins]
        yield {"clean_spec": spec, "shapes": shapes,
               "dtypes": ["<f8"] * len(ins), "layouts": [layouts[0]] * len(ins)}


def _signatures_from_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def replay(signatures, use_runner, extra_layout_pass, rng, capacity):
    from .adapter import EfContractionAdapter
    oracle = EfContractionAdapter(oracle=True)
    runner = EfContractionAdapter(capacity=capacity, device="cpu") if use_runner else None
    rows = []
    for sig in signatures:
        spec = sig["clean_spec"]
        layout_variants = [sig["layouts"]]
        if extra_layout_pass:
            layout_variants.append(["non-contiguous"] * len(sig["shapes"]))
        row = {"spec": spec, "oracle": "skip", "runner": "skip"}
        for layouts in layout_variants:
            ops = [make_operand(sig["shapes"][i], sig["dtypes"][i], layouts[i], rng)
                   for i in range(len(sig["shapes"]))]
            try:
                ref = np.einsum(spec, *ops)
            except Exception as e:
                row["oracle"] = row["runner"] = f"numpy-invalid: {type(e).__name__}"
                break
            for lane, ad in (("oracle", oracle), ("runner", runner)):
                if ad is None:
                    continue
                try:
                    got = ad(spec, *ops)
                    ok = np.allclose(np.asarray(got), ref)
                    status = "OK" if ok else "MISMATCH"
                except Exception as e:
                    status = f"ERROR {type(e).__name__}: {str(e)[:60]}"
                # keep the worst status across layout variants
                if row[lane] in ("skip", "OK") and status != "OK":
                    row[lane] = status
                elif row[lane] == "skip":
                    row[lane] = status
        rows.append(row)
    return rows


def summarize(rows, use_runner):
    def tally(lane):
        c = {}
        for r in rows:
            k = r[lane].split(":")[0].split(" ")[0]
            c[k] = c.get(k, 0) + 1
        return c
    print(f"replayed {len(rows)} distinct signatures")
    print(f"  ef.evaluate (oracle): {tally('oracle')}")
    if use_runner:
        print(f"  ef.runner           : {tally('runner')}")
    gaps = [r for r in rows if r["oracle"] not in ("OK",)
            or (use_runner and r["runner"] not in ("OK",))]
    nontrivial = [r for r in gaps if not r["oracle"].startswith("numpy-invalid")]
    if nontrivial:
        print(f"  gaps ({len(nontrivial)} specs ef handled differently than numpy):")
        for r in nontrivial[:40]:
            print(f"    {r['spec']:28} oracle={r['oracle']:22} runner={r['runner']}")
    else:
        print("  no ef-vs-numpy gaps among numpy-valid specs.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", help="captured signatures (real shapes/dtypes/layouts)")
    src.add_argument("--from-corpus", action="store_true", help="synthetic uniform-extent dry run")
    ap.add_argument("--extent", type=int, default=3, help="uniform index extent for --from-corpus")
    ap.add_argument("--runner", action="store_true", help="also run ef.runner (slower)")
    ap.add_argument("--capacity", type=int, default=1 << 30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    if args.from_corpus:
        sigs = list(_signatures_from_corpus(args.extent, ["C"]))
        extra = True   # add a non-contiguous pass for coverage
    else:
        sigs = list(_signatures_from_jsonl(args.jsonl))
        extra = False  # trust captured layouts
    rows = replay(sigs, args.runner, extra, rng, args.capacity)
    summarize(rows, args.runner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
